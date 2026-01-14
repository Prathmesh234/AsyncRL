"""
FastAPI server for DisTrainer.
Exposes HTTP endpoints for training control via a distributed command loop.
"""

import threading
import torch.distributed as dist
import queue
import time
import uuid
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from .trainer import Trainer, Config
from .mesh import is_main_rank

# Global items
trainer: Optional[Trainer] = None
app = FastAPI(title="DisTrainer", description="Distributed GRPO Trainer API")

# Command Queue Mechanism
# The HTTP thread (Rank 0 only) puts commands here.
# The Main Thread (Rank 0) reads them and broadcasts to all other ranks.
cmd_queue = queue.Queue()
cmd_results = {}  # {uuid: result_data}

class Command:
    TRAIN = "TRAIN"
    CHECKPOINT = "CHECKPOINT"
    LOAD = "LOAD"
    STOP = "STOP"

# --- Models ---

class TrainRequest(BaseModel):
    num_steps: int = 1

class TrainResponse(BaseModel):
    status: str
    steps_completed: int
    metrics: List[Dict[str, Any]]

class StatusResponse(BaseModel):
    step: int
    batches_available: int
    batches_processed: int
    checkpoints: List[int]
    model: str
    gpu_count: int

class CheckpointResponse(BaseModel):
    status: str
    step: int
    path: Optional[str] = None

class LoadCheckpointRequest(BaseModel):
    step: Optional[int] = None

def get_trainer() -> Trainer:
    """Get the trainer instance, raise if not initialized."""
    if trainer is None:
        raise HTTPException(status_code=503, detail="Trainer not initialized")
    return trainer


# --- Endpoints (Run on Rank 0 HTTP Thread only) ---

@app.post("/train", response_model=TrainResponse)
async def train(request: TrainRequest):
    """Queue a training job and wait for results."""
    if not is_main_rank():
        raise HTTPException(status_code=400, detail="Only Rank 0 can accept requests")
        
    cmd_id = str(uuid.uuid4())
    # Enqueue command for the Main Thread
    cmd_queue.put({
        "type": Command.TRAIN,
        "payload": request.num_steps,
        "id": cmd_id
    })
    
    # Block and wait for result from Main Thread
    while cmd_id not in cmd_results:
        time.sleep(0.01)
        
    result = cmd_results.pop(cmd_id)
    return TrainResponse(
        status="success",
        steps_completed=len(result),
        metrics=result
    )

@app.post("/checkpoint", response_model=CheckpointResponse)
async def checkpoint():
    """Trigger a checkpoint save."""
    cmd_id = str(uuid.uuid4())
    cmd_queue.put({
        "type": Command.CHECKPOINT,
        "id": cmd_id
    })
    while cmd_id not in cmd_results:
        time.sleep(0.01)
    
    result = cmd_results.pop(cmd_id)
    return CheckpointResponse(status="saved", step=result["step"], path=result["path"])

@app.post("/load_checkpoint", response_model=CheckpointResponse)
async def load_checkpoint(request: LoadCheckpointRequest):
    """Trigger a checkpoint load."""
    cmd_id = str(uuid.uuid4())
    cmd_queue.put({
        "type": Command.LOAD,
        "payload": request.step,
        "id": cmd_id
    })
    while cmd_id not in cmd_results:
        time.sleep(0.01)
        
    result = cmd_results.pop(cmd_id)
    return CheckpointResponse(status="loaded", step=result)

@app.get("/status", response_model=StatusResponse)
async def status():
    """Get status directly from local trainer (safe on Rank 0)."""
    t = get_trainer()
    s = t.get_status()
    # No need to broadcast for read-only status
    return StatusResponse(
        step=s["step"],
        batches_available=s["batches_available"],
        batches_processed=s["batches_processed"],
        checkpoints=s["checkpoints"],
        model=s["model"],
        gpu_count=s["gpu_count"]
    )

@app.get("/health")
async def health():
    return {"status": "healthy", "trainer_ready": trainer is not None}


# --- Core Logic ---

def init_trainer(config_path: str, use_base_model: bool = False):
    """Initialize the global trainer instance."""
    global trainer
    config = Config.from_toml(config_path)
    trainer = Trainer(config, use_base_model=use_base_model)
    return trainer

def distributed_worker(trainer_instance: Trainer, auto_train: bool = True, poll_interval: float = 5.0):
    """
    Main Loop for ALL Ranks (0, 1, ...).
    Keeps everyone in sync by broadcasting commands from Rank 0.
    
    When auto_train=True (default), automatically:
    1. Polls for new batch files
    2. Trains when new data is available
    3. Saves checkpoint after training
    4. Deletes processed batch file
    
    Args:
        trainer_instance: The Trainer instance
        auto_train: If True, automatically train when new batches arrive
        poll_interval: Seconds between polling for new batches (when idle)
    """
    last_poll_time = 0
    
    while True:
        # 1. Rank 0 checks for commands OR polls for auto-training
        cmd_data = None
        if is_main_rank():
            # First check for explicit HTTP commands
            if not cmd_queue.empty():
                cmd_data = cmd_queue.get()
            # If auto-train enabled and no explicit command, check for new batches
            elif auto_train:
                current_time = time.time()
                if current_time - last_poll_time >= poll_interval:
                    last_poll_time = current_time
                    
                    # Check if there are new batches available
                    if trainer_instance.data_loader.count_available() > 0:
                        # Auto-generate a TRAIN command
                        cmd_data = {
                            "type": Command.TRAIN,
                            "payload": 1,  # Process 1 batch at a time
                            "id": f"auto-{uuid.uuid4().hex[:8]}",
                            "auto": True  # Mark as auto-generated
                        }
                        print(f"New batch detected! Starting auto-training...")
        
        # 2. Broadcast the decision (Command or None) to everyone
        object_list = [cmd_data]
        dist.broadcast_object_list(object_list, src=0)
        cmd_data = object_list[0]
        
        # 3. Process the command
        if cmd_data is None:
            time.sleep(0.1)  # Avoid burning CPU when idle
            continue
            
        cmd_type = cmd_data.get("type")
        cmd_id = cmd_data.get("id")
        is_auto = cmd_data.get("auto", False)
        
        # --- Handle TRAIN ---
        if cmd_type == Command.TRAIN:
            num_steps = cmd_data["payload"]
            metrics_list = []
            batch_files_to_delete = []
            
            # Everyone runs the training steps in lockstep
            for step_i in range(num_steps):
                # Track which batch file we're about to process (for deletion)
                if is_main_rank():
                    next_batch = trainer_instance.data_loader.peek_next_batch_file()
                    if next_batch:
                        batch_files_to_delete.append(next_batch)
                
                m = trainer_instance.train_step()
                metrics_list.append(m)
                
                # If no data, everyone stops together
                if m.get("status") == "no_data":
                    if is_main_rank() and is_auto:
                        print("⏳ No more batches available. Waiting for new data...")
                    break
                    
                # Log progress
                if is_main_rank():
                    loss = m.get("loss", 0)
                    avg_reward = m.get("avg_reward", 0)
                    step = m.get("step", 0)
                    print(f"✅ Step {step}: loss={loss:.4f}, avg_reward={avg_reward:.3f}")
            
            # Auto-training: delete processed batch files (checkpoint is handled by trainer interval)
            if is_auto and metrics_list and metrics_list[-1].get("status") != "no_data":
                # Note: Checkpoint is saved automatically by trainer.train_step() 
                # based on checkpoint.save_interval config (e.g., every 10 steps)
                # We DON'T save a checkpoint after every batch - that's too frequent!
                
                # Delete processed batch files (only on rank 0)
                if is_main_rank():
                    for batch_file in batch_files_to_delete:
                        try:
                            if batch_file.exists():
                                batch_file.unlink()
                                print(f" Deleted processed batch: {batch_file.name}")
                        except Exception as e:
                            print(f"⚠️ Failed to delete {batch_file}: {e}")
            
            # Only Rank 0 reports results back to HTTP (for manual /train calls)
            if is_main_rank() and not is_auto:
                cmd_results[cmd_id] = metrics_list
                
        # --- Handle CHECKPOINT ---
        elif cmd_type == Command.CHECKPOINT:
            path = trainer_instance.save_checkpoint()
            if is_main_rank():
                cmd_results[cmd_id] = {"path": path, "step": trainer_instance.step}
                
        # --- Handle LOAD ---
        elif cmd_type == Command.LOAD:
            step = cmd_data["payload"]
            loaded_step = trainer_instance.load_checkpoint(step)
            if is_main_rank():
                cmd_results[cmd_id] = loaded_step
        
        # --- Handle STOP ---
        elif cmd_type == Command.STOP:
            break


def run_server(config_path: str, host: str = "0.0.0.0", port: int = 8000, use_base_model: bool = False):
    """Start the distributed system."""
    import uvicorn

    # 1. Initialize Trainer (ALL Ranks)
    init_trainer(config_path, use_base_model=use_base_model)
    t = get_trainer()
    
    # 2. Rank 0 starts HTTP Server in Background Thread
    if is_main_rank():
        server_thread = threading.Thread(
            target=uvicorn.run,
            args=(app,),
            kwargs={"host": host, "port": port},
            daemon=True
        )
        server_thread.start()
        print(f"Server started on port {port}. Waiting for commands...")
    
    # 3. ALL Ranks enter the Command Loop
    # This blocks the Main Thread until shutdown
    distributed_worker(t)
