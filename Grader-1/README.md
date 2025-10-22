## Running with Docker

1. Build the image (from the `Grader-1` directory):
   ```bash
   docker build -t grader-1 .
   ```

2. Run the container with both GPUs and your `.env` file:
   ```bash
   docker run --gpus "device=0,1" --env-file /path/to/.env grader-1
   ```

The entrypoint loads `.env`, fixes prefill on GPU 0, decode on GPU 1, and launches `scripts/start_all.sh` to start every service.
