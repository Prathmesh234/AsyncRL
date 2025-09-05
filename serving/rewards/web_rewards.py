import asyncio
import json
import logging
import os
from dotenv import load_dotenv
import sys
sys.path.append('..')
from servicebus_web import ServiceBusQueueWeb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def check_rewards():
    """
    Simple script to check for web rewards every 5 seconds.
    """
    connection_string = os.getenv("SERVICE_BUS_CONNECTION_STRING")
    if not connection_string:
        logger.error("SERVICE_BUS_CONNECTION_STRING not found")
        return
    
    # Create ServiceBusQueueWeb instance for rewardqueue
    web_queue = ServiceBusQueueWeb(connection_string, "rewardqueue")
    
    logger.info("Starting to check for rewards every 5 seconds...")
    
    while True:
        try:
            # Use the new simple function
            reward = await web_queue.receive_web_reward_async()
            
            if reward != {"message": "No rewards received"}:
                print(f"Reward received: {json.dumps(reward, indent=2, ensure_ascii=False)}")
            else:
                print("No new rewards")
                
        except Exception as e:
            logger.error(f"Error checking rewards: {str(e)}")
            
        await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(check_rewards())
