import asyncio
import json
import logging
import os
from dotenv import load_dotenv
import sys
sys.path.append('..')
from servicebus_web import ServiceBusTopicWeb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def check_rewards():
    """
    Periodically poll the reward topic subscription for web rewards.
    """
    connection_string = os.getenv("SERVICE_BUS_CONNECTION_STRING")
    if not connection_string:
        logger.error("SERVICE_BUS_CONNECTION_STRING not found")
        return

    topic_name = os.getenv("REWARD_TOPIC_NAME", "rewardtopic")
    subscription_name = os.getenv("REWARD_SUBSCRIPTION_NAME", "rlcommandbustopic")

    # Create topic helper
    web_topic = ServiceBusTopicWeb(connection_string, topic_name=topic_name, subscription_name=subscription_name)

    logger.info(f"Polling reward topic '{topic_name}' subscription '{subscription_name}' every 5s...")

    while True:
        try:
            reward = await web_topic.receive_web_reward_async()
            if reward != {"message": "No rewards received"}:
                print(f"Reward received: {json.dumps(reward, indent=2, ensure_ascii=False)}")
            else:
                print("No new rewards")
        except Exception as e:
            logger.error(f"Error checking rewards: {str(e)}")
        await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(check_rewards())
