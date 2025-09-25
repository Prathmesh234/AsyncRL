import json
import logging
import asyncio
from typing import Dict, Any, Optional

from azure.servicebus import ServiceBusClient, ServiceBusMessage
from azure.servicebus.aio import ServiceBusClient as AsyncServiceBusClient

logger = logging.getLogger(__name__)

class ServiceBusTopic:
    """
    A utility class for sending and receiving messages to/from Azure Service Bus topics/subscriptions.
    """
    
    def __init__(self, connection_string: str, topic_name: str = "commandtopic", subscription_name: str = "websubscription"):
        """
        Initialize the Service Bus web topic sender/receiver.
        
        Args:
            connection_string (str): Azure Service Bus connection string
            topic_name (str): Name of the topic to send/receive messages to/from
            subscription_name (str): Name of the subscription used when receiving
        """
        self.connection_string = connection_string
        self.topic_name = topic_name
        self.subscription_name = subscription_name
        self.client = None
        
    def __enter__(self):
        """Context manager entry."""
        self.client = ServiceBusClient.from_connection_string(self.connection_string)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.client:
            self.client.close()
            
    def send_web_result(self, response_data: Dict[str, Any],
                       message_id: Optional[str] = None,
                       *,
                       wrap: bool = True,
                       message_type: Optional[str] = "web_result") -> bool:
        """Send payload to the Service Bus topic.

        Args:
            response_data: The payload to send (already JSON-serialisable).
            message_id: Optional AMQP message identifier for correlation/telemetry.
            wrap: When True, embed payload in the legacy `{type:"web_result", data:...}` envelope.
            message_type: Override the outer `type` field for wrapped payloads; ignored if wrap=False
                unless the payload omits a `type` key (then it is injected).
        """
        try:
            if not self.client:
                logger.error("Service Bus client is not initialized. Use as context manager.")
                return False

            if wrap:
                message_payload: Dict[str, Any] = {
                    "type": message_type or "web_result",
                    "timestamp": None,  # Will be set by Service Bus
                    "request_id": message_id,
                    "data": response_data,
                }
            else:
                message_payload = dict(response_data)

            message_body = json.dumps(message_payload, ensure_ascii=False)

            message = ServiceBusMessage(body=message_body)

            if message_id:
                message.message_id = message_id

            # Topic sender now
            with self.client.get_topic_sender(topic_name=self.topic_name) as sender:
                sender.send_messages(message)

            logger.info(f"Message sent successfully to topic '{self.topic_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to send message to topic '{self.topic_name}': {str(e)}")
            return False

            
    async def receive_messages_async(self, max_message_count: int = 1, max_wait_time: int = 5) -> Dict[str, Any]:
        """
        Receive messages from the Service Bus subscription (asynchronous) - simplified like read_command.
        
        Args:
            max_message_count (int): Maximum number of messages to receive
            max_wait_time (int): Maximum time to wait for messages in seconds
            
        Returns:
            Dict[str, Any]: The received message data or error message
        """
        try:
            async with AsyncServiceBusClient.from_connection_string(self.connection_string) as client:
                async with client.get_subscription_receiver(topic_name=self.topic_name, subscription_name=self.subscription_name) as receiver:
                    received_msgs = await receiver.receive_messages(
                        max_message_count=max_message_count, 
                        max_wait_time=max_wait_time
                    )
                    
                    for msg in received_msgs:
                        try:
                            # Parse message body - simple like read_command
                            message_data = json.loads(str(msg))
                            
                            # Complete the message
                            await receiver.complete_message(msg)
                            logger.info(f"Message received and completed from subscription '{self.subscription_name}' on topic '{self.topic_name}'")
                            
                            # Return just the data object when available; otherwise return payload as-is
                            payload = {
                                "data": message_data,
                                "message_id": msg.message_id,
                                "delivery_count": msg.delivery_count
                            }
                            
                            if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
                                return payload["data"]
                            else:
                                return payload
                            
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse message from subscription: {str(e)}")
                            await receiver.dead_letter_message(msg)
                            
        except Exception as e:
            logger.error(f"Failed to receive messages from subscription '{self.subscription_name}' on topic '{self.topic_name}': {str(e)}")
            
        # Return default message if no messages received or error occurred
        return {"message": "No messages received"}
    
    async def receive_web_reward_async(self) -> Dict[str, Any]:
        """
        Simple function to receive web rewards - like read_command.
        
        Returns:
            Dict[str, Any]: The received reward data or default message
        """
        try:
            async with AsyncServiceBusClient.from_connection_string(self.connection_string) as client:
                async with client.get_subscription_receiver(topic_name=self.topic_name, subscription_name=self.subscription_name) as receiver:
                    received_msgs = await receiver.receive_messages(
                        max_message_count=1, 
                        max_wait_time=5
                    )
                    
                    for msg in received_msgs:
                        try:
                            # Parse message body
                            message_data = json.loads(str(msg))
                            
                            # Complete the message
                            await receiver.complete_message(msg)
                            logger.info(f"Web reward received from subscription '{self.subscription_name}' on topic '{self.topic_name}'")
                            
                            # Create payload like read_command
                            payload = {
                                "data": message_data,
                                "message_id": msg.message_id,
                                "delivery_count": msg.delivery_count
                            }
                            
                            # Return just the data object when available; otherwise return payload as-is
                            if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
                                return payload["data"]
                            else:
                                return payload
                            
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse reward message: {str(e)}")
                            await receiver.dead_letter_message(msg)
                            
        except Exception as e:
            logger.error(f"Failed to receive web rewards from subscription '{self.subscription_name}' on topic '{self.topic_name}': {str(e)}")
            
        # Return default message like read_command does
        return {"message": "No rewards received"}

