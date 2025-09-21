import json
import logging
import asyncio
from azure.servicebus import ServiceBusClient, ServiceBusMessage
from azure.servicebus.aio import ServiceBusClient as AsyncServiceBusClient
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class ServiceBusTopicAzure:
    """
    A utility class for sending and receiving messages to/from Azure Service Bus using a topic + subscription.

    This replaces the previous queue-based implementation (`ServiceBusQueueAzure`).
    """
    
    def __init__(self, connection_string: str, topic_name: str = "commandtopic", subscription_name: str = "rlcommandbustopic"):
        """
        Initialize the Service Bus Azure topic sender/receiver.
        
        Args:
            connection_string (str): Azure Service Bus connection string
            topic_name (str): Name of the topic to send/receive messages to/from
            subscription_name (str): Name of the subscription (required for receiving)
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
            
    def send_azure_result(self, response_data: Dict[str, Any], 
                         request_id: Optional[str] = None,
                         *,
                         wrap: bool = True,
                         message_type: Optional[str] = "azure_result") -> bool:
        """Send payload to the Service Bus topic.

        Args:
            response_data: Already JSON-serialisable payload
            request_id: Optional correlation id (mapped to AMQP message_id)
            wrap: When True, wrap data inside an envelope with type/timestamp/request_id
            message_type: Envelope type when wrap=True (default: azure_result)
        """
        try:
            if not self.client:
                logger.error("Service Bus client is not initialized. Use as context manager.")
                return False

            if wrap:
                message_payload: Dict[str, Any] = {
                    "type": message_type or "azure_result",
                    "timestamp": None,
                    "request_id": request_id,
                    "data": response_data,
                }
            else:
                message_payload = dict(response_data)
                if request_id is not None and "request_id" not in message_payload:
                    message_payload["request_id"] = request_id
                if message_type and "type" not in message_payload:
                    message_payload["type"] = message_type

            message_body = json.dumps(message_payload, ensure_ascii=False)
            message = ServiceBusMessage(body=message_body)

            if request_id:
                message.message_id = request_id

            # Topic sender (replaces queue sender)
            with self.client.get_topic_sender(topic_name=self.topic_name) as sender:
                sender.send_messages(message)

            logger.info(f"Message sent successfully to topic '{self.topic_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to send message to topic '{self.topic_name}': {str(e)}")
            return False

            
    async def receive_messages_async(self, max_message_count: int = 1, max_wait_time: int = 5) -> List[Dict[str, Any]]:
        """
        Receive messages from the Service Bus Azure subscription (asynchronous).
        
        Args:
            max_message_count (int): Maximum number of messages to receive
            max_wait_time (int): Maximum time to wait for messages in seconds
            
        Returns:
            List[Dict[str, Any]]: List of received messages (parsed payloads)
        """
        try:
            messages: List[Dict[str, Any]] = []
            async with AsyncServiceBusClient.from_connection_string(self.connection_string) as client:
                async with client.get_subscription_receiver(topic_name=self.topic_name, subscription_name=self.subscription_name) as receiver:
                    received_msgs = await receiver.receive_messages(
                        max_message_count=max_message_count, 
                        max_wait_time=max_wait_time
                    )
                    
                    for msg in received_msgs:
                        try:
                            message_data = json.loads(str(msg))
                            messages.append({
                                "data": message_data,
                                "message_id": msg.message_id,
                                "delivery_count": msg.delivery_count
                            })
                            await receiver.complete_message(msg)
                            logger.info(f"Message received and completed from subscription '{self.subscription_name}' on topic '{self.topic_name}'")
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse message from subscription: {str(e)}")
                            await receiver.dead_letter_message(msg)
            return messages
        except Exception as e:
            logger.error(f"Failed to receive messages from subscription '{self.subscription_name}' on topic '{self.topic_name}': {str(e)}")
            return []

