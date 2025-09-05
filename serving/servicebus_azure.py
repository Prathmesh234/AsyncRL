import json
import logging
import asyncio
from azure.servicebus import ServiceBusClient, ServiceBusMessage
from azure.servicebus.aio import ServiceBusClient as AsyncServiceBusClient
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class ServiceBusQueueAzure:
    """
    A utility class for sending and receiving messages to/from Azure Service Bus azure queue.
    """
    
    def __init__(self, connection_string: str, queue_name: str = "azurequeue"):
        """
        Initialize the Service Bus azure queue sender/receiver.
        
        Args:
            connection_string (str): Azure Service Bus connection string
            queue_name (str): Name of the queue to send/receive messages to/from
        """
        self.connection_string = connection_string
        self.queue_name = queue_name
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
                         request_id: Optional[str] = None) -> bool:
        """
        Send an azure result to the Service Bus queue with proper formatting.
        
        Args:
            response_data (Dict[str, Any]): The azure response data
            request_id (Optional[str]): Optional request ID for tracking
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        try:
            if not self.client:
                logger.error("Service Bus client is not initialized. Use as context manager.")
                return False
            
            # Add metadata to the message
            message_payload = {
                "type": "azure_result",
                "timestamp": None,  # Will be set by Service Bus
                "request_id": request_id,
                "data": response_data
            }
            
            # Convert the data to JSON string
            message_body = json.dumps(message_payload, ensure_ascii=False)
            
            # Create the Service Bus message
            message = ServiceBusMessage(body=message_body)
            
            # Set optional properties
            if request_id:
                message.message_id = request_id
                
            # Send the message
            with self.client.get_queue_sender(queue_name=self.queue_name) as sender:
                sender.send_messages(message)
                
            logger.info(f"Azure result sent successfully to queue '{self.queue_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send azure result to queue '{self.queue_name}': {str(e)}")
            return False

            
    async def receive_messages_async(self, max_message_count: int = 1, max_wait_time: int = 5) -> List[Dict[str, Any]]:
        """
        Receive messages from the Service Bus azure queue (asynchronous).
        
        Args:
            max_message_count (int): Maximum number of messages to receive
            max_wait_time (int): Maximum time to wait for messages in seconds
            
        Returns:
            List[Dict[str, Any]]: List of received messages
        """
        try:
            messages = []
            async with AsyncServiceBusClient.from_connection_string(self.connection_string) as client:
                async with client.get_queue_receiver(queue_name=self.queue_name) as receiver:
                    received_msgs = await receiver.receive_messages(
                        max_message_count=max_message_count, 
                        max_wait_time=max_wait_time
                    )
                    
                    for msg in received_msgs:
                        try:
                            # Parse message body
                            message_data = json.loads(str(msg))
                            messages.append({
                                "data": message_data,
                                "message_id": msg.message_id,
                                "delivery_count": msg.delivery_count
                            })
                            
                            # Complete the message
                            await receiver.complete_message(msg)
                            logger.info(f"Message received and completed from azure queue '{self.queue_name}'")
                            
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse message from azure queue: {str(e)}")
                            await receiver.dead_letter_message(msg)
                            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to receive messages from azure queue '{self.queue_name}': {str(e)}")
            return []

