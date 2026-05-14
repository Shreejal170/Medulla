import json
import logging
from aiokafka import AIOKafkaProducer
from pydantic import BaseModel

from src.ports.output.message_publisher import MessagePublisherPort

logger = logging.getLogger(__name__)

class AioKafkaPublisherAdapter(MessagePublisherPort):
    """
    Concrete implementation of the MessagePublisherPort using aiokafka.
    """
    # *******
    def __init__(self, bootstrap_servers: str):
        """
        Stores configuration. Connection happens in the start() method.
        """
        self.bootstrap_servers = bootstrap_servers
        self.producer = None

    # *******
    async def start(self) -> None:
        """
        Initializes the async producer and connects to the broker.
        Must be called during application startup.
        """
        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'), # Automatically serialize Python dicts to UTF-8 encoded JSON bytes
                acks='all',  # Guarantee message is fully committed by the broker
                retry_backoff_ms=500, # Wait half a second before retrying on failure
            )
        except Exception as e:
            pass

    # *******
    async def stop(self) -> None:
        """
        Gracefully shuts down the producer. 
        Must be called during application shutdown to prevent memory leaks.
        """
        if self.producer:
            await self.producer.stop()
            logger.info("AIOKafkaProducer stopped cleanly.")

    # *******
    async def publish(self, topic: str, message: BaseModel):
        """
        Serializes and asynchronously publishes a Pydantic model to Kafka.
        """
        if not self.producer:
            raise RuntimeError("Producer has not been started. Call start() first.")
        
        try:
            # model_dump() converts the Pydantic schema to a standard dictionary
            payload = message.model_dump()

            # send_and_wait ensures we don't proceed until the broker acknowledges receipt
            await self.producer.send_and_wait(topic, value=payload)
            logger.debug(f"Message successfully published to topic: {topic}")

        except Exception as e:
            logger.error(f"Failed to publish message to {topic}: {e}")
            raise
        