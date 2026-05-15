import json
import logging
from confluent_kafka import Producer
# from aiokafka import AIOKafkaProducer
from pydantic import BaseModel

from src.ports.output.message_publisher import MessagePublisherPort

logger = logging.getLogger(__name__)

class ConfluentKafkaPublisherAdapter(MessagePublisherPort):
    """
    Concrete implementation of the MessagePublisherPort using aiokafka.
    """
    # *******
    def __init__(self, bootstrap_servers: str):
        """
        Stores configuration. Connection happens in the start() method.
        """
        try:
            self.producer = None
            self.bootstrap_servers = bootstrap_servers
            
        except Exception as e:
            logger.error(e)

    # *******
    async def start(self) -> None:
        """
        Initializes the async producer and connects to the broker.
        Must be called during application startup.
        """
        try:
            self.producer = Producer({
                    'bootstrap.servers': self.bootstrap_servers,
                    # Equivalent to acks='all'
                    'request.required.acks': 'all', 
                    'client.id': 'medulla-fovea-producer'
                })
            
            logger.info("Producer Started.")
        except Exception as e:
            logger.error(f"Failed to start Kafka producer: {e}", exc_info=True)
            raise
    

    # *******
    async def stop(self) -> None:
        """
        Gracefully shuts down the producer. 
        Must be called during application shutdown to prevent memory leaks.
        """
        if self.producer is None:
            return

        logger.info("Flushing remaining messages to Kafka...")
        self.producer.flush(timeout=5.0)
        logger.info("Kafka Producer stopped.")

    # *******
    async def publish(self, topic: str, message: BaseModel) -> None:
        """
        Serializes and publishes a Pydantic model to Kafka.
        """
        if self.producer is None:
            raise RuntimeError("Kafka producer has not been started. Call start() before publish().")

        def delivery_report(err, msg):
            if err is not None:
                logger.error(f"Message delivery failed to {msg.topic()}: {err}")
            else:
                logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")

        try:
            # model_dump_json() outputs a string, which encode to bytes
            payload = message.model_dump_json().encode('utf-8')
            
            # Hand the message to the background C-thread
            self.producer.produce(
                topic, 
                value=payload, 
                callback=delivery_report
            )
            
            # Trigger the background thread to evaluate callbacks
            self.producer.poll(0)
            
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            raise
        