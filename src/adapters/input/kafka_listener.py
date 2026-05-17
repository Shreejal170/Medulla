import json
import logging
import asyncio
from typing import Callable, Awaitable
from confluent_kafka import Consumer, KafkaError, KafkaException

from src.ports.input.message_consumer import MessageConsumerPort

logger = logging.getLogger(__name__)


class ConfluentKafkaListenerAdapter(MessageConsumerPort):
    """
    Concrete implementation of the MessageConsumerPort using confluent_kafka.
    Runs a non-blocking polling loop in the background.

    Attrs:
        bootstrap_servers: Kafka broker address (e.g. "localhost:9092")
        group_id: Consumer group ID
    """

    def __init__(self, bootstrap_servers: str, group_id: str):
        """Initializes the consumer with the given configuration. Connection happens in start().
        Args:
            bootstrap_servers: Kafka broker address (e.g. "localhost:9092")
            group_id: Consumer group ID
        Returns:
            None
        """

        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.consumer = Consumer(
            {
                "bootstrap.servers": bootstrap_servers,
                "group.id": group_id,
                "auto.offset.reset": "earliest",  # Read from the beginning if no previous offset exists
                "enable.auto.commit": False,  # manually commit after successful processing
            }
        )
        self.handler = None
        self.topic = None
        self._running = False
        self._consume_task = None

    def subscribe(self, topic: str, handler: Callable[[str], Awaitable[None]]) -> None:
        """
        Registers the topic and the Application Service handler.

        Args:
            topic: Kafka topic to subscribe to
            handler: Async function that takes a raw JSON string and processes it
        Returns:
            None
        """
        self.topic = topic
        self.handler = handler
        self.consumer.subscribe([self.topic])
        logger.info(f"Subscribed to topic: {self.topic}")

    async def start(self) -> None:
        """
        Starts the background polling loop.
        Args:
            None
        Returns:
            None
        """
        if not self.topic or not self.handler:
            raise RuntimeError("You must call subscribe() before calling start().")

        self._running = True
        # Launch the polling loop as a background asyncio task
        self._consume_task = asyncio.create_task(self._consume_loop())
        logger.info("Kafka Consumer background loop started.")

    async def stop(self) -> None:
        """
        Gracefully signals the loop to stop and closes the connection.

        Args:
            None
        Returns:
            None
        """
        self._running = False
        if self._consume_task:
            await self._consume_task

        self.consumer.close()
        logger.info("Kafka Consumer closed cleanly.")

    async def _consume_loop(self) -> None:
        """
        The internal loop that constantly checks for new messages without blocking Python.
        Args:
            None
        Returns:
            None
        """
        while self._running:
            # Poll for messages but timeout quickly (0.1s) so it doesn't freeze the loop
            msg = self.consumer.poll(timeout=0.1)

            if msg is None:
                # Yield control back to the main application so other tasks can run
                await asyncio.sleep(0)
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # Reached the end of the topic, just keep waiting
                    continue
                else:
                    logger.error(f"Kafka Consumer Error: {msg.error()}")
                    continue

            # For a real message! Decode it and hand it
            try:
                raw_json_string = msg.value().decode("utf-8")
                logger.debug(f"Received message from partition {msg.partition()}")

                # Await the application service handler
                await self.handler(raw_json_string)

                # Only commit the message if the handler finishes without crashing
                self.consumer.commit(msg)

            except Exception as e:
                logger.error(f"Failed to process message: {e}")
