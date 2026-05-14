from abc import ABC, abstractmethod
from typing import Callable, Awaitable

class MessageConsumerPort(ABC):
    """
    Abstract interface for consuming messages asynchronously from a message broker.
    """

    @abstractmethod
    async def start(self) -> None:
        """
        Connects to the broker and begins the continuous polling loop.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        shuts down the consumer and disconnects from the broker.
        """
        pass

    @abstractmethod
    def subscribe(self, topic: str, handler: Callable[[str], Awaitable[None]]) -> None:
        """
        Registers an application service function to process incoming messages.
        
        Args:
            topic (str): The destination topic name (e.g., 'frames-ready-for-ai').
            handler (Callable): An async function that accepts the raw JSON string 
                                from the broker and processes it.
        """
        pass