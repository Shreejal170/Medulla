from abc import ABC, abstractmethod
from pydantic import BaseModel

class MessagePublisherPort(ABC):
    """
    Abstract interface for publishing messages asynchronously to a message broker.
    """
    
    @abstractmethod
    async def publish(self, topic: str, message: BaseModel) -> None:
        """
        Serializes and publishes a Pydantic model to a specific topic asynchronously.
        
        Args:
            topic (str): The destination topic name.
            message (BaseModel): The domain event or payload to send.
        """
        pass