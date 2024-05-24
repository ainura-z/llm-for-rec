import typing as tp
from llm4rec.memory.base_memory import BaseMemory


class UserAugmentation:
    """
    Task that adds user profile info
    """

    def __init__(self, user_memory: BaseMemory):
        """
        Initializes UserAugmentation

        Args:
            user_memory: UserMemory instance with all available information about user
        """
        super().__init__()
        self.memory = user_memory

    def transform(self, user_token_id: str):
        """
        Generates user profile
        """
        user_profile = self.memory.construct_user_profile(user_token_id)
        return user_profile
