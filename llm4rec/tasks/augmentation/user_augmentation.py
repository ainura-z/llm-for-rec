import typing as tp
from llm4rec.memory.base_memory import BaseMemory

class UserAugmentation:
    def __init__(self, user_memory: BaseMemory):
        super().__init__()
        self.memory = user_memory

    def transform(self, user_token_id: str):
        user_profile = self.memory.construct_user_profile(user_token_id)
        return user_profile