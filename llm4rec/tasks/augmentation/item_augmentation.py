import typing as tp
from llm4rec.memory.base_memory import BaseMemory

class ItemAugmentation:
    def __init__(self, item_memory: BaseMemory):
        super().__init__()
        self.memory = item_memory

    def transform(self, prev_interactions: tp.List[str]):
        prev_interactions_info = []

        for item_id in prev_interactions:
            additional_text_data = self.memory.retrieve(item_id)
            prev_interactions_info.append(additional_text_data)

        return dict(zip(prev_interactions, prev_interactions_info))