import typing as tp
from llm4rec.memory.base_memory import BaseMemory


class ItemAugmentation:
    """
    A task that augments items with new data from external knowledge
    """

    def __init__(self, item_memory: BaseMemory) -> None:
        """
        Initializes ItemAugmentation

        Args:
            item_memory: ItemMemory storage with additional information about each item
        """
        super().__init__()
        self.memory = item_memory

    def transform(self, prev_interactions: tp.List[str]) -> tp.Dict[str, str]:
        """
        For each item id in prev_interactions extract additional information
        from memory.

        Args:
            prev_interactions (List[str]): List of item ids

        Returns:
            dict[str, str]: dictionary mapping from item ids to corresponding text data
        """
        prev_interactions_info = []

        for item_id in prev_interactions:
            additional_text_data = self.memory.retrieve(item_id)
            prev_interactions_info.append(additional_text_data)
        augmented_prev_interactions = dict(
            zip(prev_interactions, prev_interactions_info)
        )
        return augmented_prev_interactions
