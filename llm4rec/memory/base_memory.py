from abc import ABC, abstractmethod
import typing as tp

class BaseMemory(ABC):
    def __init__(self, *args, **kwargs):
        self.memory_store = {}

    @abstractmethod
    def update(self, id, data, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, id, *args, **kwargs):
        raise NotImplementedError

    def clear(self):
        self.memory_store.clear()

    def __len__(self):
        return len(self.memory_store)

    def __getitem__(self, idx):
        return self.memory_store[idx]
        
    @property
    def get_memory(self) -> tp.Dict[tp.Any, tp.Any]:
        """
        Return memory.

        Returns:
            tp.Dict[tp.Any, tp.Any]: The memory.
        """
        return self.memory_store