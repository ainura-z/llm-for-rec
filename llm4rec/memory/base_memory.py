from abc import ABC, abstractmethod
import typing as tp
import json

class BaseMemory(ABC):
    """
    Base class for memory.

    Parameters:
        memory_store (Dict): The memory.
    """
    def __init__(self, *args, **kwargs) -> None:
        self.memory_store = {}

    @abstractmethod
    def update(self, id: str, data: tp.Any, *args, **kwargs):
        """
        Update values in memory
        
        Args:
            id (str): ID of memory_store
            data (tp.Any): Value to store in memory
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, id: str, *args, **kwargs):
        """
        Retrieve values from memory.
        """
        raise NotImplementedError

    def clear(self) -> None:
        self.memory_store.clear()

    def __len__(self) -> int:
        return len(self.memory_store)

    def __getitem__(self, idx: str) -> tp.Any:
        return self.memory_store[idx]
        
    @property
    def get_memory(self) -> tp.Dict[str, tp.Any]:
        """
        Return memory.

        Returns:
            tp.Dict[tp.Any, tp.Any]: The memory.
        """
        return self.memory_store
        
    def save(self, filename: str)  -> None:
        """
        Save memory values to json file.
        
        Args:
            filename (str): Complete file path and file name ending with extention .json
        """
        assert filename.split('.')[-1] == 'json'
        
        with open(filename, 'w') as f:
            json.dump(self.memory_store, f)

    def load(self, filename):
        """
        Load memory values from json file.
        
        Args:
            filename (str): Complete file path and file name ending with extention .json
        """
        with open(filename) as f:
            self.memory_store = json.load(f)