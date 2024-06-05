import typing as tp
from abc import ABCMeta, abstractmethod


class PipelineBase(metaclass=ABCMeta):
    """Base Pipeline class.
    
    Warning: This class should not be used directly.
    Use derived classes instead."""

    def __init__(self, tasks: tp.List[tp.Callable], *args: tp.Any, verbose: bool = True, **kwargs: tp.Any) -> None:
        self.tasks = tasks
        self.verbose = verbose

    @abstractmethod
    def recommend(self, data: tp.Any, *args: tp.Any, **kwargs: tp.Any):
        raise NotImplementedError