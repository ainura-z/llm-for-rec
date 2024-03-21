import typing as tp
from abc import ABCMeta, abstractmethod


class Recommender(metaclass=ABCMeta):
    """Base class for recommender model."""

    @abstractmethod
    def recommend(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        """Creating recommendations.

        Returns:
            Any: Recommended data.
        """
        raise NotImplementedError