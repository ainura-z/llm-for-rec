import torch
import typing as tp
from llm4rec.tasks.information_retrieval import OpenAIRetrievalRecommender
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.data.dataset import Dataset


class RecBoleRetrievalRecommender(OpenAIRetrievalRecommender, SequentialRecommender):
    """
    Adapter for RecBole of RetrievalRecommender model

    Attributes:
        fake_fn (torch.nn.Module): Fake function to simplify adaptation to optimized RecBole models with .parameters()
    """
    def __init__(self, config: tp.Dict[str, tp.Any], dataset: Dataset, *args, **kwargs):
        OpenAIRetrievalRecommender.__init__(self, *args, **kwargs)
        SequentialRecommender.__init__(self, config=config, dataset=dataset)
        self.fake_fn = torch.nn.Linear(1, 1)
