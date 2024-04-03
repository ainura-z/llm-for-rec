import torch
import typing as tp
from llm4rec.pipelines.pipeline import Pipeline
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.data.dataset import Dataset


class RecBolePipelineRecommender(Pipeline, SequentialRecommender):
    """
    Adapter for RecBole of RetrievalRecommender model

    Attributes:
        fake_fn (torch.nn.Module): Fake function to simplify adaptation to optimized RecBole models with .parameters()
    """
    def __init__(self, config: tp.Dict[str, tp.Any], dataset: Dataset, *args, **kwargs):
        Pipeline.__init__(self, *args, **kwargs)
        SequentialRecommender.__init__(self, config=config, dataset=dataset)
        self.fake_fn = torch.nn.Linear(1, 1)