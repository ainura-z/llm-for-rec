from recbole.trainer import Trainer
from recbole.utils import EvaluatorType, set_color
from recbole.model.abstract_recommender import AbstractRecommender
from recbole.data.dataloader import AbstractDataLoader
from tqdm import tqdm
import torch
import numpy as np
import typing as tp


class PipelineTrainer(Trainer):
    """
    A tool for running training and evaluation of pipeline
    """
    def __init__(
        self, config: tp.Dict[str, tp.Any], model: AbstractRecommender):
        super().__init__(config, model)

    @torch.no_grad()
    def evaluate(
        self, eval_data: AbstractDataLoader, show_progress: bool = False
    ) -> tp.OrderedDict[str, float]:
        """
        Run evaluation of pipeline on test dataset.

        Args:
            eval_data (AbstractDataLoader): DataLoader from RecBole with test data.
            show_progress (bool): Add tqdm logging to dataset iteration process
        """
        self.model.eval()
        if self.config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data._dataset.item_num
        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )

        num_sample = 0
        for batched_data in iter_data:
            num_sample += len(batched_data)
            interaction, history_index, positive_u, positive_i = batched_data
            batch_size = len(interaction["user_id"])

            scores = torch.full((batch_size, self.model.n_items), -10000.0)

            for inter_idx in range(batch_size):
                user_id = interaction[inter_idx]["user_id"]
                history_ids = interaction[inter_idx]["item_id_list"]
                history_length = min(
                    self.config["MAX_ITEM_LIST_LENGTH"],
                    interaction[inter_idx]["item_length"],
                )
                history_names = eval_data.dataset.id2text(history_ids[:history_length])
                history_item_ids = eval_data.dataset.id2token("item_id", history_ids[:history_length])
                prev_interactions = dict(zip(history_item_ids, history_names))

                user_profile = eval_data.dataset.id2token("user_id", user_id)

                # model part
                candidates = self.model.recommend(
                    user_profile=user_profile,
                    prev_interactions=prev_interactions,
                    top_k=self.config['search_kwargs']['k']
                )
                candidate_ids = eval_data.dataset.token2id("item_id", candidates)

                # scores for metrics
                for i, id in enumerate(candidate_ids):
                    scores[inter_idx, id] = max(
                        len(candidate_ids) - i, scores[inter_idx, id]
                    )

            scores = scores.view(-1, self.tot_item_num)
            scores[:, 0] = -np.inf
            self.eval_collector.eval_batch_collect(
                scores, interaction, positive_u, positive_i
            )

        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        if not self.config["single_spec"]:
            result = self._map_reduce(result, num_sample)
        self.wandblogger.log_eval_metrics(result, head="eval")
        return result