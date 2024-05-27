from recbole.data.interaction import Interaction
from llm4rec.tasks.base_recommender import Recommender
import torch
import warnings
import typing as tp


class GeneralRecBoleModelWrapper(Recommender):
    def __init__(self, model, config, dataset, top_k, **kwargs):
        model_config = config.copy()
        model_config.update(**kwargs)
        self.model = model(model_config, dataset)
        if 'k' in kwargs:
            # warning that top_k and k should be the same value
            if top_k != kwargs['k']:
                warnings.warn("Top_k parameter should be the same as k parameter in model. " 
                              + f"Got top_k={top_k} and k={kwargs['k']}. Automatically setting top_k to k value.")
            top_k = kwargs['k']
        self.top_k = top_k

        self.user_token2id = lambda x: dataset.token2id(self.model.USER_ID, x)
        self.item_id2token = lambda x: dataset.id2token(self.model.ITEM_ID, x)

    def recommend(self, user_token_id: str) -> tp.List[str]:
        #self.model.k = top_k
        user_id = self.user_token2id(user_token_id)
        new_inter = {
            self.model.USER_ID: torch.tensor(user_id)
        }
        scores = self.model.full_sort_predict(Interaction(new_inter))
        _, top_indices = torch.topk(scores, k=self.top_k, largest=True, sorted=True)

        # Convert item indices to item tokens
        recommended_item_tokens = list(self.item_id2token(top_indices.cpu().numpy().flatten()))

        return recommended_item_tokens
        
        
class SequentialRecBoleModelWrapper(Recommender):
    def __init__(self, model, config, dataset, top_k, **kwargs):
        model_config = config.copy()
        model_config.update(**kwargs)
        self.model = model(model_config, dataset)
        self.top_k = top_k

        self.item_id2token = lambda x: dataset.id2token(self.model.ITEM_ID, x)
        self.item_token2id = lambda x: dataset.token2id(self.model.ITEM_ID, x)


    def recommend(self, prev_interactions: tp.List[str], candidates: tp.List[str] = None) -> tp.List[str]:
        prev_interaction_ids = [self.item_token2id(token) for token in prev_interactions]
        new_inter = {
            self.model.ITEM_SEQ: torch.tensor(prev_interaction_ids).unsqueeze(0),
            self.model.ITEM_SEQ_LEN: torch.tensor(len(prev_interactions))
        }
        scores = self.model.full_sort_predict(Interaction(new_inter)).squeeze()

        if candidates is not None:
            candidate_ids = [self.item_token2id(token) for token in candidates]
            scores = scores[candidate_ids]

        _, top_indices = torch.topk(scores, k=self.top_k, largest=True, sorted=True)
        recommended_item_tokens = [candidates[idx] for idx in list(top_indices.cpu().numpy().flatten())]
        return recommended_item_tokens