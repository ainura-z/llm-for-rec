import os
from typing import List
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
import numpy as np


class EmbeddingsFromFile(Embeddings):
    """
    Custom Embiddings class for loading pretrained embeddings from file

    Attributes:
        user_emb (np.array): The matrix of user embeddings of shape (n_users, emb_dim)
        item_emb (np.array): The matrix of item embeddings of shape (n_items, emb_dim)
        user_id_map (Dict[str, int]): The mapping from text user ids to internal numerical user ids
        item_id_map (Dict[str, int]): The mapping from text item ids to internal numerical item ids
    """
    def __init__(self, emb_file_path: str, *args, **kwargs):
        """
        Initialize embeddings from .npz file. 
        .npz file must contain following fields:
            'user_factors': np.array of user embeddings
            'item_factors': np.array of item embeddings
            'user_ids': ordered text user ids
            'item_ids': ordered text item ids 

        Args:
            emb_file_path (str): Path to embeddings .npz file
        """
        Embeddings.__init__(self, *args, **kwargs)

        if not os.path.exists(emb_file_path):
            raise ValueError("Wrong path to embeddings file")

        with np.load(emb_file_path) as emb_data:
            assert sorted(list(emb_data.keys())) == sorted(['user_factors', 'item_factors', 'user_ids', 'item_ids'])
            self.user_emb = emb_data['user_factors']
            self.item_emb = emb_data['item_factors']
            self.user_id_map = dict(zip(emb_data['user_ids'], range(len(emb_data['user_ids']))))
            self.item_id_map =  dict(zip(emb_data['item_ids'], range(len(emb_data['item_ids']))))


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        idxs = [self.item_id_map[text.split()[1] if len(text.split()) > 0 else text] for text in texts]
        return self.item_emb[idxs]

    def embed_query(self, text: str) -> List[float]:
        idx = self.user_id_map[text]
        return self.user_emb[idx]