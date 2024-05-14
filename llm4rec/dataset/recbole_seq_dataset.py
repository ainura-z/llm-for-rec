import numpy as np
import os.path as osp
from recbole.data.dataset import SequentialDataset
import typing as tp


class RecboleSeqDataset(SequentialDataset):
    """
    Dataset that returns user_id, previous interaction history and next item_id

    Attributes:
        data_path (str): The path to dataset files.
        dataset_name (str): The name of the dataset.
        id_token (List[str]): The mapping from internal numerical item ids to item ids from dataset file.
        preprocess_text_fn (Callable): The function to transform text feature of an item.
        item_text (Dict[int, str]): The mapping from internal numerical id of item to items text feature.
    """
    def __init__(
        self, config: tp.Dict[str, tp.Any], preprocess_text_fn: tp.Callable = None
    ):
        """
        Initializes the PipelineDataset.

        Args:
            config (str): The config file from RecBole.
            preprocess_text_fn (Optional[Callable]]): The function to apply to specified text feature of an item.
        """
        super().__init__(config)
        self.data_path = config["data_path"]
        self.dataset_name = config["dataset"]
        self.item_id_token = self.field2id_token["item_id"]
        self.user_id_token = self.field2id_token["user_id"]
        self.preprocess_text_fn = preprocess_text_fn
        self.user_text = self.load_user_text()
        self.item_text = self.load_item_text()

    def load_user_text(self) -> tp.List[str]:
        # from internal ids to text
        token_text = {}
        user_text = np.full(len(self.user_id_token), "", dtype=object)
        user_text[0] = "[PAD]"
        user_file_path = osp.join(self.data_path, f"{self.dataset_name}.user")
        
        if not osp.exists(user_file_path):
            self.logger.info(
                "Dataset seem to have no information about users."
            )
            return user_text
        
        # token id to text mapping
        with open(user_file_path, "r", encoding="utf-8") as file:
            col_names = file.readline().strip().split("\t")
            col_names = [col.split(":")[0] for col in col_names]
            
            text_col_idx = list(range(1, len(col_names)))
            #print(col_names)
            
            for line in file:
                description = line.strip().split("\t")
                user_id = description[0]
                text = "; ".join([f'{col_names[col_idx]}: {description[col_idx]}' for col_idx in text_col_idx])
                #print(user_id, text)
                token_text[user_id] = text

        # internal id to text mapping
        for i, token in enumerate(self.user_id_token):
            if token == "[PAD]":
                continue
            raw_text = token_text[token]
            #if self.preprocess_text_fn:
            #    raw_text = self.preprocess_text_fn(raw_text)
            user_text[i] = raw_text
        return user_text


    def load_item_text(self) -> tp.List[str]:
        # from internal ids to text
        token_text = {}
        item_text = np.full(len(self.item_id_token), "", dtype=object)
        item_text[0] = "[PAD]"
        item_file_path = osp.join(self.data_path, f"{self.dataset_name}.item")

        # token id to text mapping
        with open(item_file_path, "r", encoding="utf-8") as file:
            col_names = file.readline().strip().split("\t")
            col_names = [col.split(":")[0] for col in col_names]
            
            if type(self.config["text_col"]) == type(list):
                text_col_idx = [col_names.index(self.config["text_col"])]
            else:
                text_col_idx = [col_names.index(col_name) for col_name in self.config["text_col"]]

            for line in file:
                description = line.strip().split("\t")
                item_id = description[0]
                text = "; ".join([f'{col_names[col_idx]}: {description[col_idx]}' for col_idx in text_col_idx])
                token_text[item_id] = text

        # internal id to text mapping
        for i, token in enumerate(self.item_id_token):
            if token == "[PAD]":
                continue
            raw_text = token_text[token]
            if self.preprocess_text_fn:
                raw_text = self.preprocess_text_fn(raw_text)
            item_text[i] = raw_text
        return item_text
    
    def user_id2text(self, id: int) -> str:
        # internal id to text
        return self.user_text[id]
        
    def user_token2text(self, token: str) -> str:
        internal_id = self.token2id('user_id', token)
        return self.user_id2text(internal_id)
    
    def item_id2text(self, id: int) -> str:
        # internal id to text
        return self.item_text[id]
        
    def item_token2text(self, token: str) -> str:
        internal_id = self.token2id('item_id', token)
        return self.item_id2text(internal_id)