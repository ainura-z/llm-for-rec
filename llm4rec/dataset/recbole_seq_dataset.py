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
        self.id_token = self.field2id_token["item_id"]
        self.preprocess_text_fn = preprocess_text_fn
        self.item_text = self.load_text()

    def load_text(self) -> tp.List[str]:
        # from internal ids to text
        token_text = {}
        item_text = np.full(len(self.id_token), "", dtype=object)
        item_text[0] = "[PAD]"
        item_file_path = osp.join(self.data_path, f"{self.dataset_name}.item")

        # token id to text mapping
        with open(item_file_path, "r", encoding="utf-8") as file:
            col_names = file.readline().strip().split("\t")
            col_names = [col.split(":")[0] for col in col_names]
            text_col_idx = col_names.index(self.config["text_col"])

            for line in file:
                description = line.strip().split("\t")
                item_id = description[0]
                text = description[text_col_idx]
                token_text[item_id] = text

        # internal id to text mapping
        for i, token in enumerate(self.id_token):
            if token == "[PAD]":
                continue
            raw_text = token_text[token]
            if self.preprocess_text_fn:
                raw_text = self.preprocess_text_fn(raw_text)
            item_text[i] = raw_text
        return item_text

    def id2text(self, id: int) -> str:
        # internal id to text
        return self.item_text[id]
