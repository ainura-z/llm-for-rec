from llm4rec.memory.base_memory import BaseMemory
from langchain_core.documents import Document
from langchain.chains.summarize import load_summarize_chain
from tqdm import tqdm
import json


class ItemMemory(BaseMemory):
    def __init__(self, item_ids, dataset_info_map, title_col, summary_llm,
                 augmentation_loader=None, load_filename=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.summary_llm = summary_llm
        if self.summary_llm:
            self.summary_chain = load_summarize_chain(self.summary_llm, chain_type="stuff")

        self.dataset_info_map = dataset_info_map
        self.title_col = title_col
        self.augmentation_loader = augmentation_loader
        if load_filename:
            self.load(load_filename)
        self._construct_memory(item_ids)

    def _construct_memory(self, item_ids):
        for item_id in tqdm(item_ids[500:600]):
            if item_id not in self.memory_store:
                item_attr = self.dataset_info_map(item_id)
                text_data = self.title_col + ": " + item_attr[self.title_col]
                text_attr = "; ".join([f'{key}: {item_attr[key]}' for key in item_attr])

                if self.augmentation_loader:
                    docs = self.augmentation_loader(query=text_data).load()
                    if len(docs) == 0:
                        docs = self.augmentation_loader(query=text_attr).load()

                    additional_text_data = "\n".join([doc.page_content for doc in docs])

                    if self.summary_llm:
                        doc = [Document(page_content=additional_text_data)]
                        additional_text_data = self.summary_chain.run(doc)
                else:
                    additional_text_data = text_data

                additional_text_data = text_attr + "; additional info: " + additional_text_data.strip()
                self.update(item_id, additional_text_data)

    def update(self, id, data: str):
        self.memory_store[id] = data

    def __getitem__(self, id):
        return self.memory_store.get(id, "")

    def retrieve(self, id):
        return self.memory_store.get(id, "")

    def save(self, filename):
        assert filename.split('.')[-1] == 'json'

        with open(filename, 'w') as f:
            json.dump(self.memory_store, f)

    def load(self, filename):
        with open(filename) as f:
            self.memory_store = json.load(f)