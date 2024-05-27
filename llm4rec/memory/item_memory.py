from llm4rec.memory.base_memory import BaseMemory
from langchain_core.documents import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from tqdm import tqdm
import typing as tp


class ItemMemory(BaseMemory):
    """
    ItemMemory stores additional text information about items
    """

    def __init__(
        self,
        item_ids: tp.List[int],
        dataset_info_map: tp.Callable,
        title_col: str,
        summary_llm: tp.Union[BaseLLM, BaseChatModel] = None,
        augmentation_loader: tp.Callable = None,
        load_filename: str = None,
        *args,
        **kwargs,
    ):
        """
        Initializes item memory

        Args:
            item_ids (List[int]): List of RecBole dataset item ids
            dataset_info_map (Callable): Mapping from item ids to available in dataset text information about items
            title_col (str): The name of column with title of item
            summary_llm (LLM): The model for summarization of item info
            augmentation_loader (Callable): The mapping from item ids to external text item information (Wiki, Google Search)
            load_filename (str): The path to saved memory values
        """
        super().__init__(*args, **kwargs)
        self.summary_llm = summary_llm
        if self.summary_llm:
            self.summary_chain = load_summarize_chain(
                self.summary_llm, chain_type="stuff"
            )
        self.dataset_info_map = lambda item_id: "; ".join([f"{key}: {dataset_info_map(item_id)[key]}" for key in dataset_info_map(item_id)])
        self.title_col = title_col
        self.title_map = lambda item_id: dataset_info_map(item_id)[self.title_col]
        self.augmentation_loader = augmentation_loader
        if load_filename:
            self.load(load_filename)
        self._construct_memory(item_ids)

    def _construct_memory(self, item_ids: tp.List[int]) -> None:
        """
        Create memory values by storing in memory information from dataset about item as well as additional
        external information. Perform summarization if needed.
        """
        for item_id in tqdm(item_ids):
            if item_id not in self.memory_store:
                #item_attr = self.dataset_info_map(item_id)
                text_data = self.title_col + ": " + self.title_map(item_id)
                text_attr = self.dataset_info_map(item_id)#"; ".join([f"{key}: {item_attr[key]}" for key in item_attr])

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
                additional_text_data = (
                    text_attr + "; additional info: " + additional_text_data.strip()
                )
                self.update(item_id, additional_text_data)

    def update(self, id: str, data: str) -> None:
        self.memory_store[id] = data

    def __getitem__(self, id) -> str:
        return self.memory_store.get(id, "")

    def retrieve(self, id, retr_type="dataset_info") -> str:
        if retr_type=='title':
            return self.title_map(id)
        elif retr_type == "dataset_info":
            return self.dataset_info_map(id)
        else:
            return self.memory_store.get(id, "")