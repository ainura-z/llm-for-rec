from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
import torch

import os
import typing as tp

from llm4rec.tasks.base_recommender import Recommender
from llm4rec.memory.base_memory import BaseMemory

class RetrievalRecommender(Recommender):
    """
    Recommender that uses a retrieval-based approach to recommend similar content based on user interactions.

    Attributes:
        base_query (str): The query for the retrieval.
        loader (CSVLoader): The CSVLoader instance for loading data from the file.
        text_splitter (CharacterTextSplitter): The CharacterTextSplitter instance for splitting documents.
        embeddings_model (Embeddings): The embeddings model instance for generating embeddings.
        retriever (Optional[FAISS]): The FAISS retriever.
    """

    base_query = """
        The user {user_profile} have interacted with this content: {user_history}. What content is similar to it?
    """

    def __init__(
        self,
        item2text: tp.Callable,
        items_info_path: str = "",
        embeddings: Embeddings = None,
        item_memory: BaseMemory = None,
        load_from_file: bool = True,
        csv_loader_args: tp.Dict[tp.Any, tp.Any] = dict(csv_args={'delimiter':'\t'}, source_column='item_id:token'),
        text_splitter_args: tp.Dict[str, tp.Any] = dict(
            chunk_size=1000, chunk_overlap=0
        ),
        search_type: str = 'similarity',
        search_kwargs: tp.Dict[str, tp.Any] = {"k": 20},
        emb_model_name: str = "all-MiniLM-L6-v2",
        emb_model_kwargs: tp.Dict[str, tp.Any] = {"device":"cuda:0" if torch.cuda.is_available() else "cpu"},
        query=None,
    ):
        """
        Initializes the Retriever.

        Args:
            items_info_path (str): The file path to the data file containing info about items.
            embeddings (Embeddings): The embeddings model instance
            csv_loader_args (Dict[Any, Any], optional): Additional arguments for CSVLoader. Defaults to {"source_column":"item_id", "csv_args":None}.
            text_splitter_args (Dict[str, Any], optional): Additional arguments for CharacterTextSplitter. Defaults to {"chunk_size": 1000, "chunk_overlap": 0}.
            search_kwargs: (Dict[str, Any], optional): Additional argumnts for retriever, could be number of k items to retrieve
            emb_model_name: (str, optional): The name of the embedding model if no embeddings are passed
            emb_model_kwargs (Dict[str, Any], optional): Additional arguments for Embeddings instance. Defaults to {"device": "cuda:0"}.
            query (str, optional): Custom query for the retrieval. Defaults to None.

        """
        self.item2text = item2text
        self.item_memory = item_memory
        self.text_splitter = CharacterTextSplitter(**text_splitter_args)

        if load_from_file:
            if not os.path.isfile(items_info_path):
                raise FileNotFoundError("CSV file not found.")
            docs = self._load_from_file(items_info_path, csv_loader_args)
        else:
            if type(self.item_memory) == type(None):
                raise ValueError("Item memory instance should be provided.")
            
            docs = self._load_from_memory()
            
        if not embeddings:
            embeddings = HuggingFaceEmbeddings(
                model_name=emb_model_name, model_kwargs=emb_model_kwargs
            )
        self.embeddings = embeddings
        vectorstore = FAISS.from_documents(docs, self.embeddings)

        self.retriever = vectorstore.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs.copy()
        )
        self.query = query if query else self.base_query

    def _load_from_memory(self) -> tp.List[Document]:
        documents = self.text_splitter.create_documents(
            texts=list(self.item_memory.get_memory.values()),
            metadatas=list({'source': _id} for _id in self.item_memory.get_memory.keys())
        )
        return documents

    def _load_from_file(self, items_info_path, csv_loader_args) -> tp.List[Document]:
        self.loader = CSVLoader(
            file_path=items_info_path, **csv_loader_args
        )
        documents = self.loader.load()
        docs = self.text_splitter.split_documents(documents)
        return docs

    def _prepare_prev_interactions(self, prev_interactions_texts: tp.List[str]) -> str:
        prev_interactions_str = " ".join(
            [
                f"Content-{idx}: {content}"
                for idx, content in enumerate(prev_interactions_texts)
            ]
        )
        return prev_interactions_str

    def _filter_prev_interactions(
        self, reco_items: tp.List[str], prev_interactions: tp.List[str]
    ) -> tp.List[str]:
        filtered_items = list(
            dict.fromkeys(
                x for x in dict.fromkeys(reco_items) if x not in dict.fromkeys(prev_interactions)
            ).keys()
        )
        return filtered_items

    def _set_top_k(self, top_k: int):
        self.retriever.search_kwargs["k"] = top_k

    def _remove_duplicate_item_ids(self, reco_items: tp.List[str]) -> tp.List[str]:
        return list(dict.fromkeys(reco_items))

    def parse(self, data: tp.List[Document]) -> tp.List[tp.Any]:
        item_ids = [doc.metadata["source"] for doc in data]
        return item_ids

    def recommend(
        self,
        prev_interactions: tp.List[str],
        top_k: int,
        user_profile: str = "",
        filter_viewed: bool = True,
        candidates: tp.Any = None
    ) -> tp.List[tp.Any]:
        if len(prev_interactions) == 0:
            raise ValueError(
                f"The user must have at least one interaction with the content."
            )
        prev_interactions_texts = [self.item2text(item) for item in prev_interactions]
        prev_items = self._prepare_prev_interactions(prev_interactions_texts)
        self._set_top_k(top_k + len(prev_interactions_texts) if filter_viewed else top_k)

        query = self.query.format(user_profile=user_profile, user_history=prev_items)

        documents = self.retriever.get_relevant_documents(query)
        parsed_item_ids = self.parse(documents)
        item_ids = self._remove_duplicate_item_ids(parsed_item_ids)

        if filter_viewed:
            item_ids = self._filter_prev_interactions(item_ids, prev_interactions)
        item_ids = item_ids[:top_k]
        return item_ids