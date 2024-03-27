from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

import os
import typing as tp

from llm4rec.tasks.base_recommender import Recommender


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
        items_info_path: str,
        embeddings: Embeddings = None,
        csv_args: tp.Dict[tp.Any, tp.Any] = None,
        source_column: str = "item_id",
        text_splitter_args: tp.Dict[str, tp.Any] = dict(
            chunk_size=1000, chunk_overlap=0
        ),
        search_type: str = None,
        search_kwargs: tp.Dict[str, tp.Any] = None,
        emb_model_name: str = "all-MiniLM-L6-v2",
        emb_model_kwargs: tp.Dict[str, tp.Any] = dict(device="cuda:0"),
        query=None,
    ):
        """
        Initializes the Retriever.

        Args:
            items_info_path (str): The file path to the data file containing info about items.
            embeddings (Embeddings): The embeddings model instance
            csv_args (Dict[Any, Any], optional): Additional arguments for CSVLoader. Defaults to None.
            source_column (str, optional): The column name for the source data in the CSV file. Defaults to "item_id".
            text_splitter_args (Dict[str, Any], optional): Additional arguments for CharacterTextSplitter. Defaults to {"chunk_size": 1000, "chunk_overlap": 0}.
            search_kwargs: (Dict[str, Any], optional): Additional argumnts for retriever, could be number of k items to retrieve
            emb_model_name: (str, optional): The name of the embedding model if no embeddings are passed
            emb_model_kwargs (Dict[str, Any], optional): Additional arguments for Embeddings instance. Defaults to {"device": "cuda:0"}.
            query (str, optional): Custom query for the retrieval. Defaults to None.

        """
        if not os.path.isfile(items_info_path):
            raise FileNotFoundError("CSV file not found.")
        self.loader = CSVLoader(
            file_path=items_info_path, source_column=source_column, csv_args=csv_args
        )
        documents = self.loader.load()
        self.text_splitter = CharacterTextSplitter(**text_splitter_args)
        docs = self.text_splitter.split_documents(documents)

        if not embeddings:
            embeddings = HuggingFaceEmbeddings(
                model_name=emb_model_name, model_kwargs=emb_model_kwargs
            )
        self.embeddings = embeddings
        vectorstore = FAISS.from_documents(docs, self.embeddings)

        self.retriever = vectorstore.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )
        self.query = query if query else self.base_query

    def _prepare_prev_interactions(self, prev_interactions: tp.List[str]) -> str:
        prev_inreactions = " ".join(
            [
                f"Content-{idx}: {content}"
                for idx, content in enumerate(prev_interactions)
            ]
        )
        return prev_interactions

    def _filter_prev_interactions(
        self, reco_items: tp.List[str], prev_interactions: tp.Dict[str, str]
    ) -> tp.List[str]:
        filtered_items = list(
            dict.fromkeys(
                x for x in dict.fromkeys(reco_items) if x not in prev_interactions
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
        user_profile: str,
        prev_interactions: tp.Dict[str, str],
        top_k: int,
        filter_viewed: bool = True,
    ) -> tp.List[tp.Any]:
        if len(prev_interactions) == 0:
            raise ValueError(
                f"The user must have at least one interaction with the content."
            )
        prev_items = self._prepare_prev_interactions(prev_interactions.values())
        self._set_top_k(top_k + len(prev_items) if filter_viewed else top_k)

        query = self.query.format(user_profile=user_profile, user_history=prev_items)
        documents = self.retriever.get_relevant_documents(query)
        parsed_item_ids = self.parse(documents)
        item_ids = self._remove_duplicate_item_ids(parsed_item_ids)

        if filter_viewed:
            item_ids = self._filter_prev_interactions(item_ids, prev_interactions)
        item_ids = item_ids[:top_k]
        return item_ids