import logging
import os
import sys
import typing as tp
from logging import Formatter, StreamHandler

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from llm4rec.tasks.base_recommender import Recommender


class OpenAIRetrievalRecommender(Recommender):
    """
    Recommender that uses a retrieval-based approach to recommend similar content based on user interactions.

    Attributes:
        prompt (str): The prompt for the recommendation.
        loader (CSVLoader): The CSVLoader instance for loading data from the file.
        text_splitter (CharacterTextSplitter): The CharacterTextSplitter instance for splitting documents.
        embeddings_model (OpenAIEmbeddings): The OpenAIEmbeddings instance for generating embeddings.
        vectorstore (Optional[FAISS]): The FAISS document store.
    """

    default_prompt = (
        """User interacted with this content: {}. What content is similar to it?"""
    )

    def __init__(
        self,
        data_path: str,
        openai_api_key: str,
        csv_args: tp.Dict[tp.Any, tp.Any] = None,
        source_column: str = "item_id",
        text_splitter_args: tp.Dict[str, tp.Any] = dict(
            chunk_size=1000, chunk_overlap=0
        ),
        embeddings_model_args: tp.Dict[str, tp.Any] = dict(
            model="text-embedding-ada-002"
        ),
        custom_prompt: str = None,
        log: bool = True,
        search_type: str = None,
        search_kwargs: tp.Dict[str, tp.Any] = None,
    ):
        """
        Initializes the RetrievalRecommender.

        Args:
            data_path (str): The file path to the data file.
            openai_api_key (str): The API key for OpenAI.
            csv_args (Dict[Any, Any], optional): Additional arguments for CSVLoader. Defaults to None.
            source_column (str, optional): The column name for the source data in the CSV file. Defaults to "item_id".
            text_splitter_args (Dict[str, Any], optional): Additional arguments for CharacterTextSplitter. Defaults to {"chunk_size": 1000, "chunk_overlap": 0}.
            embeddings_model_args (Dict[str, Any], optional): Additional arguments for OpenAIEmbeddings. Defaults to {"model": "text-embedding-ada-002"}.
            custom_prompt (str, optional): Custom prompt for the recommendation. Defaults to None.
            log (bool, optional): Whether to log initialization information. Defaults to True.

        """
        if custom_prompt is not None:
            self.prompt = custom_prompt
        else:
            self.prompt = self.default_prompt

        if log:
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.INFO)

            handler = StreamHandler(stream=sys.stdout)
            handler.setFormatter(
                Formatter(fmt="[%(asctime)s: %(levelname)s] %(message)s")
            )

            self.logger.addHandler(handler)
        else:
            self.logger = None

        if not os.path.isfile(data_path):
            raise FileNotFoundError("CSV file not found.")

        self.loader = CSVLoader(
            file_path=data_path, source_column=source_column, csv_args=csv_args
        )
        self.text_splitter = CharacterTextSplitter(**text_splitter_args)
        self.embeddings_model = OpenAIEmbeddings(
            openai_api_key=openai_api_key, **embeddings_model_args
        )

        try:
            vectorstore: tp.Optional[FAISS] = self.create_docstore()
        except Exception as e:
            raise ValueError(f"OpenAI API key is invalid.")

        self.retriever = vectorstore.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )

    def create_docstore(self) -> FAISS:
        if self.logger:
            self.logger.info("Initialize vectore store...")

        docs = self.loader.load()
        docs = self.text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(docs, self.embeddings_model)
        return vectorstore

    def _prepare_prev_interactions(self, prev_inreactions: tp.List[str]) -> str:
        prev_inreactions = " ".join(
            [
                f"Content-{idx}: {content}"
                for idx, content in enumerate(prev_inreactions)
            ]
        )
        return prev_inreactions

    def parse(self, data: tp.List[Document]) -> tp.List[tp.Any]:
        item_ids = [doc.metadata["source"] for doc in data]
        return item_ids

    def recommend(
        self,
        prev_interactions: tp.List[str],
    ) -> tp.List[tp.Any]:
        if len(prev_interactions) == 0:
            raise ValueError(
                f"The user must have at least one interaction with the content."
            )
        prev_interactions = self._prepare_prev_interactions(prev_interactions)
        query = self.prompt.format(prev_interactions)

        documents = self.retriever.invoke(query)
        item_ids = self.parse(documents)
        return item_ids
