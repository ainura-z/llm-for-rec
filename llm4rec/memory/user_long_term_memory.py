from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from llm4rec.memory.base_memory import BaseMemory
from langchain_core.embeddings import Embeddings
import faiss


class UserLongTermMemory(BaseMemory):
    """
    LongTermMemory stores long-term preferences of user. It contains snapshots of
    short-term user preferences. The memory is built with Vector Database Retrieval
    """

    def __init__(self, embeddings: Embeddings, emb_size: int, k: int, *args, **kwargs):
        """
        Initialize UserLongTermMemory

        Args:
            embeddings (Embeddings): Model for creating embeddings from text
            emb_size (int): Num of embedding vector dimensions
            k (int): Number of documents to retrieve from memory
        """
        super().__init__(*args, **kwargs)
        self.embeddings = embeddings
        self.k = k
        self.emb_size = emb_size
        self.retrievers = {}

    def update(self, id: str, data: str) -> None:
        """
        Update values in long-term memory retrievers with text data.
        Truncation of memory if the available space exceeded is not implemented.
        """
        if id not in self.memory_store:
            self.memory_store[id] = []
        self.memory_store[id].append(data)

        new_doc = [Document(page_content=data)]

        if id not in self.retrievers:
            index = faiss.IndexFlatL2(self.emb_size)
            vectorstore = FAISS(self.embeddings, index, InMemoryDocstore({}), {})
            retriever = vectorstore.as_retriever(search_kwargs=dict(k=self.k))
            self.retrievers[id] = retriever
        self.retrievers[id].add_documents(new_doc)

    def retrieve(self, id: str, query: str) -> str:
        """
        Retrieve concatenated k relevant to query items from memory.
        """
        try:
            docs = self.retrievers[id].invoke(query)
            return "\n".join([doc.page_content for doc in docs])
        except KeyError:
            return ""
