import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from llm4rec.memory.base_memory import BaseMemory

class UserLongTermMemory(BaseMemory):
    def __init__(self, embeddings, emb_size, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings = embeddings
        self.k = k
        self.emb_size = emb_size
        self.retrievers = {}

    def update(self, id, data: str):
        if id not in self.memory_store:
            self.memory_store[id] = []
        # decide how to truncate
        #elif len(self.memory_store[id]) >= self.memory_limit:
        #    self.memory_store[id].pop(0)

        self.memory_store[id].append(data)

        new_doc = [Document(page_content=data)]

        if id not in self.retrievers:
            index = faiss.IndexFlatL2(self.emb_size)
            # повторное хранение в виде словаря?
            vectorstore = FAISS(self.embeddings, index, InMemoryDocstore({}), {})
            retriever = vectorstore.as_retriever(search_kwargs=dict(k=self.k))
            self.retrievers[id] = retriever

        self.retrievers[id].add_documents(new_doc)


    def retrieve(self, id, query):
        try:
            docs = self.retrievers[id].invoke(query)
            return  "\n".join([doc.page_content for doc in docs])
        except KeyError:
            return ""