from llm4rec.memory.base_memory import BaseMemory

class ItemMemory(BaseMemory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, id, data: str):
        if id not in self.memory_store:
            self.memory_store[id] = ""

        self.memory_store[id] += data

    def retrieve(self, id):
        return self.memory_store.get(id, "")