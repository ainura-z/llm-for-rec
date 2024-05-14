from langchain import PromptTemplate
from llm4rec.memory.base_memory import BaseMemory

class UserShortTermMemory(BaseMemory):
    profile_prompt_template = PromptTemplate(template="You are a helper to understand user needs and preferences in terms of content. "
                                         + "The user has interacted with following items and gave following ratings: \n{items_with_rating}. "
                                         + "Summarize user preferences based on the items the user has interacted with in four sentences. "
                                         + "User preferences: ", input_variables=['items_with_rating'])

    def __init__(self, item_memory, llm, memory_limit, reflect_template=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_limit = memory_limit
        self.llm = llm
        self.reflect_prompt = reflect_template if reflect_template else self.profile_prompt_template
        self.item_memory = item_memory

    def update(self, id, data):
        if id not in self.memory_store:
            self.memory_store[id] = []
        elif len(self.memory_store[id]) >= self.memory_limit:
            self.memory_store[id].pop(0)

        self.memory_store[id].append(data)

    def reflect(self, id):
        user_memory = self.retrieve(id)
        interactions = [f"{item['item_id']} {self.item_memory.retrieve(item['item_id'])}, user gave rating: {item['rating']}" for item in user_memory]
        prompt = self.profile_prompt_template.format(items_with_rating="\n".join(interactions))
        history_summary = self.llm.invoke(prompt)
        return history_summary

    def retrieve(self, id, *args, **kwargs):
        return self.memory_store.get(id, {})
