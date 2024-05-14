import typing as tp
from llm4rec.memory.user_long_term_memory import UserLongTermMemory
from llm4rec.memory.user_short_term_memory import UserShortTermMemory


class UserMemory:
    def __init__(self, user_attributes: tp.Callable, 
                 short_term_limit,
                 llm, embeddings, emb_size, 
                 item_memory, 
                 num_to_retrieve=3, 
                 update_long_term_every=None):
        # global memory
        self.user_attributes = user_attributes

        # personalized memory
        self.short_term_memory = UserShortTermMemory(llm=llm, item_memory=item_memory, memory_limit=short_term_limit)
        self.long_term_memory = UserLongTermMemory(embeddings=embeddings, emb_size=emb_size, k=num_to_retrieve)
        self.short_term_limit = short_term_limit
        self.update_long_term_every = update_long_term_every if update_long_term_every else short_term_limit

        self.llm = llm

    def update(self, id, data):
        self.short_term_memory.update(id, data)
        update_counts = len(self.short_term_memory[id])

        if update_counts % self.update_long_term_every == 0:
            self.long_term_memory.update(id, self.short_term_memory.reflect(id))

    def retrieve(self, id, query, memory_type='all'):
        if memory_type == "long":
            return self.long_term_memory.retrieve(id, query)
        elif memory_type == "short":
            return self.short_term_memory.retrieve(id, query)
        elif memory_type == "all":
            return {
                "long_term": self.long_term_memory.retrieve(id, query),
                "short_term": self.short_term_memory.retrieve(id, query)
            }

    def get_short_term_memory(self, id):
        return self.short_term_memory[id]

    def get_long_term_memory(self, id):
        return self.long_term_memory[id]

    def construct_user_profile(self, id):
        short_term_pref = self.short_term_memory.reflect(id)
        long_term_pref = self.retrieve(id, short_term_pref, memory_type="long")

        profile = f"User {id}"
        if self.user_attributes(id) != "":
            profile += f" attributes: {self.user_attributes(id)}\n"
        else:
            profile += '\n'

        update_counts = len(self.short_term_memory[id])
        if (update_counts - 1) % self.update_long_term_every != 0:
            profile += f"User recent preferences: {short_term_pref}\n"
        profile += f"User long-term preferences: {long_term_pref}."

        return profile