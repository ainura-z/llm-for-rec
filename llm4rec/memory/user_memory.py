from llm4rec.memory.user_long_term_memory import UserLongTermMemory
from llm4rec.memory.user_short_term_memory import UserShortTermMemory
import numpy as np
import typing as tp
import os

class UserMemory:
    def __init__(self, user_attributes: tp.Callable, 
                 short_term_limit,
                 llm, embeddings, emb_size, 
                 item_memory, 
                 train_dataset,
                 min_rating=1, max_rating=5,
                 num_to_retrieve=3, 
                 update_long_term_every=None,
                 load_filename=None):
        # global memory
        self.user_attributes = user_attributes

        # personalized memory
        self.short_term_memory = UserShortTermMemory(llm=llm, item_memory=item_memory, memory_limit=short_term_limit)
        self.long_term_memory = UserLongTermMemory(embeddings=embeddings, emb_size=emb_size, k=num_to_retrieve)
        self.short_term_limit = short_term_limit
        self.update_long_term_every = update_long_term_every if update_long_term_every else short_term_limit

        self.llm = llm
        
        if load_filename:
            self.load(load_filename)
            
        self._construct_memory(train_dataset, min_rating, max_rating)
        
    def _construct_memory(self, train_dataset, min_rating=1, max_rating=5):
        history_item_matrix = train_dataset.history_item_matrix()
        inter_matrix =  train_dataset.inter_matrix('csr', value_field='rating')
        user_id_mapping = lambda user_ids:  train_dataset.id2token('user_id', user_ids)
        item_id_mapping =  lambda item_ids:  train_dataset.id2token('item_id', item_ids)
        history_matrix, _, history_lens = history_item_matrix
    
        for user_id in [102]:#range(1, len(history_matrix)):
            if user_id not in self.short_term_memory.memory_store:
                user_history = history_matrix[user_id][:history_lens[user_id]].tolist()
                ratings = inter_matrix[user_id, :].toarray() * (max_rating-min_rating) + min_rating
                ratings = ratings.astype('int')[0]
    
                user_id_token = user_id_mapping(user_id)
                item_id_tokens = item_id_mapping(user_history)
    
                for item, rating in zip(item_id_tokens, ratings):
                    self.update(user_id_token, {'rating':int(rating), 'item_id':str(item)})

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
        
    def save(self, folder_path):
        self.short_term_memory.save(folder_path+'/short_term_mem.json')
        self.long_term_memory.save(folder_path+'/long_term_mem.json')
        
    def load(self, folder_path):
        assert os.path.exists(folder_path+'/short_term_mem.json')
        assert os.path.exists(folder_path+'/long_term_mem.json')
        
        self.short_term_memory.load(folder_path+'/short_term_mem.json')
        self.long_term_memory.load(folder_path+'/long_term_mem.json')
