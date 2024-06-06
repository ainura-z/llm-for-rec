from llm4rec.memory.user_long_term_memory import UserLongTermMemory
from llm4rec.memory.user_short_term_memory import UserShortTermMemory
from llm4rec.memory.base_memory import BaseMemory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.embeddings import Embeddings
from recbole.data.dataset import Dataset
from tqdm import tqdm

import numpy as np
import typing as tp
import os


class UserMemory:
    """
    Stores and manipulates the historical data about user preferences and interactions.
    Can create user profile based on preferences.
    """

    def __init__(
        self,
        user_attributes: tp.Callable,
        short_term_limit: int,
        llm: tp.Union[BaseLLM, BaseChatModel],
        embeddings: Embeddings,
        emb_size: int,
        item_memory: BaseMemory,
        train_dataset: Dataset,
        min_rating: int = 1,
        max_rating: int = 5,
        num_to_retrieve: int = 3,
        update_long_term_every: int = None,
        load_filename: str = None,
    ):
        """
        Initialized UserMemory

        Args:
            user_attributes (Callable): A mapping fuction to get text user information from dataset.
            short_term_limit (int): Limit value for short-term memory.
            llm (BaseLLM, BaseChatModel): LLM model for reflecting on recent user preferences.
            embeddings (Embeddings): Embeddings model for retrieving of user long-term memory.
            emb_size (int): The dimension number of embedding vectors.
            item_memory (BaseMemory): Storage of item text information.
            train_dataset (Dataset): Dataset instance to initialize memory with dataset values.
            min_rating, max_rating: Values for scaling normalized rating values. Default range is 1-5
            num_to_retrieve (int): Number of retrieved relevant chunks from user long-term memory.
            update_long_term_every (int): How often to update long-term memory.
            load_filename (str): Complete file path to directory with saved memory values
        """
        # global memory
        self.user_attributes = user_attributes

        # personalized memory
        self.short_term_memory = UserShortTermMemory(
            llm=llm, item_memory=item_memory, memory_limit=short_term_limit
        )
        self.long_term_memory = UserLongTermMemory(
            embeddings=embeddings, emb_size=emb_size, k=num_to_retrieve
        )
        self.short_term_limit = short_term_limit
        self.update_long_term_every = (
            update_long_term_every if update_long_term_every else short_term_limit
        )

        self.llm = llm

        if load_filename is not None:
            self.load(load_filename)
        self._construct_memory(train_dataset, min_rating, max_rating)

    def _construct_memory(
        self, train_dataset: Dataset, min_rating: int = 1, max_rating: int = 5
    ) -> None:
        """
        Construct memory from values of train_dataset.
        """
        history_item_matrix = train_dataset.history_item_matrix()
        inter_matrix = train_dataset.inter_matrix("csr", value_field="rating")
        user_id_mapping = lambda user_ids: train_dataset.id2token("user_id", user_ids)
        item_id_mapping = lambda item_ids: train_dataset.id2token("item_id", item_ids)
        history_matrix, _, history_lens = history_item_matrix

        for user_id in tqdm(range(1, len(history_matrix))):
            user_id_token = user_id_mapping(user_id)

            if user_id_token not in self.short_term_memory.memory_store:
                user_history = history_matrix[user_id][: history_lens[user_id]].tolist()
                ratings = (
                    inter_matrix[user_id, :].toarray() * (max_rating - min_rating)
                    + min_rating
                )
                ratings = ratings.astype("int")[0]

                user_id_token = user_id_mapping(user_id)
                item_id_tokens = item_id_mapping(user_history)

                for item, rating in zip(item_id_tokens, ratings):
                    self.update(
                        user_id_token, {"rating": int(rating), "item_id": str(item)}
                    )

    def update(self, id: str, data: tp.Any) -> None:
        """
        Update user memory.
        Long-term memory is updated every update_long_term_every times for user
        based on the number of performed updates.
        """
        self.short_term_memory.update(id, data)
        update_counts = self.short_term_memory.get_update_counts(id)

        if update_counts % self.update_long_term_every == 0:
            self.long_term_memory.update(id, self.short_term_memory.reflect(id))

    def retrieve(self, id: str, query: str, memory_type: str = "all") -> tp.Any:
        """
        Retrieve values from memory based on memory_type
        """
        if memory_type == "long":
            return self.long_term_memory.retrieve(id, query)
        elif memory_type == "short":
            return self.short_term_memory.retrieve(id, query)
        elif memory_type == "all":
            return {
                "long_term": self.long_term_memory.retrieve(id, query),
                "short_term": self.short_term_memory.retrieve(id, query),
            }

    def get_short_term_memory(self, id: str) -> tp.Any:
        return self.short_term_memory[id]

    def get_long_term_memory(self, id: str) -> tp.Any:
        return self.long_term_memory[id]

    def construct_user_profile(self, id: str, use_short_term: bool=False, use_long_term: bool=False) -> str:
        """
        Create user profile information by concatenating available information
        about user from dataset, short-term memory reflection and long-term memory retrieved values.
        """
        short_term_pref = self.short_term_memory.reflect(id)
        long_term_pref = self.retrieve(id, short_term_pref, memory_type="long")

        profile = f"User {id}"
        if self.user_attributes(id) != "":
            profile += f" attributes: {self.user_attributes(id)}\n"
        else:
            profile += "\n"

        if use_short_term:
            update_counts = self.short_term_memory.get_update_counts(id)
            if (update_counts - 1) % self.update_long_term_every != 0:
                profile += f"User recent preferences: {short_term_pref}\n"
        if use_long_term:
            profile += f"User long-term preferences: {long_term_pref}."

        #self.llm.invoke("Construct user profile using the following information: ")
        return profile

    def save(self, folder_path: str) -> None:
        """
        Save short-term memory and long-term memory to folder
        """
        self.short_term_memory.save(folder_path + "/short_term_mem.json")
        self.long_term_memory.save(folder_path + "/long_term_mem.json")

    def load(self, folder_path: str) -> None:
        """
        Load short-term memory and long-term memory from folder
        """
        assert os.path.exists(folder_path + "/short_term_mem.json")
        assert os.path.exists(folder_path + "/long_term_mem.json")

        self.short_term_memory.load(folder_path + "/short_term_mem.json")
        self.long_term_memory.load(folder_path + "/long_term_mem.json")