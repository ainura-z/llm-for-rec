from langchain import PromptTemplate
from llm4rec.memory.base_memory import BaseMemory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
import typing as tp

class UserShortTermMemory(BaseMemory):
    """
    Memory for storing recent user preferences.
    """

    reflect_prompt_template = PromptTemplate(
        template="You are a helper to understand user needs and preferences in terms of content. "
        + "The user has interacted with following items and gave following ratings: \n{items_with_rating}. "
        + "Summarize user preferences based on the items the user has interacted with in four sentences. "
        + "User preferences: ",
        input_variables=["items_with_rating"],
    )

    def __init__(
        self,
        item_memory: BaseMemory,
        llm: tp.Union[BaseLLM, BaseChatModel],
        memory_limit: int,
        reflect_template: str = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Initializes UserShortTermMemory

        Args:
            item_memory (BaseMemory): The ItemMemory instance with information about items
            memory_limit (int): The number of values to store in memory.
            reflect_template (str): The prompt for reflection of memory values
            llm (BaseLLM, BaseChatModel): The LLM model for reflecting on user recent preferences.
        """
        super().__init__(*args, **kwargs)
        self.memory_limit = memory_limit
        self.llm = llm
        self.reflect_prompt = (
            reflect_template if reflect_template else self.reflect_prompt_template
        )
        self.item_memory = item_memory

        # Storing the number of updates for each user

        self.update_counts = {}

    def update(self, id: str, data: tp.Any) -> None:
        """
        Update memory storage with new data value. If the memory storage size is
        larger than memory limit, the first item from memory is removed and the new
        item is added to memory, following the sliding window approach.
        """
        if id not in self.memory_store:
            self.memory_store[id] = []
            self.update_counts[id] = 0
        elif len(self.memory_store[id]) >= self.memory_limit:
            self.memory_store[id].pop(0)
        self.memory_store[id].append(data)
        self.update_counts[id] += 1

    def get_update_counts(self, id: str) -> int:
        """
        Return number of updates for user with given id
        """
        return self.update_counts.get(id, 0)

    def reflect(self, id: str) -> str:
        """
        Construct text description of user preference values stored in memory
        """
        user_memory = self.retrieve(id)
        interactions = [
            f"{item['item_id']} {self.item_memory.retrieve(item['item_id'])}, user gave rating: {item['rating']}"
            for item in user_memory
        ]
        prompt = self.reflect_prompt.format(items_with_rating="\n".join(interactions))
        history_summary = self.llm.invoke(prompt)
        return history_summary

    def retrieve(self, id: str, *args, **kwargs) -> tp.Any:
        return self.memory_store.get(id, {})
