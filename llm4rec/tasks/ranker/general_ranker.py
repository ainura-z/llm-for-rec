from langchain import PromptTemplate
from langchain.schema import AIMessage
from llm4rec.tasks.base_recommender import Recommender
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
import typing as tp
import re
import warnings


class RankerRecommender(Recommender):
    """
    Ranker that uses base LLM models to rank candidates.

    Attributes:
        llm (BaseChatModel, BaseLLM): LLM model for ranking.
        prompt (str): Prompt used for ranking by LLM.
    """

    default_prompts = {
        "sequential": PromptTemplate(
            template="I've been interested in the following items in the past in order:\n{prev_interactions}.\n\n"
            + "Now there are {num_candidates} candidate items that I may be interested in:\n{candidates}.\n"
            + "Please rank these {num_candidates} items by measuring the possibilities that "
            + "I would like to interact with next most, according to my interest history. Please think step by step.\n"
            + "Please show me your ranking results with order numbers. "
            + "Split your output with line break. You MUST rank the given candidate items. "
            + "You can not generate items that are not in the given candidate list.",
            input_variables=["prev_interactions", "num_candidates", "candidates"],
        ),
        "recency": PromptTemplate(
            template="I've been interested in the following items in the past in order:\n{prev_interactions}.\n\n"
            + "Now there are {num_candidates} candidate items that I may be interested in:\n{candidates}.\n"
            + "Please rank these {num_candidates} items by measuring the possibilities that "
            + "I would like to interact with next most, according to my interest history. Please think step by step.\n"
            + "Note that my most recently interacted item is {last_item}. "
            + "Please show me your ranking results with order numbers. "
            + "Split your output with line break. You MUST rank the given candidate items. "
            + "You can not generate items that are not in the given candidate list.",
            input_variables=[
                "prev_interactions",
                "num_candidates",
                "candidates",
                "last_item",
            ],
        ),
        "in_context": PromptTemplate(
            template="I've been interested in the following items in the past in order:\n{prev_interactions}.\n"
            + "Then if I ask you to recommend a new item to me according to my interest history, you should recommend {next_item} and now that "
            + "I've just interacted with {next_item}, there are {num_candidates} candidate items that "
            + "I can interact with next:\n{candidates}.\n "
            + "Please rank these {num_candidates} items by measuring the possibilities that "
            + "I would like to interact with next most, according to my interest history. Please think step by step.\n"
            + "Please show me your ranking results with order numbers. "
            + "Split your output with line break. You MUST rank the given candidate items. "
            + "You can not generate items that are not in the given candidate list.",
            input_variables=[
                "prev_interactions",
                "num_candidates",
                "candidates",
                "next_item",
            ],
        ),
    }

    def __init__(
        self,
        llm: tp.Union[BaseChatModel, BaseLLM],
        custom_prompt: str = None,
        type_prompt: str = "sequential",
    ) -> None:
        """
        Initializes Ranker.

        Args:
            llm (BaseChatModel, BaseLLM): LLM model for ranking.
            custom_prompt (str): Custom prompt for ranking.
            type_prompt (str): Type of default prompt for ranking. Available options: ['sequential', 'in_context', 'recency']
        """
        if custom_prompt:
            self.prompt = custom_prompt
        else:
            if type_prompt not in self.default_prompts.keys():
                raise ValueError(
                    f"The type of default prompt should be one of: {list(self.default_prompts.keys())}. Got: {type_prompt}"
                )
            self.prompt = self.default_prompts[type_prompt]
        self.llm = llm

    def _parse(
        self, document: tp.Union[str, AIMessage], candidates: tp.Dict[str, str]
    ) -> tp.List[tp.Any]:
        """
        Parses the output of LLM by sorting the occurences of each title text from candidate_items.
        The parsing method is unstable and must be carefully implemented for each model.

        Args:
            document (str, AIMessage): The output from LLM.
            candidates (Dict[str, str]): Candidate items with text item info.
        """
        if isinstance(document, AIMessage):
            document = document.content
        positions = []
        for idx, candidate in enumerate(candidates.values()):
            try:
                candidate = candidate.split(";")[0].split(":")[1]
            except:
                candidate = candidate
            candidate_match = re.search(f"\d\.\s*{candidate}", document)
            if candidate_match:
                positions.append(candidate_match.start())
            else:
                positions.append(len(document) + idx)
        ranked_item_ids = [
            cand_id for rank, cand_id in sorted(zip(positions, list(candidates.keys())))
        ]
        return ranked_item_ids

    def recommend(
        self, prev_interactions: tp.Dict[str, str], candidates: tp.Dict[str, str]
    ) -> tp.List[tp.Any]:
        if len(candidates) < 2:
            raise ValueError(f"User has to have at least two candidate for ranking")
        if len(prev_interactions) < 1:
            warnings.warn(
                f"User should have previous interaction data available for ranking"
            )
        prev_items = list(prev_interactions.values())
        candidate_items = list(candidates.values())

        # Get last item and next item info for in_context and recency prompts

        last_item = prev_items[-1]
        next_item = prev_items[-1]
        if "next_item" in self.prompt.input_variables:
            if len(prev_items) >= 2:
                last_item = prev_items[-2]
                prev_items = prev_items[:-1]
            elif len(prev_interactions) > 0:
                prev_items = prev_items[:-1]
        prompt = self.prompt.format(
            prev_interactions=",\n".join(prev_items),
            num_candidates=len(candidate_items),
            candidates=",\n".join(candidate_items),
            next_item=next_item,
            last_item=last_item,
        )

        result = self.llm.invoke(prompt)

        ranked_items = self._parse(result, candidates)

        if sum(
            [
                (c_item == r_item) * 1
                for c_item, r_item in zip(list(candidates.keys()), ranked_items)
            ]
        ) == len(candidate_items):
            warnings.warn(
                "The ranking stage failed. The order of candidates remained the same"
            )
        return ranked_items