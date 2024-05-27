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
            + "Please show me your ranking results with item titles and order numbers. "
            + "Split your output with line break. You MUST rank the given candidate items. "
            + "You can not generate items that are not in the given candidate list. You MUST NOT include items the user has already interacted with.",
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
        item2text: tp.Callable,
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
        self.item2text = item2text

    def _parse(
        self, document: tp.Union[str, AIMessage], candidate_ids: tp.List[str], candidate_texts: tp.List[str]
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
            
        document = document.replace("**", "")
        positions = []
        for idx, candidate in enumerate(candidate_texts):
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
            cand_id for rank, cand_id in sorted(zip(positions, list(candidate_ids)))
        ]
        return ranked_item_ids

    def recommend(
        self, prev_interactions: tp.List[str], candidates: tp.List[str],  user_profile: str= None
    ) -> tp.List[tp.Any]:
        if len(candidates) < 2:
            raise ValueError(f"User has to have at least two candidate for ranking")
        if len(prev_interactions) < 1:
            warnings.warn(
                f"User should have previous interaction data available for ranking"
            )
        
        prev_items_texts = [self.item2text(item_id) for item_id in prev_interactions]#list(prev_interactions.values())
        candidate_items_texts = [self.item2text(item_id) for item_id in candidates]#list(candidates.values())

        # Get last item and next item info for in_context and recency prompts
        last_item = prev_items_texts[-1]
        next_item = prev_items_texts[-1]
        if "next_item" in self.prompt.input_variables:
            if len(prev_items_texts) >= 2:
                last_item = prev_items_texts[-2]
                prev_items_texts = prev_items_texts[:-1]
            elif len(prev_interactions) > 0:
                prev_items_texts = prev_items_texts[:-1]
        
        prompt = self.prompt.format(
            prev_interactions=",\n".join(prev_items_texts),
            num_candidates=len(candidate_items_texts),
            candidates=",\n".join(candidate_items_texts),
            next_item=next_item,
            last_item=last_item,
        )

        if user_profile is not None:
            prompt = "My profile: "+ user_profile +'.\n' + prompt

        result = self.llm.invoke(prompt)

        ranked_items = self._parse(result, candidates, candidate_items_texts)

        if sum(
            [
                (c_item == r_item) * 1
                for c_item, r_item in zip(list(candidates), ranked_items)
            ]
        ) == len(candidates):
            warnings.warn(
                "The ranking stage failed. The order of candidates remained the same"
            )
        return ranked_items