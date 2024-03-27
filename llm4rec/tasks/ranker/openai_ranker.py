import typing as tp
import re
from llm4rec.tasks.base_recommender import Recommender
from langchain_openai import OpenAI


class OpenAIRanker(Recommender):
    """
    Ranker that uses base LLM models to rank candidates.
    """
    default_prompts = {
        "sequential": """I’ve interested the following items in the past in order:\n{}Now there are N candidate items 
        that maybe I should be interested: \n{} Please rank these N items by measuring the possibilities that I would
        like to watch next most, according to my interest history. Please think step by step.\n Please show me your 
        ranking results with order numbers.Split your output with line break. You MUST rank the given candidate movies.
        You can not generate movies that are not in the given candidate list.""",
        "recency": """I’ve watched the following movies in the past in order:\n{}\n Now there are 20 candidate movies 
        that I can watch next: {} \n Please rank these 20 movies by measuring the 
        possibilities that I would like to watch next most, according to my watching history. Please think step by 
        step.\n Note that my most recently watched movie is Gladiator. Please show me your ranking results with 
        order numbers. Split your output with line break. You MUST rank the given candidate movies. You can not generate
        movies that are not in the given candidate list.""",
        "in_context": """I’ve watched the following movies in the past in order:\n {}\nThen if I ask you to recommend a 
        new movie to me according to my watching history, you should recommend Shampoo and now that I’ve just watched 
        Shampoo, there are 20 candidate movies that I can watch next:\n{}\n Please rank these 20 movies by measuring 
        the possibilities that I would like to watch next most, according to my watching history. Please think step by 
        step.\n Please show me your ranking results with order numbers. Split your output with line break. You MUST 
        rank the given candidate movies. You can not generate movies that are not in the given candidate list."""
    }

    def __init__(
        self,
        openai_api_key: str,
        custom_prompt: str = None,
        type_prompt: str = 'sequential',
        model: str = "gpt-3.5-turbo-instruct"
    ) -> None:
        """
        Attributes:
        custom_prompt (str): Prompt for ranking.
        type_prompt (str): User has 3 variants of default prompt. He can use custom prompt as well.
        model (str): User can choose any variants of models.
        """
        if custom_prompt is not None:
            self.prompt = custom_prompt
        else:
            self.prompt = self.default_prompts[type_prompt]

        self.llm = OpenAI(model=model, openai_api_key=openai_api_key)
        try:
            self.llm.invoke(self.prompt)
        except Exception:
            raise ValueError(f"OpenAI API key is invalid.")

    @staticmethod
    def __parser(
        document: str
    ) -> tp.List[tp.Any]:
        result = re.sub(r'(\d+\.)|(\d+) ', '', document.replace('\n\n', '')).split('\n')
        result = list(filter(None, result))
        return result

    def recommend(
        self,
        prev_interactions: tp.List[str],
        candidates: tp.List[str]
    ) -> tp.List[tp.Any]:
        if len(candidates) < 2:
            raise ValueError(
                f"User have to have at least two candidate for ranking"
            )
        if len(prev_interactions) == 0:
            prev_interactions = ['Nothing']
        candidates = ["{}".format(candidates[i]) for i in range(len(candidates))]
        self.prompt = self.prompt.format(",\n".join(prev_interactions), ",\n".join(candidates))
        document = self.llm.invoke(self.prompt)
        document_after = self.__parser(document)
        return document_after