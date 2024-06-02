import torch
import os
import typing as tp
from llm4rec.tasks.base_recommender import Recommender
from recbole.data.interaction import Interaction
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain import PromptTemplate, LLMChain


class ExplainableRecommender(Recommender):
    """
    Class for building the explanation about recommended candidate items to user.

    Attributes:
        config (str): The config file from RecBole.
        llm: The LLM model for building an explanation. 
    """
    def __init__(self, 
                 config: tp.Dict[str, tp.Any], 
                 llm: tp.Union[BaseChatModel, BaseLLM]
                 ):
        """
        Initializes the ExplainableRecommender.

        Args:
            config (str): The config file from RecBole.
            llm: The LLM model for building an explanation. 
        """

        self.config = config
        self.llm = llm

    def _separate_preferences(self, rating: tp.List[float], history_length: int) -> tp.Tuple[tp.List[str], tp.List[str]]:
        """
        Separates liked and disliked movies based on ratings.
        """

        movies_preferences = torch.where(rating >= self.config['threshold_rating'], torch.tensor(1), torch.tensor(0))
        indices_liked = torch.nonzero(movies_preferences[:history_length] == 1).squeeze()
        indices_disliked = torch.nonzero(movies_preferences[:history_length] == 0).squeeze()

        return indices_liked, indices_disliked 

    def _construct_prompt(self, movies_liked: tp.List[str], movies_disliked: tp.List[str]) -> PromptTemplate:
        """
        Constructs the prompt based on liked and disliked movies.
        """
        prompt = 'There is a person {user_profile}.\n'
        if len(movies_liked) > 0 and len(movies_disliked) > 0:
            prompt += "The person has a list of liked items: {movies_liked}. "
            prompt += "The person has a list of disliked items: {movies_disliked}. "
        elif len(movies_liked) > 0:
            prompt += "The person has a list of liked items: {movies_liked} "
            prompt += "and there is no information about disliked items but based on the provided liked items \
                       try to understand what can be potential disliked items."
        elif len(movies_disliked) > 0:
            prompt += "The person has a list of disliked items: {movies_disliked} "
            prompt += "and there is no information about liked items but based on the provided disliked items try to understand what can be potential liked items."


        prompt += """Generate ONE SENTENCE explanation why the following candidate item {candidate_item} can be liked by the user.
                     Answer should be ONLY ONE sentence in the following way: You might like the movie because...
                     .
                  """

        prompt_template = PromptTemplate(template=prompt, input_variables=["user_profile", "movies_liked", "movies_disliked", "candidate_item"])
        return prompt_template


    def recommend(
        self,
        user_interaction: Interaction, 
        history_names: tp.List[str], 
        candidate_item: str, 
        user_profile: str = ""
    ):
        """
        Generates explanations for recommendations based on user preferences.
        """
        indices_liked, indices_disliked = self._separate_preferences(user_interaction['rating_list'], self.config['history_length'])

        # getting the liked/disliked movies' names
        movies_liked = history_names[indices_liked]
        movies_disliked = history_names[indices_disliked]


        prompt_template = self._construct_prompt(movies_liked.tolist(), movies_disliked.tolist())

        chain = LLMChain(prompt=prompt_template, llm=self.llm, verbose=True)
        explanation = chain.run({"user_profile": user_profile, 'movies_liked':', '.join(movies_liked), 'movies_disliked': ', '.join(movies_disliked),'candidate_item': candidate_item})

        return explanation