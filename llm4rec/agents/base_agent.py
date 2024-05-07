import typing as tp
from abc import ABCMeta
from langchain.tools import BaseTool
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field



class FinalResponse(BaseModel):
    """The final response/answer."""

    response: str


class Replan(BaseModel):
    feedback: str = Field(
        description="Analysis of the previous attempts and recommendations on what needs to be fixed."
    )


class ReflexionOutputs(BaseModel):
    """Decide whether to replan or whether you can return the final response."""

    thought: str = Field(
        description="The chain of thought reasoning for the selected action"
    )
    action: tp.Union[FinalResponse, Replan]


class AgentBase(metaclass=ABCMeta):
    """
    Base Agent class.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    default_prompt_for_agent_planning: str = None # TODO
    default_prompt_for_agent_replanning: str = None # TODO
    default_prompt_for_agent_reflexion: str = None # TODO
    default_prompt_for_agent_executor: str = "You are very powerful assistant for recommedation system, which uses information based on historical user data. "\
                                            + "You have access to the following tools: {tool_description_with_args}. " \
                                            + "Please use the tools to provide the best recommendations for the user."       
  
    default_prompt_for_user: str = (
        "Task: User {user_profile}. This User has previous interactions with these items: {item_ids_with_meta}. Please give {top_k} candidate items recommendations for this user considering his preferences."
    )


    def __init__(
        self,
        tools: tp.Sequence[BaseTool],
        *args: tp.Any,
        llm_executor: tp.Optional[tp.Any] = None,
        llm_for_planning: tp.Optional[tp.Any] = None,
        llm_for_reflection: tp.Optional[tp.Any] = None,
        prompt_for_agent_executor: tp.Optional[str] = None,
        prompt_for_agent_planning: tp.Optional[str] = None,
        prompt_for_agent_replanning: tp.Optional[str] = None,
        prompt_for_agent_reflection: tp.Optional[str] = None,
        planning: bool = False,
        reflection: bool = False,
        max_iter_steps: int = 3,
        verbose: bool = True,
        **kwargs: tp.Any
    ) -> None:
        self.verbose = verbose
        self.tools = tools
        self.max_iter_steps = max_iter_steps


        self.prompt_for_agent_executor = prompt_for_agent_executor or self.default_prompt_for_agent_executor
        self.prompt_for_agent_planning = prompt_for_agent_planning or self.default_prompt_for_agent_planning
        self.prompt_for_agent_replanning = prompt_for_agent_replanning or self.default_prompt_for_agent_replanning
        self.prompt_for_agent_reflexion = prompt_for_agent_reflection or self.default_prompt_for_agent_reflexion
        
        self.llm_executor = llm_executor
        # self.agent_executor = self._create_agent_executor(*args, **kwargs)

        if reflection and llm_for_reflection:
            self.llm_for_reflection = llm_for_reflection
            self.agent_reflection = self._create_agent_reflection(*args, **kwargs)
        else:
            self.llm_for_reflection = None
            self.agent_reflection = None


        if planning and llm_for_planning:
            self.llm_for_planning = llm_for_planning
            self.agent_planning = self._create_agent_planner(*args, **kwargs)
        else:
            self.llm_for_planning = None
            self.agent_planning = None

        
        self.messages: tp.List[BaseMessage] = []

    def _create_agent_executor(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        raise NotImplementedError
    
    def _create_agent_planner(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        raise NotImplementedError

    def _create_agent_reflection(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        raise NotImplementedError

    def _parse_agent_output(self, agent_output: str, *args: tp.Any, **kwargs: tp.Any) -> tp.List[tp.Any]:
        raise NotImplementedError
    
    def recommend(  
        self, 
        *args: tp.Any,
        **kwargs: tp.Any
    )-> None:
        raise NotImplementedError