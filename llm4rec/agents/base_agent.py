from abc import ABCMeta
from langchain.tools import BaseTool
from llm4rec.prompts import PROMPT_FOR_USER, EXECUTOR_PROMPT, PLANNER_PROMPT, REPLANNER_PROMPT
import typing as tp


class AgentBase(metaclass=ABCMeta):
    """
    Base Agent class.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    default_prompt_for_agent_planning: str = PLANNER_PROMPT

    default_prompt_for_agent_replanning: str = REPLANNER_PROMPT
    default_prompt_for_agent_reflection: str = None # TODO
    default_prompt_for_agent_executor: str = EXECUTOR_PROMPT
    
    default_prompt_for_user: str = PROMPT_FOR_USER


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
        """
        Args:
            tools (Sequence[BaseTool]): A sequence of tools necessary for performing the action.
            *args (Any): Additional positional arguments for the method.
            llm_executor (Optional[Any]): LLM for Agent Executor.
            llm_for_planning (Optional[Any]): LLM for Agent Planning.
            llm_for_reflection (Optional[Any]): LLM used for Agent Reflection.
            prompt_for_agent_executor (Optional[str]): A prompt for the executor agent.
            prompt_for_agent_planning (Optional[str]): A prompt for the planning agent.
            prompt_for_agent_replanning (Optional[str]): A prompt for the replanning agent.
            prompt_for_agent_reflection (Optional[str]): A prompt for the reflection agent.
            planning (bool): A flag indicating whether planning is enabled or not.
            reflection (bool): A flag indicating whether reflection is enabled or not.
            max_iter_steps (int): Maximum number of iteration steps.
            verbose (bool): Verbosity flag indicating whether to print detailed information.
            **kwargs (Any): Additional keyword arguments for the method.
        """
        self.verbose = verbose
        self.tools = tools
        self.max_iter_steps = max_iter_steps


        self.prompt_for_agent_executor = prompt_for_agent_executor or self.default_prompt_for_agent_executor
        self.prompt_for_agent_planning = prompt_for_agent_planning or self.default_prompt_for_agent_planning
        self.prompt_for_agent_replanning = prompt_for_agent_replanning or self.default_prompt_for_agent_replanning
        self.prompt_for_agent_reflexion = prompt_for_agent_reflection or self.default_prompt_for_agent_reflection
        
        self.llm_executor = llm_executor
        self.agent_executor = self._create_agent_executor(*args, **kwargs)

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