import typing as tp
from abc import ABCMeta
from langchain.tools import BaseTool


class AgentBase(metaclass=ABCMeta):
    """
    Base Agent class.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    default_prompt_for_agent_planning: str = """
                                                For the given objective, come up with a simple step by step plan.
                                                Here are the tools could be used: 
                                                retrieval_recommender: tool for finding similar candidate items based on previous interactions of the user
                                                item_dataset_pair_info: tool for mapping id of the item to its attributes, used before Ranker
                                                ranker_recommender: tool for ranking candidates items

                                                First you need to think whether to use tools. If no, give the answer.

                                                Objective: {objective}
                                                Plan should be in the following form:
                                                {{
                                                    "steps": tp.List[str] = Field(description="different steps to follow, should be in sorted order")
                                                }}
                                                Just give the Plan WITHOUT calling the functions.
                                            """

    default_prompt_for_agent_replanning: str = """
                                                There is a recommendation agent.
                                                The agent could use several tools to deal with the objective. Here are the description of those tools: {tools_description}
                                                When giving judgement, you should consider whether the tool using is reasonable? 
                                                For example, ranker tool cannot be used before retrieval tool. And as retrieval tool returns only ids, item_dataset_pair_info should be used before ranker tool.
                                                But the agent could use only retrieval tool, which is also fine.

                                                If the plan is reasonable, you should ONLY output "Yes". 
                                                If the plan is not reasonable, you should give "No. The response is not good because ...".

                                                The plan is the following: {plan}
                                            """
    default_prompt_for_agent_reflection: str = None # TODO
    default_prompt_for_agent_executor: str = """
                                                You are very powerful assistant for recommedation system, which uses information based on historical user data.
                                                You have access to the following tools: {tool_description_with_args}.
                                                Please use the tools to provide the best recommendations for the user.       
                                            """
    
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