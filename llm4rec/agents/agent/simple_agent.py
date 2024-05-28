import typing as tp
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import BaseTool
from llm4rec.agents import AgentBase
from llm4rec.utils import prepare_input_per_users


class SimpleAgent(AgentBase):
    """
    Implementation of a Simple Agent based on the AgentBase class.
    This Agent runs a direct recommendation pipeline. 
    """
    def __init__(
            self,
            tools: tp.Sequence[BaseTool],
            llm_executor: tp.Optional[tp.Any] = None,
            llm_for_planning: tp.Optional[tp.Any] = None,
            llm_for_reflection: tp.Optional[tp.Any] = None,
            prompt_for_agent_executor: tp.Optional[str] = None,
            prompt_for_agent_planning: tp.Optional[str] = None,
            prompt_for_agent_replanning: tp.Optional[str] = None,
            prompt_for_agent_reflection: tp.Optional[str] = None,
            planning: bool = False,
            reflection: bool = False,
            max_iter_steps = 3,
            verbose = True
            ):
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
        super().__init__(
                        tools=tools,
                        llm_executor=llm_executor,
                        prompt_for_agent_executor=prompt_for_agent_executor,
                        llm_for_planning=llm_for_planning,
                        llm_for_reflection=llm_for_reflection,
                        prompt_for_agent_planning=prompt_for_agent_planning,
                        prompt_for_agent_replanning=prompt_for_agent_replanning,
                        prompt_for_agent_reflection=prompt_for_agent_reflection,
                        planning=planning,
                        reflection=reflection,
                        max_iter_steps = max_iter_steps,
                        verbose = verbose
                        )
        

    def _create_agent_executor(self):
        """Method for inititalizing the Agent Executor"""

        self.prompt_for_agent_executor = ChatPromptTemplate.from_messages(
                    [
                        (
                        "system",
                            self.prompt_for_agent_executor
                        ),
                        ("user", "{input}"),
                        ("placeholder", "{agent_scratchpad}"),
                    ]
            )

        self.prompt_for_agent_executor = self.prompt_for_agent_executor.partial(
                    tools_description_with_args="\n".join([f"name: {t.name}\ndescription: {t.description}\nargs: {t.args}" for t in self.tools]),
                    )
        
        agent = create_tool_calling_agent(self.llm_executor, self.tools, self.prompt_for_agent_executor)
        agent_executor = AgentExecutor(
                                agent=agent,
                                tools=self.tools, 
                                verbose=True,
                                return_intermediate_steps=True,
                                handle_parsing_errors=True)
        return agent_executor

        
    def recommend(
            self,
            user_profile: str,
            prev_interactions: tp.List[str],
            top_k: int
            ):
        
        prompt_for_user = prepare_input_per_users(self.default_prompt_for_user, user_profile, prev_interactions, top_k)
        
        rec = self.agent_executor.invoke({"input": prompt_for_user})

        return rec
