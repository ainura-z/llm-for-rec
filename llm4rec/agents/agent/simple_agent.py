import typing as tp
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import BaseTool
from llm4rec.agents import AgentBase
from llm4rec.utils import prepare_input_per_users


class SimpleAgent(AgentBase):
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
                    tool_description_with_args="\n".join([f"name: {t.name}\ndescription: {t.description}\nargs: {t.args}" for t in self.tools]),
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
            prev_interactions: tp.Dict[str, str],
            top_k: int
            ):
        
        prompt_for_user = prepare_input_per_users(self.default_prompt_for_user, user_profile, prev_interactions, top_k)
        
        rec = self.agent_executor.invoke({"input": prompt_for_user})

        return rec