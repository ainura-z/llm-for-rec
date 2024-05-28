from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import AIMessage
from langchain.tools import BaseTool
from llm4rec.utils import prepare_input_per_users
from llm4rec.agents import SimpleAgent
import typing as tp
import json
import re


class PlanExecuteAgent(SimpleAgent):
    """
    Implementation of a Plan-and-Execute Agent.
    This method orchestrates a loop pipeline involving the participation 
    of an Agent Planner, Agent Reflection, and Agent Executor.
    """
    def __init__(
            self,
            tools: tp.Sequence[BaseTool],
            llm_executor: tp.Optional[tp.Any],
            prompt_for_agent_executor: tp.Optional[str] = None,
            llm_for_planning: tp.Optional[tp.Any] = None,
            llm_for_reflection: tp.Optional[tp.Any] = None,
            prompt_for_agent_planning: tp.Optional[str] = None,
            prompt_for_agent_replanning: tp.Optional[str] = None,
            prompt_for_agent_reflection: tp.Optional[str] = None,
            planning: bool = True,
            reflection: bool = True,
            max_iter_steps: int = 3,
            verbose: bool = True,
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
                        max_iter_steps=max_iter_steps,
                        verbose=verbose
                        )
        
    def _parse_agent_planning(self, agent_output: AIMessage) -> tp.List[str]:
        plan_cleaned = re.sub(r'\\', '', agent_output.content, flags=re.DOTALL)
        start_index = plan_cleaned.find('{')
        end_index = plan_cleaned.find('}')
        
        plan = json.loads(fr"{plan_cleaned[start_index:end_index+1]}")

        return plan["steps"]
    
    def _create_agent_planner(self):
        """Method for inititalizing the Agent Planner"""

        self.prompt_for_agent_planning = ChatPromptTemplate.from_template(self.prompt_for_agent_planning)
        planner = self.prompt_for_agent_planning | self.llm_for_planning | self._parse_agent_planning
        return planner
    
    
    def _create_agent_reflection(self):
        self.prompt_for_agent_replanning = ChatPromptTemplate.from_template(self.prompt_for_agent_replanning)
        reflection = self.prompt_for_agent_replanning | self.llm_for_reflection
        return reflection
        
    def recommend(
            self,
            user_profile: str,
            prev_interactions: tp.List[str],
            top_k: int
            ):
        
        prompt_for_user = prepare_input_per_users(self.default_prompt_for_user, user_profile, prev_interactions, top_k)
        
        
        reflection_response = AIMessage(content="")
        flag = False
        while not flag:
            plan = self.agent_planning.invoke({"objective": prompt_for_user + f"\nFeedback from the reflection agent: {reflection_response.content}"})

            reflection_response = self.agent_reflection.invoke({"plan": plan})
            if reflection_response.content[:3] == "Yes":
                flag = True
                break

        agent_response = self.agent_executor.invoke({"input": prompt_for_user\
                                                      + "\nYou should follow the generated plan. Plan:" + '\n'.join([f"{index+1}: {item}" for index, item in enumerate(plan)])})       

        return agent_response
