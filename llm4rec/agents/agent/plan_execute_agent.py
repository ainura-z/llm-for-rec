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
        

    def _create_agent_planner(self) -> None:
        self.prompt_for_agent_planning = ChatPromptTemplate.from_template(self.prompt_for_agent_planning)
        planner = self.prompt_for_agent_planning | self.llm_for_planning
        return planner

    def _create_agent_reflection(self) -> None:
        self.prompt_for_agent_replanning = ChatPromptTemplate.from_template(self.prompt_for_agent_replanning)
        reflection = self.prompt_for_agent_replanning | self.llm_for_reflection
        return reflection
        
    def recommend(
            self,
            user_profile: str,
            prev_interactions: tp.Dict[str, str],
            top_k: int
            ):
        
        prompt_for_user = prepare_input_per_users(self.default_prompt_for_user, user_profile, prev_interactions, top_k)
        
        tools_description = "\n".join([f"name: {t.name}\ndescription: {t.description}" for t in self.tools])
        
        state = {'input': "Based on user previous interactions, give candidate items recommendations for this user considering his preferences", 
                'plan':[]}
        
        reflection_response = AIMessage(content="")
        flag = False
        while not flag:
            plan = self.agent_planning.invoke({"tools_description": tools_description, "objective": state["input"] + f"\nFeedback from the reflection agent: {reflection_response.content}"})
            plan_cleaned = re.sub(r'\\', '', plan.content, flags=re.DOTALL)
            start_index = plan_cleaned.find('{')
            end_index = plan_cleaned.find('}')
            
            json_object = json.loads(fr"{plan_cleaned[start_index:end_index+1]}")

            state["plan"] = json_object["steps"]

            reflection_response = self.agent_reflection.invoke({"tools_description":tools_description, "plan": state["plan"]})
            if reflection_response.content[:3] == "Yes":
                flag = True
                break

        agent_response = self.agent_executor.invoke({"input": prompt_for_user + "\nYou should follow the generated plan. Plan:" + '\n'.join([f"{index+1}: {item}" for index, item in enumerate(state["plan"])])})       

        return agent_response, state
