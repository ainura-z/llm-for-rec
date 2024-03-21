import typing as tp
from abc import ABCMeta

import pandas as pd
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field

from llm4recsys.utils.prompt_creation import prepare_input_for_planning_per_users


class Columns(BaseModel):
    User: str = "user_id"
    Item: str = "item_id"


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
    default_prompt_for_users: str = (
        "User {user_id}: {meta_info}. This User has interactions with these items: {item_ids_with_meta}. Please give personal recommendations for these user."
    )


    def __init__(
        self,
        tools: tp.Sequence[BaseTool],
        *args: tp.Any,
        llm_for_planning: tp.Optional[tp.Any] = None,
        llm_for_reflexion: tp.Optional[tp.Any] = None,
        prompt_for_agent_planning: tp.Optional[str] = None,
        prompt_for_agent_replanning: tp.Optional[str] = None,
        prompt_for_agent_reflexion: tp.Optional[str] = None,
        planning: bool = True,
        reflexion: bool = True,
        max_iter_steps: int = 3,
        verbose: bool = True,
        **kwargs: tp.Any
    ) -> None:
        self.verbose = verbose
        self.tools = tools
        self.max_iter_steps = max_iter_steps

        if reflexion and llm_for_reflexion:
            self.llm_for_reflexion = llm_for_reflexion
        else:
            self.llm_for_reflexion = llm_for_planning

        self.prompt_for_agent_planning = prompt_for_agent_planning or self.default_prompt_for_agent_planning
        self.prompt_for_agent_replanning = prompt_for_agent_replanning or self.default_prompt_for_agent_replanning
        self.prompt_for_agent_reflexion = prompt_for_agent_reflexion or self.default_prompt_for_agent_reflexion

        if planning and reflexion:
            self.agent_planning = self._create_agent_planner(*args, **kwargs)
            self.agent_reflexion = self._create_agent_reflextion(*args, **kwargs)
        elif planning:
            self.agent_planning = self._create_agent_orchestrator(*args, **kwargs)
            self.agent_reflexion = None
        else:
            self.agent_planning = None
            self.agent_reflexion = None

        self.messages: tp.List[BaseMessage] = []

    def _create_agent_planner(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        raise NotImplementedError

    def _create_agent_reflextion(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        raise NotImplementedError

    def _create_agent_orchestrator(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        raise NotImplementedError

    def _parse_agent_output(self, agent_output: str, *args: tp.Any, **kwargs: tp.Any) -> tp.List[tp.Any]:
        raise NotImplementedError

    def _recommend_per_user_as_plan_and_execute(self, prompt: str, k: int, *args: tp.Any, **kwargs: tp.Any) -> tp.List[tp.Any]:
        curr_step = 0
        self.messages += [BaseMessage(contenr=prompt)]
        reco_ids: tp.List[tp.Any] = []
        while curr_step < self.max_iter_steps:
            # 1. and 2. Get plan and execution tools
            tool_messages = self.agent_planning.invoke(self.messages)
            
            # 3. Reflexition: if ok go to the next step else to first step
            self.messages += tool_messages
            decision: ReflexionOutputs = self.agent_reflexion.invoke(self.messages)

            if isinstance(decision.action, Replan):
                self.messages += [
                    AIMessage(content=f"Thought: {decision.thought}"),
                    SystemMessage(
                        content=f"Context from last attempt: {decision.action.feedback}"
                    )
                ]

                curr_step += 1
            else:
                break
        
        reco_ids = tool_messages[-1].content[: k]
        return reco_ids

    def _recommend_per_user_as_orchestrator(self, prompt: str, k: int, *args: tp.Any, **kwargs: tp.Any) -> tp.List[tp.Any]:
        agent_output = self.agent_planning.invoke({"input": prompt})["output"]
        reco_ids = self._parse_agent_output(agent_output, *args, **kwargs)[: k]
        return reco_ids
        
    def _recommend_per_user_tools_sequentially(
        self, 
        prev_interactions: tp.List[tp.Any], 
        k: int, 
        candidates: tp.List[tp.Any] = None
    ) -> tp.List[tp.Any]:
        # TODO: added to signature base recommender `candidates` and `k`
        reco_last_step: tp.List[tp.Any] = candidates
        for tool in self.tools:
            reco_last_step = tool.invoke(
                dict(prev_interactions=prev_interactions, candidates=reco_last_step, k=k)
            )
        return reco_last_step  
    
    def recommend(
        self, 
        users_ids: tp.Optional[tp.List[tp.Any]],
        k: int,
        dataset_with_interactions: pd.DataFrame,
        dataset_user_metadata: pd.DataFrame,
        dataset_item_metadata: pd.DataFrame,
        filter_viewed: bool,
        *args: tp.Any,
        items_to_recommend: tp.Optional[tp.List[tp.Any]] = None,
        prompt_for_users: tp.Union[str, tp.List[str]] = None,
        columns_for_user_meta: tp.Optional[tp.List[str]] = None,
        columns_for_item_meta: tp.Optional[tp.List[str]] = None,
        **kwargs: tp.Any
    ) -> pd.DataFrame:
        dataset_with_interactions = dataset_with_interactions[
            (dataset_with_interactions[Columns.User].isin(users_ids))
        ]  # TODO: transfer to spacy.sparse.csr_matrix
        dataset_user_metadata = dataset_user_metadata[dataset_user_metadata[Columns.User].isin(users_ids)]
        dataset_item_metadata = dataset_item_metadata[dataset_item_metadata[Columns.Item].isin(items_to_recommend)]

        prompt_for_users = prompt_for_users or self.default_prompt_for_users

        if filter_viewed:
            k_reco += dataset_with_interactions.groupby(Columns.User)[Columns.Item].count().max()
        else:
            k_reco = k

        all_user_ids = []
        all_reco_ids = []
        for user_id in users_ids:
            # May be create calling self._recommend_u2i and with method will be NonImplementError???
            prev_interactions = dataset_with_interactions[dataset_with_interactions[Columns.User] == user_id][Columns.Item].values

            if self.agent_planning or self.reflection:
                prompt = prepare_input_for_planning_per_users(
                    prompt=prompt_for_users,
                    user_id=user_id, 
                    prev_interactions=dataset_with_interactions[dataset_with_interactions[Columns.User] == user_id][Columns.Item].values,
                    dataset_user=dataset_user_metadata[dataset_user_metadata[Columns.User] == user_id],
                    dataset_item=dataset_item_metadata[dataset_item_metadata[Columns.Item].isin(prev_interactions)],
                    columns_for_user_meta=columns_for_user_meta,
                    columns_for_item_meta=columns_for_item_meta
                )
            
                if self.agent_planning and self.reflection:
                    reco_ids = self._recommend_per_user_as_plan_and_execute(prompt, k=k_reco, *args, **kwargs)

                else:
                    reco_ids = self._recommend_per_user_as_orchestrator(prompt, k=k_reco, *args, **kwargs)

            else:
                reco_ids = self._recommend_per_user_tools_sequentially(prev_interactions=prev_interactions, candidates=items_to_recommend, k=k_reco)
            
            if filter_viewed:
                reco_ids = list(set(reco_ids).difference(set(prev_interactions)))
                reco_ids = reco_ids[: k_reco]

            all_user_ids += [user_id] * len(reco_ids)
            all_reco_ids += reco_ids

        reco = pd.DataFrame({Columns.User: all_user_ids, Columns.Item: all_reco_ids})

        return reco