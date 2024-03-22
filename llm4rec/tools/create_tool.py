import typing as tp
from functools import partial

from langchain.tools import StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field

from llm4rec.tasks import Recommender


class BaselInput(BaseModel):
    """Input to the task."""

    prev_interactions: tp.List[str] = Field(
        description="previous interactions for model"
    )


def _get_reco(task, **kwargs) -> tp.List[str]:
    recos = task.recommend(**kwargs)
    return recos


async def _aget_reco(task, **kwargs) -> tp.List[str]:
    recos = task.recommend(**kwargs)
    return recos


def create_tool(
    task: Recommender,
    name: str,
    description: str,
    args_schema: tp.Optional[BaseModel] = None,
    return_direct: bool = False,
    infer_schema: bool = True,
    **kwargs: tp.Any,
) -> StructuredTool:
    """Create a tool to do any recommendation task.

    Args:
        task (Recommender): The task to use for the retrieval
        name (str): The name for the tool. This will be passed to the language model,
            so should be unique and somewhat descriptive.
        description (str): The description for the tool. This will be passed to the language
            model, so should be descriptive.
        args_schema (BaseModel): The schema of the tool's input arguments.
        return_direct (bool): Whether to return the result directly or as a callback.
        infer_schema (bool): Whether to infer the schema from the function's signature.

    Returns:
        (Tool): Tool class to pass to an agent
    """
    args_schema = args_schema or BaselInput

    func = partial(
        _get_reco,
        task=task,
    )
    afunc = partial(
        _aget_reco,
        task=task,
    )

    try:
        return StructuredTool.from_function(
            func=func,
            afunc=afunc,
            name=name,
            description=description,
            args_schema=args_schema,
            return_direct=return_direct,
            infer_schema=infer_schema,
            **kwargs,
        )
    except Exception as e:
        raise ValueError(f"Error creating tool: {e}")
