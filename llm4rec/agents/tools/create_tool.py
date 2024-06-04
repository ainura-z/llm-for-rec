from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool,  BaseTool
from langchain_core.tools import ToolException
from functools import partial
import typing as tp
from llm4rec.tasks.base_recommender import Recommender

class BaseInput(BaseModel):
    """Input for tool"""
    
    user_profile: str = Field(description='user profile description')
    prev_interactions: tp.Dict[str, str] = Field(
        description="previous interactions for model"
    )

def _recommend(task, **kwargs) -> tp.List[tp.Any]:
    rec_output = task.recommend(**kwargs)
    return rec_output


async def _arecommend(task, **kwargs) -> tp.List[tp.Any]:
    rec_output = task.recommend(**kwargs)
    return rec_output

def _handle_error(error: ToolException) -> str:
    return (
        "The following errors occurred during tool execution:"
        + error.args[0]
        + "Please try another tool."
    )

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
    args_schema = args_schema or BaseInput

    func = partial(
        _recommend,
        task=task,
    )
    afunc = partial(
        _arecommend,
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
            handle_tool_error=_handle_error,
            **kwargs,
        )
    except Exception as e:
        raise ValueError(f"Error while creating tool: {e}")
