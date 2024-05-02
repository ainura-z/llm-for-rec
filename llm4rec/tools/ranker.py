import typing as tp
from langchain.tools import Tool
from langchain_core.pydantic_v1 import BaseModel, Field
from llm4rec.tools import create_tool
from llm4rec.tasks.base_recommender import Recommender


class RankerInput(BaseModel):
    """Input for ranker tool"""
    prev_interactions: tp.Dict[str, str] = Field(
        description="previous interactions for model"
    )
    candidates: tp.Dict[str, str] = Field(
        description="candidate items for recommendation from previous step"
    )


def create_ranker_tool(
    ranker: Recommender,
    name: str = "retrieval_recommender",
    description: str = (
        """Search content that is most similar to content 
        from previous interactions with the recommender system. 
        If you have any questions about searching related content, 
        you should use this tool"""
    ),
    args_schema: tp.Optional[BaseModel] = RankerInput,
    return_direct: bool = False,
    infer_schema: bool = True,
) -> Tool:
    """Create a tool to do retrieval recommendendation.

    Args:
        retrieval (Recommender): The retrieval to use for the retrieval
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
    return create_tool(
        task=ranker,
        name=name,
        description=description,
        args_schema=args_schema,
        return_direct=return_direct,
        infer_schema=infer_schema,
    )