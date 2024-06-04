import typing as tp
from langchain.tools import Tool
from langchain_core.pydantic_v1 import BaseModel, Field
from llm4rec.agent.tools import create_tool
from llm4rec.tasks.base_recommender import Recommender


class RankerInput(BaseModel):
    """Input for ranker tool"""
    prev_interactions: tp.List[str] = Field(
        description="Item ids of previous interactions"
    )
    candidates: tp.List[str] = Field(
        description="Item ids of candidate items for recommendation from previous step"
    )


def create_ranking_tool(
    ranker: Recommender,
    name: str = "ranker_recommender",
    description: str = (
        """
        The tool is useful to refine items order (for better experiences) or remove unwanted items from the top. \
        The input of the tool should be previous interaction data (item ids and their text data and candidate recommendation items (item ids and ther text data). \
        The candidates depend on previous tool using. Only when there is a list of candidate items to recommend \
        this tool could be used.
        """
    ),
    args_schema: tp.Optional[BaseModel] = RankerInput,
    return_direct: bool = False,
    infer_schema: bool = True,
) -> Tool:
    """Create a tool to do ranker recommendendation.

    Args:
        ranker (Recommender): The ranker to use 
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