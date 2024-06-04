import typing as tp
from langchain.tools import Tool
from langchain_core.pydantic_v1 import BaseModel, Field
from llm4rec.agents.tools import create_tool
from llm4rec.tasks.base_recommender import Recommender


class RetrievalBaseInput(BaseModel):
    """Input for tool"""
    
    user_profile: str = Field(description='User profile description')
    prev_interactions: tp.List[str] = Field(
        description="Item ids of previous interactions of the user"
    )
    top_k: int = Field(description='Number of items to retrieve')


def create_retrieval_tool(
    retrieval: Recommender,
    name: str = "retrieval_recommender",
    description: str = (
        """
        The tool can find similar items for specific list of previous items. \
        Never use this tool if you don't want to find some items similar with provided items. \
        There is a similarity score threshold in the tool, only {item}s with similarity above the threshold would be kept. \
        Besides, the tool could be used to retrieve the items similar to previous items for ranking tool to refine. \
        The input of the tool should be a list of previous item titles/names, which should be a Python list of strings, the user_profile information in type of string and top_k which is number of items to retrieve \
        Do not fake any item names.
        """
    ),
    args_schema: tp.Optional[BaseModel] = RetrievalBaseInput,
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
        task=retrieval,
        name=name,
        description=description,
        args_schema=args_schema,
        return_direct=return_direct,
        infer_schema=infer_schema,
    )