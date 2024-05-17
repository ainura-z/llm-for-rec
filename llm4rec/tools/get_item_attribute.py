from functools import partial
from langchain.tools import StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field
from llm4rec.dataset.recbole_seq_dataset import RecboleSeqDataset
import typing as tp


class ItemListInput(BaseModel):
    """Input to the task."""

    item_id_list: tp.List[str] = Field(
        description="The list of item ids from the dataset"
    )
    only_title: bool = Field(
        description='Whether to return the whole information about item or only the title'
    )


def get_info_dict_from_dataset(dataset, item_id_list, only_title=False) -> tp.Dict[str, str]:
    if only_title:
        texts = [text.split(';')[0] for text in dataset.item_token2text(item_id_list)]
    else:
        texts = dataset.item_token2text(item_id_list)
    
    return {item_id_list[i]:texts[i] for i in range(len(item_id_list))}


def create_dataset_item_dict_info_tool(
                dataset: RecboleSeqDataset, 
                name: str = "item_dataset_pair_info",
                description: str = "Should be used before Ranker. " \
                                + "Based on the item_ids gets the attributes like title, release_year. ",
                args_schema: tp.Optional[BaseModel] = None,
                return_direct: bool = False,
                infer_schema: bool = True,
                **kwargs: tp.Any,
    ):
    text_info_dict_from_dataset = partial(get_info_dict_from_dataset, dataset=dataset)

    try:
        return StructuredTool.from_function(
                func=text_info_dict_from_dataset,
                name=name,
                description=description,
                args_schema=ItemListInput,
                return_direct=return_direct,
                infer_schema=infer_schema,
                **kwargs,
            )
    except Exception as e:
        raise ValueError(f"Error while creating tool: {e}")