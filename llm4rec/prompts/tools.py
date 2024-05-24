RETRIEVAL_TOOL_DESC = """
                    Tool for finding similar candidate items based on previous interactions of the user.
                    This tool should be used as a first stage of recommendation pipeline.
                    This tool has 3 parameters: user_profile, which is just a string, previous_interactions, which is dictionary of item id and attribute, and top_k. 
                    """


ITEM_DATASET_PAIR_INFO_DESC = """
                            Tool for mapping id of the item to its attributes.
                            This is an auxiliary tool, which is used before Ranker.
                            """


RANKER_DESC = """
            Tool is used to rank candidate items.
            This tool should be used after retrieval tools.
            """


_TOOL_DESC = {
    "RETRIEVAL_TOOL_DESC": RETRIEVAL_TOOL_DESC,
    "RANKER_DESC": RANKER_DESC,
    "ITEM_DATASET_PAIR_INFO_DESC": ITEM_DATASET_PAIR_INFO_DESC
}


OVERALL_TOOL_DESC = """
- "retrieval_recommender": {RETRIEVAL_TOOL_DESC}
- "ranker_recommender": {RANKER_DESC}
- "item_dataset_pair_info": {ITEM_DATASET_PAIR_INFO_DESC}
""".format(
    **_TOOL_DESC
)


RULES = """
        - Retrieval tool should be first invoked.
        - Ranker tool should be invoked after retrieval tool.
        - If Ranker will be invoked, then item_dataset_pair_info tool should be invoked before ranker tool as retrieval tool returns only ids.
        """