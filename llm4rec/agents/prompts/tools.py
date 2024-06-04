RETRIEVAL_TOOL_DESC = """
                    Tool for finding similar candidate items based on previous interactions of the user.
                    This tool should be used as a first stage of recommendation pipeline.
                    This tool has 3 parameters: user_profile, which is just a string, previous_interactions, which is a list of item ids, and top_k. 
                    """


RANKER_DESC = """
            Tool is used to rank candidate items.
            This tool should be used after retrieval tools.
            """


_TOOL_DESC = {
    "RETRIEVAL_TOOL_DESC": RETRIEVAL_TOOL_DESC,
    "RANKER_DESC": RANKER_DESC
}


OVERALL_TOOL_DESC = """
- "retrieval_recommender": {RETRIEVAL_TOOL_DESC}
- "ranker_recommender": {RANKER_DESC}
""".format(
    **_TOOL_DESC
)


RULES = """
        - Retrieval tool should be first invoked.
        - Ranker tool should be invoked after retrieval tool.
        """