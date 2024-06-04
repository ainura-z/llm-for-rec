from llm4rec.prompts.tools import OVERALL_TOOL_DESC, RULES


PROMPT_FOR_USER= \
"""
Task: User {user_profile}. This User has previous interactions with these items: {item_ids_with_meta}. 
Please give {top_k} candidate items recommendations for this user considering his preferences.
"""


EXECUTOR_PROMPT = \
"""
You are very powerful assistant for recommedation system, which uses information based on historical user data.
You have access to the following tools: 
{tools_description}

Tools have the following parameters:
{{tools_description_with_args}}

Please CALL the tools to provide the best recommendations for the user.
The final output can be ONLY a list of recommended items.
""".format(tools_description=OVERALL_TOOL_DESC)


PLANNER_PROMPT = \
"""
For the given objective, come up with a simple step by step plan. \
Here are the tools could be used:
{tools_description} 

Here are some rules to follow while using tools: 
{rules}
First you need to think whether to use tools. If no, give the answer.

Objective: {{objective}}
Plan should be in the following form:
{{{{
    "steps": tp.List[str] = Field(description="different steps to follow, should be in sorted order")
}}}}
Just give the Plan WITHOUT calling the functions.
""".format(tools_description=OVERALL_TOOL_DESC, rules=RULES)


REPLANNER_PROMPT = \
"""
There is a recommendation agent.
The agent could use several tools to deal with the objective. Here are the description of those tools: 
{tools_description}

When giving judgement, you should consider whether the tool using is reasonable? 

There are some rules to follow while using tools:
{rules}

If the plan is reasonable, you should ONLY output "Yes". 
If the plan is not reasonable, you should give "No. The response is not good because ...".

The plan is the following: {{plan}}
""".format(tools_description=OVERALL_TOOL_DESC, rules=RULES)


REFLECTION_PROMPT = None # TODO