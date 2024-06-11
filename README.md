# LLM4Rec
LLM4Rec is a comprehensive framework for flexible development and reproducible evaluation of LLM-based components in recommendation systems.

![project plan](/docs/imgs/project_structure.jpg)

# Overview
Our framework allows to use both sequential processing of recommendation tasks and usage of intelligent agent systems.

Currently the following stages for sequential pipeline running are supported:

- User and Item Augmentation
- Information retrieval
- Ranking
- Explanation

And agents:
- Simple Agent
- Plan and Execute Agent

# Quick Start
To start working on this project, you need to have `python3` and `pip` installed in your system. Then follow these steps:

1. Clone this repository:

    git clone https://github.com/ainura-z/llm-for-rec.git    
    
3. Install dependencies:

    pip install -r requirements.txt
    
2. (Optional) Run example of sequential pipeline evaluate.py

    python3 llm4rec/evaluation/evaluate.py

# Examples
To demonstrate how to work with our framework we provide following examples:

- [Information Retrieval](./examples/information_retrieval_example.ipynb)
- [Sequential Pipeline with retrieval and ranking](./examples/pipeline_example.ipynb)
- [Sequential Pipeline with augmentation, retrieval and ranking](./examples/augmentation_example.ipynb)
- [Explanation of recommendation](./examples/explanation_of_recommendation_example.ipynb)
- [Using traditional models in pipeline](./examples/classic_recsys_llmrank_example.ipynb)
- [Agents](./examples/agents.ipynb)
- [Creating tools for agents](./examples/tools.ipynb)






    
