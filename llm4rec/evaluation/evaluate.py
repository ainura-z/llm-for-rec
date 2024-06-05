from llm4rec.dataset import RecboleSeqDataset
from llm4rec.pipelines import RecBolePipelineRecommender
from llm4rec.evaluation.trainer import PipelineTrainer
from recbole.data.utils import data_preparation
from recbole.config import Config
from recbole.model.abstract_recommender import AbstractRecommender
from torch.utils.data import Dataset 
from llm4rec.tasks.base_recommender import Recommender
from llm4rec.tasks import UserAugmentation, ItemAugmentation
import typing as tp
import os
from dotenv import load_dotenv

def evaluate_pipeline(config: Config, dataset: Dataset, tasks: tp.List[tp.Union[Recommender, ItemAugmentation, UserAugmentation]]):
    _, _, test_data = data_preparation(config, dataset)

    model = RecBolePipelineRecommender(config=config,
                                    dataset=dataset,
                                    tasks=tasks, 
                                    verbose=False)

    trainer = PipelineTrainer(config, model)
    test_result = trainer.evaluate(test_data, show_progress=config['show_progress'])

    return config['model'], config['dataset'], {
        'valid_score_bigger': config['valid_metric_bigger'],
        'test_result': test_result
    }


if __name__ == "__main__":
    from llm4rec.tasks import RetrievalRecommender, RankerRecommender
    from llm4rec.utils.dataset_utils import ml100k_preprocess
    from langchain_groq import ChatGroq

    config_file_list = ['./llm4rec/configs/overall.yaml',
                    './llm4rec/configs/dataset.yaml']
    
    dataset_name = 'ml-100k'
    config = Config(model=RecBolePipelineRecommender, dataset=dataset_name, 
            config_file_list=config_file_list)
    
    dataset = RecboleSeqDataset(config, preprocess_text_fn=ml100k_preprocess)
    
    path_to_env = "../../api_keys.env"
    load_dotenv(path_to_env)
    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0)

    retrieval = RetrievalRecommender(
                embeddings=None,
                item2text=dataset.item_token2text,
                items_info_path=os.path.join(config['data_path'], f"{config['dataset']}.item"),
                search_kwargs={'k':max(config['topk'])})
    
    ranking = RankerRecommender(llm=llm, item2text=dataset.item_token2text)
    tasks = [retrieval, ranking]
    
    result = evaluate_pipeline(config, dataset, tasks)
    print(result)