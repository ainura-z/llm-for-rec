from llm4rec.dataset import RecboleSeqDataset
from llm4rec.tasks.information_retrieval import RecBoleRetrievalRecommender
from llm4rec.trainer import PipelineTrainer
from recbole.data.utils import data_preparation
from recbole.config import Config
from recbole.model.abstract_recommender import AbstractRecommender
from typing import List
import os
from dotenv import load_dotenv

def evaluate_pipeline(config_file_list: List[str], model_cls: AbstractRecommender, dataset_name: str):
    config = Config(model=model_cls, dataset=dataset_name, 
            config_file_list=config_file_list)
    
    config['openai_api_key'] = os.environ.get("API_KEY")
    assert os.path.exists(config['data_path'])

    dataset = RecboleSeqDataset(config)
    _, _, test_data = data_preparation(config, dataset)

    model = model_cls(config=config,
                    dataset=dataset,
                    openai_api_key=config['openai_api_key'],
                    data_path=os.path.join(config['data_path'],
                    f"{config['dataset']}.item"), csv_args=config['csv_args'],
                    source_column=config['source_column'])

    trainer = PipelineTrainer(config, model)
    test_result = trainer.evaluate(test_data, show_progress=config['show_progress'])

    return config['model'], config['dataset'], {
        'valid_score_bigger': config['valid_metric_bigger'],
        'test_result': test_result
    }


if __name__ == "__main__":
    path_to_openai_env = "../../openai.env"
    load_dotenv(path_to_openai_env)

    config_file_list = ['./llm4rec/configs/pipeline.yaml',
                    './llm4rec/configs/overall.yaml',
                    './llm4rec/configs/dataset.yaml']
    
    result = evaluate_pipeline(config_file_list, RecBoleRetrievalRecommender, 'amazon-books')
    print(result)