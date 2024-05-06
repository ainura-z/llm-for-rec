from llm4rec.pipelines.base_pipeline import PipelineBase
from llm4rec.tasks import RetrievalRecommender
from llm4rec.tasks import RankerRecommender
import typing as tp

class Pipeline(PipelineBase):
    """Pipeline class"""

    def __init__(self, tasks: tp.List[tp.Callable], token2text, *args: tp.Any, verbose: bool = True, **kwargs: tp.Any) -> None:
        self.tasks = tasks
        self.verbose = verbose
        self.token2text = token2text
        
        if len(self.tasks) == 0:
            raise ValueError("The list of tasks should not be empty!")

    def run(self, *args: tp.Any, **inputs: tp.Any):
        for i, task in enumerate(self.tasks):
            num_args = task.recommend.__code__.co_argcount
            arg_names = task.recommend.__code__.co_varnames#[1:num_args]
            task_inputs = {arg: inputs[arg] for arg in arg_names if arg in inputs}
            outputs = task.recommend(**task_inputs)
            
            if self.verbose:
                print(f"Task {i+1} outputs: ", outputs)

            candidate_texts = self.token2text(outputs)
            outputs = dict(zip(outputs, candidate_texts))
            inputs['candidates'] = outputs
            
        return list(outputs.keys())
