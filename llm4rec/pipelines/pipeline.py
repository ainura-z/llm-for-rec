from llm4rec.pipelines.base_pipeline import PipelineBase
from llm4rec.tasks import RetrievalRecommender, RankerRecommender, UserAugmentation
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
            if "transform" in dir(task): 
                run_method = task.transform
            else:
                run_method = task.recommend
            
            num_args = run_method.__code__.co_argcount
            arg_names = run_method.__code__.co_varnames#[1:num_args]
            
            # костыль для того чтобы после classic recsys добавлялись тексты к prev_interactions
            if isinstance(task, RankerRecommender) or isinstance(task, RankerRecommender):
                if 'prev_interactions' in inputs and type(inputs['prev_interactions']) != type({}):
                    prev_int_texts = [self.token2text(id) for id in inputs['prev_interactions']]
                    inputs['prev_interactions'] = dict(zip(inputs['prev_interactions'], prev_int_texts))
            
            task_inputs = {arg: inputs[arg] for arg in arg_names if arg in inputs}

            outputs = run_method(**task_inputs)
                
            if self.verbose:
                print(f"Task {i+1} outputs: ", outputs)

            if "transform" in dir(task):
                if isinstance(task, UserAugmentation):
                    inputs['user_profile'] = outputs
                # assume that first argument in inputs is the same var name as in outputs
                else:
                    inputs[list(task_inputs.keys())[0]] = outputs
            else:
                candidate_texts = [self.token2text(output) for output in outputs]
                outputs = dict(zip(outputs, candidate_texts))
                inputs['candidates'] = outputs
                    
        return list(outputs.keys())