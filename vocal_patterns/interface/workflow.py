import datetime
from prefect import task, flow
from vocal_patterns.interface.main import evaluate, preprocess, train
from vocal_patterns.ml_logic.ml_flow import mlflow_transition_model
from vocal_patterns.params import PREFECT_FLOW_NAME


@task
def preprocess_new_data(something):
    return preprocess(something)


@task
def evaluate_production_model(something):
    return evaluate(something)


@task
def re_train(something, something):
    return train(something)


@task
def transition_model(current_stage: str, new_stage: str):
    return mlflow_transition_model(current_stage=current_stage, new_stage=new_stage)


@flow(name=PREFECT_FLOW_NAME)
def train_flow():
    """
    It should:
        - preprocess new data
        - compute `old` by evaluating the current production model based on new test data
        - compute `new` by re-training, then evaluating the new model on the new test data
        - if the new one is better than the old one, replace the current production model with the new one
    """


if __name__ == "__main__":
    train_flow()
