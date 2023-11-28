import datetime
from prefect import task, flow
from vocal_patterns.interface.main import evaluate, preprocess, train
from vocal_patterns.ml_logic.ml_flow import mlflow_transition_model


@task
def preprocess_new_data(min_date: str, max_date: str):
    return preprocess(min_date=min_date, max_date=max_date)


@task
def evaluate_production_model(min_date: str, max_date: str):
    return evaluate(min_date=min_date, max_date=max_date)


@task
def re_train(min_date: str, max_date: str, split_ratio: str):
    return train(min_date=min_date, max_date=max_date, split_ratio=split_ratio)


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
