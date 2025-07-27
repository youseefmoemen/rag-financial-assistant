import mlflow

list_of_models = [
    "intfloat/e5-small",
    "intfloat/e5-base",
    'BAAI/bge-small-en-v1.5',
    'BAAI/bge-base-en-v1.5',
    'all-MiniLM-L6-v2',
    'all-MiniLM-L12-v2',
]


def setting_mlflow():
    TRACKING_URI = "sqlite:///mlflow/mlflow.db"
    EXPERIMENT_NAME = 'RAG-Financial-Assistant'
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"Tracking URI set to {TRACKING_URI} and experiment name set to {EXPERIMENT_NAME}.")




def run():
    pass


if __name__ == '__main__':
    run()