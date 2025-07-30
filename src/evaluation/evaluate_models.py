import mlflow
import sys
sys.path.append("/content/rag-financial-assistant")
from src.data.fiqa_dataset import FiqaDataset, collate_fn
from src.utils.losses import info_nce_loss
from src.utils.indexing_data import process_inputs, create_index
from torch.utils.data import DataLoader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import random
from sentence_transformers import SentenceTransformer
import ir_measures
from ir_measures import nDCG, Recall, MAP, MRR
from tqdm import tqdm
import torch
import warnings

warnings.filterwarnings(
    "ignore",
    message="`encoder_attention_mask` is deprecated and will be removed in version 4.55.0 for `BertSdpaSelfAttention.forward`.",
    category=FutureWarning,
    module="torch.nn.modules.module",
)


# ...rest of your imports...
def setting_mlflow():
    TRACKING_URI = "sqlite:///mlflow/mlflow.db"
    EXPERIMENT_NAME = "RAG-Financial-Assistant"
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(
        f"Tracking URI set to {TRACKING_URI} and experiment name set to {EXPERIMENT_NAME}."
    )


def load_data(split: str = "train"):
    fiqadata = FiqaDataset(split=split)
    print(f"Number of samples in FiQA {split} dataset: {len(fiqadata)}")
    loader = DataLoader(fiqadata, batch_size=32, collate_fn=collate_fn)
    return loader


def load_model(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = HuggingFaceEmbedding(model_name, device=device)
    embed_model._target_device = device
    Settings.embed_model = embed_model


def compute_loss(model_name, data_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name).to(device)
    model._target_device = device
    all_losses = []
    for batch in tqdm(data_loader, desc="Calculating initial loss"):
        query = [sample["query_text"] for sample in batch]
        corpus = [
            [c for c in sample["corpus_text"] if c != "0"] for sample in batch
        ]  # B * 4
        corpus = [random.choice(c) for c in corpus]
        query, corpus = process_inputs(query, corpus, model_name)
        query_embeddings = model.encode(query, convert_to_tensor=True, device=device)
        corpus_embeddings = model.encode(corpus, convert_to_tensor=True, device=device)
        all_losses.append(info_nce_loss(query_embeddings, corpus_embeddings))
    total_loss = sum(all_losses) / len(all_losses)
    mlflow.log_param("number_of_parameters", sum(p.numel() for p in model.parameters()))
    return total_loss.item()


def evaluate_lr(model_name, data_index, data_loader, compute_info_loss: bool = True):
    """
    Evaluate the model using the data_index and data loader.
    """
    data_index = data_index.as_retriever(similarity_top_k=5)
    runs = {}
    qrels = {}

    for batch in tqdm(data_loader, desc=f"Evaluating {model_name}"):
        query_texts = [sample["query_text"] for sample in batch]
        corpus_ids = [sample["corpus_id"] for sample in batch]
        corpus_texts = [
            [c for c in sample["corpus_text"] if c != "0"] for sample in batch
        ]
        selected_corpus = [random.choice(c) for c in corpus_texts]
        query_texts, selected_corpus = process_inputs(
            query_texts, selected_corpus, model_name
        )

        for idx, q_text in enumerate(query_texts):
            results = data_index.retrieve(q_text)
            query_id = batch[idx]["query_id"]
            runs[str(query_id)] = {
                str(result.metadata["corpus_id"]): result.score for result in results
            }
            qrels[str(query_id)] = {str(i): 1 for i in corpus_ids[idx] if i != -1}

    runs = {
        qid: dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        for qid, results in runs.items()
    }

    metrics = ir_measures.calc_aggregate(
        [nDCG @ 10, Recall @ 5, MAP @ 10, MRR @ 10], qrels, runs
    )
    if compute_info_loss:
        metrics["info_nce_loss"] = compute_loss(model_name, data_loader)
    return metrics


def run():
    setting_mlflow()
    all_models = [
        "intfloat/e5-small",
        "intfloat/e5-base",
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "all-MiniLM-L6-v2",
        "all-MiniLM-L12-v2",
        "multi-qa-MiniLM-L6-cos-v1",
        "all-mpnet-base-v2",
    ]
    all_metrics = []
    print("Loading data...")
    data_loader = load_data(split="test")
    print(f"Data loaded with {len(data_loader)} batches.")
    for model_name in all_models:
        with mlflow.start_run(run_name=f"Evaluating-vanilla-{model_name}"):
            mlflow.set_tag("isTrained", "False")
            print("Loading model:", model_name)
            mlflow.log_param("base_model", model_name)
            load_model(model_name)
            print(f"Model {model_name} loaded successfully.")
            print("Creating data_index...")
            data_index = create_index(data_loader, model_name)
            print("Index Created")

            print("Evaluating model...")
            metrics = evaluate_lr(
                model_name, data_index, data_loader, compute_info_loss=True
            )
            all_metrics.append(metrics)
            print(all_metrics)
            for k, v in metrics.items():
                mlflow.log_metric(str(k).replace("@", "_"), v)

    for i in range(len(all_metrics)):
        print(f"Evaluation metrics for {all_models[i]}: {all_metrics[i]}")


if __name__ == "__main__":
    run()
