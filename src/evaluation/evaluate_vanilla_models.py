import mlflow
from src.data.fiqa_dataset import FiqaDataset, collate_fn
from src.utils.losses import info_nce_loss
from torch.utils.data import DataLoader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, Document
import random
from sentence_transformers import SentenceTransformer
import ir_measures
from ir_measures import nDCG, Recall, MAP, MRR
from tqdm import tqdm
import torch

def prepare_mlfow():
    TRACKING_URI = "sqlite:///mlflow/mlflow.db"
    EXPERIMENT_NAME = 'RAG-Financial-Assistant'
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"Tracking URI set to {TRACKING_URI} and experiment name set to {EXPERIMENT_NAME}.")

def process_inputs(query_text, corpus_text, model_name):
    """
    Process the query and corpus text based on the model name.
    """
    if "intfloat" in model_name:
        query_text = [f"Represent the query: {q}" for q in query_text]
        corpus_text = [f"Represent the passage: {c}" for c in corpus_text]
    
    # For BGE models, we use a different format
    elif "bge" in model_name:
        query_text = [f"Represent the query: {q}" for q in query_text]
        corpus_text = [f"Represent the passage: {c}" for c in corpus_text]
    
    return query_text, corpus_text



def load_data(split: str = "train"):
    fiqadata = FiqaDataset(split=split)
    print(f"Number of samples in FiQA {split} dataset: {len(fiqadata)}")
    loader = DataLoader(fiqadata, batch_size=32, collate_fn=collate_fn)
    return loader

def load_model(model_name: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_model = HuggingFaceEmbedding(model_name=model_name, device=device)
    Settings.embed_model = embed_model

def create_index(data_loader, model_name: str):
    data_index = VectorStoreIndex([])
    for batch in tqdm(data_loader, desc=f"Indexing for {model_name}"):
        query = [item['query_text'] for item in batch]
        corpus = [item['corpus_text'] for item in batch] # B * 4
        corpus = [process_inputs(query, c, model_name=model_name)[1] for c in corpus]
        for i, c in enumerate(corpus): # Loop through each item of the batch -> 32
            for j, sub_c in enumerate(c): # Loop through each corpus inside example
                doc = Document(
                    text = sub_c,
                    metadata = {
                        'query_id': batch[i]['query_id'],
                        'corpus_id': batch[i]['corpus_id'][j]
                    }
                )
                data_index.insert(doc)
    return data_index


def compute_loss(model_name, data_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name).to(device) # Can't we use the llamaindex one
    model._target_device = device
    all_losses = []
    for batch in tqdm(data_loader, desc="Calculating initial loss"):
        query = [sample['query_text'] for sample in batch]
        corpus = [[c for c in sample['corpus_text'] if c != '0' ]for sample in batch] # B * 4
        corpus = [random.choice(c) for c in corpus]
        query, corpus = process_inputs(query, corpus, model_name)    
        query_embeddings = model.encode(
            query, 
            convert_to_tensor=True,
            device=device
        )
        corpus_embeddings = model.encode(
            corpus, 
            convert_to_tensor=True, 
            device=device
        )
        all_losses.append(info_nce_loss(query_embeddings, corpus_embeddings))
    total_loss = sum(all_losses) / len(all_losses)
    mlflow.log_metric('number_of_parameters', sum(p.numel() for p in model.parameters()))
    return total_loss.item()


def evaluate(model_name, data_index, data_loader):
    """
    Evaluate the model using the data_index and data loader.
    """
    data_index = data_index.as_retriever(similarity_top_k=5)
    runs = {}
    qrels = {}

    for batch in tqdm(data_loader, desc=f"Evaluating {model_name}"):
        query_texts = [sample['query_text'] for sample in batch]
        corpus_ids = [sample['corpus_id'] for sample in batch]
        corpus_texts = [[c for c in sample['corpus_text'] if c != '0'] for sample in batch]
        selected_corpus = [random.choice(c) for c in corpus_texts]
        query_texts, selected_corpus = process_inputs(query_texts, selected_corpus, model_name)

        for idx, q_text in enumerate(query_texts):
            results = data_index.retrieve(q_text)
            query_id = batch[idx]['query_id']
            runs[str(query_id)] = {
                str(result.metadata['corpus_id']): result.score for result in results
            }
            qrels[str(query_id)] = {
                str(i): 1 for i in corpus_ids[idx] if i != -1
            }

    # Sort results for each query by score (descending)
    runs = {
        qid: dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        for qid, results in runs.items()
    }

    metrics = ir_measures.calc_aggregate([nDCG@10, Recall@5, MAP@10, MRR@10], qrels, runs)
    metrics['info_nce_loss'] = compute_loss(model_name, data_loader)
    return metrics

def run():
    prepare_mlfow()
    all_models = [
        "intfloat/e5-small",
        "intfloat/e5-base",
        'BAAI/bge-small-en-v1.5',
        'BAAI/bge-base-en-v1.5',
        'all-MiniLM-L6-v2',
        'all-MiniLM-L12-v2',
    ]
    all_metrics = []
    print('Loading data...')
    data_loader = load_data(split="test")
    print(f"Data loaded with {len(data_loader)} batches.")
    for model_name in all_models:
        with mlflow.start_run(run_name=f'Evaluating-vanilla-{model_name}'):
            mlflow.set_tag('isTrained', 'False')
            print('Loading model:', model_name)
            mlflow.log_param("base_model", model_name)
            load_model(model_name)
            print(f"Model {model_name} loaded successfully.")
            print("Creating data_index...")
            data_index = create_index(data_loader, model_name)
            print("Index Created")

            print("Evaluating model...")
            metrics = evaluate(model_name, data_index, data_loader)
            all_metrics.append(metrics)
            print(all_metrics)
            for k, v in metrics.items():
                mlflow.log_metric(str(k).replace('@', '_'), v)

    for i in range(len(all_metrics)):
        print(f"Evaluation metrics for {all_models[i]}: {all_metrics[i]}")


if __name__ == '__main__':
    run()
