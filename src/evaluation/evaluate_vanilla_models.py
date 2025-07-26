import mlflow
from src.data.fiqa_dataset import FiqaDataset
from torch.utils.data import DataLoader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, Document
import random

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
    fida_data = FiqaDataset(split=split)
    print(f"Number of samples in FiQA {split} dataset: {len(fida_data)}")
    return DataLoader(fida_data, batch_size=32)

def load_model(model_name: str):
    print(f'Loading model: {model_name}')
    embed_model = HuggingFaceEmbedding(model_name=model_name)
    Settings.embedding_model = embed_model

def create_index(data_loader, embed_model):
    index = VectorStoreIndex([])
    for batch in data_loader:
        query, context = batch['query_text'], batch['corpus_text']
        context = [random.coice(c) for c in context]
        query_text, context_text = process_inputs(query, context, embed_model.model_name)
        context_docs = [Document(text=c) for c in context_text]
        for doc in context_docs:
            index.insert(doc)
    return index

if __name__ == '__main__':
    list_of_models = [
        "intfloat/e5-small",
        "intfloat/e5-base",
        'BAAI/bge-small-en-v1.5',
        'BAAI/bge-base-en-v1.5',
        'all-MiniLM-L6-v2',
        'all-MiniLM-L12-v2',
    ]
    
    data_loader = load_data()
    model_name = list_of_models[0]  # Change this to test different models
    model = load_model(model_name)

