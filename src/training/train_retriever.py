import mlflow
from sentence_transformers import SentenceTransformer
import torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tqdm import tqdm
from src.utils.losses import info_nce_loss
from src.evaluation.evaluate_models import evaluate_lr
from src.utils.indexing_data import create_index
from src.data.fiqa_dataset import FiqaDataset, collate_fn
from itertools import product
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from lion_pytorch import Lion
import random

def setting_mlflow():
    TRACKING_URI = "sqlite:///mlflow/mlflow.db"
    EXPERIMENT_NAME = 'RAG-Financial-Assistant'
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"Tracking URI set to {TRACKING_URI} and experiment name set to {EXPERIMENT_NAME}.")


def load_data(split: str = "train"):
    fiqadata = FiqaDataset(split=split)
    print(f"Number of samples in FiQA {split} dataset: {len(fiqadata)}")
    loader = DataLoader(fiqadata, batch_size=32, collate_fn=collate_fn)
    return loader


def load_peft_model(model_name: str, lora_config: LoraConfig) -> PeftModel:
    """
    Load Peft Model for fine-tuning.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)
    model = get_peft_model(model, lora_config).to(device)
    return model


def eval(peft_model, test_data_loader):
    peft_model.eval()
    total_loss = 0.0
    for batch in tqdm(test_data_loader, desc="Evaluation"):
        query = [item['query_text'] for item in batch]
        corpus = [[c for c in sample['corpus_text'] if c != '0' ]for sample in batch]
        corpus = [random.choice(c) for c in corpus]
        query_embeddings = peft_model.encode(
            query, 
            convert_to_tensor=True,
        )
        corpus_embeddings = peft_model.encode(
            corpus, 
            convert_to_tensor=True, 
        )
        loss = info_nce_loss(query_embeddings, corpus_embeddings)
        total_loss += loss
    return total_loss / len(test_data_loader)

def train(peft_model, train_data_loader, test_data_loader, optimizer, scheduler, num_epochs: int = 5):
    transformer = peft_model.model._first_module().auto_model 
    tokenizer = peft_model.model.tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for epoch in range(num_epochs):
        total_epoch_loss = 0.0
        progress_bar = tqdm(train_data_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")
        peft_model.train()
        for batch in progress_bar:
            query = [item['query_text'] for item in batch]
            corpus = [[c for c in sample['corpus_text'] if c != '0' ]for sample in batch] # B * 4
            corpus = [random.choice(c) for c in corpus]

            query_inputs = tokenizer(query, padding=True, truncation=True, return_tensors="pt").to(device)
            corpus_inputs = tokenizer(corpus, padding=True, truncation=True, return_tensors="pt").to(device)
            query_outputs = transformer(**query_inputs)
            corpus_outputs = transformer(**corpus_inputs)
            query_embeddings = query_outputs.pooler_output if hasattr(query_outputs, "pooler_output") else query_outputs.last_hidden_state.mean(dim=1)
            corpus_embeddings = corpus_outputs.pooler_output if hasattr(corpus_outputs, "pooler_output") else corpus_outputs.last_hidden_state.mean(dim=1)

            optimizer.zero_grad()
            loss = info_nce_loss(query_embeddings, corpus_embeddings)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_epoch_loss += loss.item()
            mlflow.log_metric('loss', loss.item())
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        epoch_loss = total_epoch_loss / len(train_data_loader)
        eval_loss = eval(peft_model, test_data_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        mlflow.log_metric('eval_loss', eval_loss)
        mlflow.log_metric('epoch_loss', epoch_loss)
    return peft_model


def run():
    setting_mlflow()
    NUM_EPOCHS = 5
    list_of_models = [
        'all-MiniLM-L12-v2',
        'all-mpnet-base-v2',
        'multi-qa-MiniLM-L6-cos-v1',
    ]
    grid = {
        'models': list_of_models,
        "lora_rank": [4, 8, 16],
        "lora_alpha": [8, 16, 32],
        "dropout": [0.0, 0.1, 0.2],
        "lr": [1e-5, 3e-5, 5e-5],
    }
    train_data_loader = load_data('train')
    test_data_loader = load_data('test')
    grid_product = list(product(*grid.values()))
    for parms in tqdm(grid_product, desc="Grid Search"):
        with mlflow.start_run():
            model_name, lora_rank, lora_alpha, lora_doropout, lr = parms
            mlflow.set_tag('isTrained', 'True')
            mlflow.log_param('base_model', model_name)
            mlflow.log_param('lora_rank', lora_rank)
            mlflow.log_param('lora_alpha', lora_alpha)
            mlflow.log_param('lora_doropout', lora_doropout)
            mlflow.log_param('lr', lr)
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_doropout,
                bias='none',
                target_modules =["query", "value"]	
            )
            peft_model = load_peft_model(model_name, lora_config) 
            optimizer = Lion(peft_model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
            scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=0.0)
            peft_model = train(peft_model, train_data_loader, test_data_loader, optimizer, scheduler, NUM_EPOCHS)
            data_index = create_index(test_data_loader, model_name)
            metrics = evaluate_lr(model_name, data_index, test_data_loader)
            for k, v in metrics.items():
                mlflow.log_metric(str(k).replace('@', '_'), v)
            save_dir = f"models/{model_name.replace('/', '-')}_rank{lora_rank}_alpha{lora_alpha}_drop{lora_doropout}"
            peft_model.save_pretrained(save_dir)
            mlflow.log_artifact(save_dir, artifact_path="lora_weights")

if __name__ == '__main__':
    run()
