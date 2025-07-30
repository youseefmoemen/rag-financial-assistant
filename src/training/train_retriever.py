import mlflow
from sentence_transformers import SentenceTransformer
import torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tqdm import tqdm
import sys
sys.path.append("/content/rag-financial-assistant")
from src.utils.losses import info_nce_loss
from src.utils.base_embedding import PEFTEmbeddingModel
from src.evaluation.evaluate_models import evaluate_lr
from src.utils.indexing_data import create_index
from src.data.fiqa_dataset import FiqaDataset, collate_fn
from itertools import product
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import warnings
warnings.filterwarnings(
    "ignore",
    message="`encoder_attention_mask` is deprecated and will be removed in version 4.55.0 for `BertSdpaSelfAttention.forward`.",
    category=FutureWarning,
    module="torch.nn.modules.module"
)
# ...rest of your imports...
def setting_mlflow():
    TRACKING_URI = "https://eb3776e0186c.ngrok-free.app"
    EXPERIMENT_NAME = 'RAG-Financial-Assistant'
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"Tracking URI set to {TRACKING_URI} and experiment name set to {EXPERIMENT_NAME}.")


def load_data(split: str = "train", batch_size: int = 64):  # Increased batch size
    fiqadata = FiqaDataset(split=split)
    print(f"Number of samples in FiQA {split} dataset: {len(fiqadata)}")
    # Add num_workers for parallel data loading
    loader = DataLoader(
        fiqadata, 
        batch_size=batch_size, 
        collate_fn=collate_fn,
        num_workers=4,  # Parallel data loading
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True  # Keep workers alive
    )
    return loader

def preprocess_batch(batch):
    """Preprocess batch once to avoid repeated operations"""
    queries = []
    corpus_texts = []
    
    for item in batch:
        queries.append(item['query_text'])
        # Pre-filter and select corpus text
        valid_corpus = [c for c in item['corpus_text'] if c != '0']
        if valid_corpus:
            corpus_texts.append(random.choice(valid_corpus))
        else:
            corpus_texts.append("")  # fallback
    
    return queries, corpus_texts

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
    with tqdm(test_data_loader, desc="Evaluation", leave=False) as eval_bar:
        for batch in eval_bar:
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
            eval_bar.set_postfix({'batch_loss': f'{loss.item():.4f}'})
    return total_loss / len(test_data_loader)

def train(peft_model, train_data_loader, test_data_loader, optimizer, scheduler, num_epochs: int = 5):
    device = next(peft_model.parameters()).device
    print(f"Training on device: {device}")
    
    transformer = peft_model.model._first_module().auto_model 
    
    for epoch in range(num_epochs):
        total_epoch_loss = 0.0
        peft_model.train()
        
        with tqdm(train_data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False) as progress_bar:
            for idx, batch in enumerate(progress_bar):
                # Stack input_ids for batching
                query_input_ids = torch.stack([item['query_inputs'] for item in batch]).to(device)
                corpus_input_ids = torch.stack([item['corpus_inputs'][0] for item in batch]).to(device)  # [0] picks the first corpus per sample

                # Optionally, create attention masks (1 for non-pad tokens)
                query_attention_mask = (query_input_ids != 0).long()
                corpus_attention_mask = (corpus_input_ids != 0).long()

                # Forward pass
                query_outputs = transformer(input_ids=query_input_ids, attention_mask=query_attention_mask)
                corpus_outputs = transformer(input_ids=corpus_input_ids, attention_mask=corpus_attention_mask)
            
                # Extract embeddings
                query_embeddings = query_outputs.last_hidden_state.mean(dim=1)
                corpus_embeddings = corpus_outputs.last_hidden_state.mean(dim=1)
                
                loss = info_nce_loss(query_embeddings, corpus_embeddings)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_epoch_loss += loss.item()
                progress_bar.set_postfix({
                    'batch_loss': f'{loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
        epoch_loss = total_epoch_loss / len(train_data_loader)
        eval_loss = eval(peft_model, test_data_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Eval Loss: {eval_loss:.4f}")
        mlflow.log_metric('eval_loss', eval_loss)
        mlflow.log_metric('epoch_loss', epoch_loss)
        
    return peft_model, eval_loss

def run():
    setting_mlflow()
    NUM_EPOCHS = 10
    BATCH_SIZE = 8
    # Optimized grid - focus on promising configurations first
    list_of_models = [
        'sentence-transformers/all-MiniLM-L12-v2',  # Add more models later
        'sentence-transformers/all-mpnet-base-v2',  # Start with one model
        'sentence-transformers/all-distilroberta-v1',
        ]
    
    grid = {
        'models': list_of_models,
        "lora_rank": [8, 16],  # Reduced grid size
        "lora_alpha": [16, 32],
        "dropout": [0.1],  
        "lr": [5e-5],
    }
    
    target_modules = {
        'sentence-transformers/all-mpnet-base-v2': ["q", "v"],
        'sentence-transformers/all-distilroberta-v1': ["q", "v"],
        'sentence-transformers/all-MiniLM-L12-v2': ["query", "value"],
    }
    
    train_data_loader = load_data('train', batch_size=BATCH_SIZE)  
    test_data_loader = load_data('test', batch_size=BATCH_SIZE)
    
    # Set environment variables for optimization
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    
    grid_product = list(product(*grid.values()))
    print('Aloo')
    with tqdm(grid_product, desc="Grid Search", total=len(grid_product)) as grid_bar:
        for parms in grid_bar:
            with mlflow.start_run(run_name=f"fine-tune-[{parms[0].split('/')[-1]}]"):
                model_name, lora_rank, lora_alpha, lora_dropout, lr = parms
                
                lora_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias='none',
                    target_modules=target_modules[model_name]
                )
                
                peft_model = load_peft_model(model_name, lora_config) 
                tokenizer = peft_model.model.tokenizer
                train_data_loader.dataset.tokenizer = tokenizer
                test_data_loader.dataset.tokenizer = tokenizer
                optimizer = torch.optim.AdamW(
                    peft_model.parameters(), 
                    lr=lr, 
                    weight_decay=0.01
                )
                scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=0.0)
                
                peft_model, eval_loss = train(
                    peft_model, train_data_loader, test_data_loader, 
                    optimizer, scheduler, NUM_EPOCHS
                )
                
                # Rest of your evaluation code...
                data_index = create_index(test_data_loader, model_name, embed_model=PEFTEmbeddingModel(peft_model, model_name))
                metrics = evaluate_lr(model_name, data_index, test_data_loader, compute_info_loss=False)
                metrics['info_nce_loss'] = eval_loss
                
                for k, v in metrics.items():
                    mlflow.log_metric(str(k).replace('@', '_'), v)
                                    
                # Log parameters
                mlflow.set_tag('isTrained', 'True')
                mlflow.log_param('base_model', model_name.split('/')[-1])
                mlflow.log_param('lora_rank', lora_rank)
                mlflow.log_param('lora_alpha', lora_alpha)
                mlflow.log_param('lora_dropout', lora_dropout)
                mlflow.log_param('lr', lr)
                mlflow.log_param('number_of_parameters', sum(p.numel() for p in peft_model.parameters()))

if __name__ == '__main__':
    run()
