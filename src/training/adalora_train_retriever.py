import mlflow
from sentence_transformers import SentenceTransformer
import torch
import json
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from peft import AdaLoraConfig, AdaLoraModel
from tqdm import tqdm
import sys
sys.path.append("/content/rag-financial-assistant")
from src.utils.losses import info_nce_loss
from src.utils.base_embedding import PEFTEmbeddingModel
from src.evaluation.evaluate_models import evaluate_lr
from src.data.indexing_data import create_index
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
def tensor_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_serializable(v) for v in obj]
    return obj

# ...rest of your imports...
def setting_mlflow():
    TRACKING_URI = "http://localhost:5000"  # Adjust as needed
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
        pin_memory=True,  # Faster GPU transfer
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
    with torch.no_grad():
        with tqdm(test_data_loader, desc="Evaluation", leave=False) as eval_bar:
            for batch in eval_bar:
                query = [item['query_text'] for item in batch]
                corpus = [[c for c in sample['corpus_text'] if c != '0' ]for sample in batch]
                corpus = [random.choice(c) for c in corpus]
                query_embeddings = peft_model.encode(
                    query, 
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                corpus_embeddings = peft_model.encode(
                    corpus, 
                    convert_to_tensor=True, 
                    show_progress_bar=False
                )
                loss = info_nce_loss(query_embeddings, corpus_embeddings)
                total_loss += loss
                eval_bar.set_postfix({'batch_loss': f'{loss.item():.4f}'})
            return total_loss / len(test_data_loader)

def train(peft_model, train_data_loader, test_data_loader, optimizer, scheduler, num_epochs: int = 5):
    device = next(peft_model.parameters()).device
    print(f"Training on device: {device}")
    
    transformer = peft_model.model._first_module().auto_model 
    best_eval_loss = 1e9
    counter, patience = 0, 2
    step = 1
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
                torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                peft_model.update_and_allocate(step)
                if loss.item() > 1 and step > 500:
                    print('Skipping bad batch')
                    serializable_batch = tensor_to_serializable(batch)
                    with open(f"bad_batch_{step}.json", "w") as f:
                        json.dump(serializable_batch, f, indent=2)
                    continue
                total_epoch_loss += loss.item()
                batch_metrics = {
                    'train_loss': total_epoch_loss / (idx+1),
                    'learning_rate': scheduler.get_last_lr()[0],
                    'batch_loss': loss.item(),
                }
                mlflow.log_metrics(batch_metrics, step=step)
                if step % 100 == 0:
                    eval_loss = eval(peft_model, test_data_loader)  
                    mlflow.log_metric('eval_loss', eval_loss, step=step)
                progress_bar.set_postfix({
                    'batch_loss': f'{loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                step += 1

        epoch_loss = total_epoch_loss / len(train_data_loader)
        eval_loss = eval(peft_model, test_data_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Eval Loss: {eval_loss:.4f}")
        mlflow.log_metric('eval_loss', eval_loss)
        mlflow.log_metric('epoch_loss', epoch_loss)
        if eval_loss < best_eval_loss - 1e-4:
            best_eval_loss = eval_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print('Early stopping at epoch {epoch}"')
                break        
    return peft_model, eval_loss

def run():
    setting_mlflow()
    NUM_EPOCHS = 4
    BATCH_SIZE = 8
    # Optimized grid - focus on promising configurations first
    list_of_models = [
        'sentence-transformers/all-mpnet-base-v2'
        ]
    
    grid = {
        'models': list_of_models,       
        "init_r": [32],
        "target_r": [2],  
        "lr": [1e-4],
        'lora_alpha': [4],
        'lora_dropout': [0.2],
    }
    target_modules = {
        'sentence-transformers/all-mpnet-base-v2': ["q", "v"], # Make it q, v
        'sentence-transformers/msmarco-distilbert-base-v4': ["k_lin", "v_lin"],
        'sentence-transformers/all-distilroberta-v1': ["query", "value"],
        'sentence-transformers/all-MiniLM-L12-v2': ["query", "value"],
    }
    
    
    # Set environment variables for optimization
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    
    grid_product = list(product(*grid.values()))
    with tqdm(grid_product, desc="Grid Search", total=len(grid_product)) as grid_bar:
        for parms in grid_bar:
            with mlflow.start_run(run_name=f"fine-tune-[{parms[0].split('/')[-1]}]"):   
                train_data_loader = load_data('train', batch_size=BATCH_SIZE)  
                test_data_loader = load_data('test', batch_size=BATCH_SIZE)
                model_name, init_r, target_r, lr, lora_alpha, lora_dropout = parms
                
                adalora_config = AdaLoraConfig(
                    peft_type="ADALORA",
                    task_type=TaskType.FEATURE_EXTRACTION,
                    target_modules=target_modules[model_name],
                    init_r=init_r,
                    target_r=target_r,
                    tinit=100,
                    tfinal=1000,
                    deltaT=20,
                    total_step=NUM_EPOCHS * len(train_data_loader),
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    beta1=0.85,
                    beta2=0.85,
                    orth_reg_weight=0.5,
                    use_rslora=True,
                )
                base = SentenceTransformer(model_name)
                peft_model = AdaLoraModel(base, adalora_config, adapter_name="default")
                tokenizer = peft_model.model.tokenizer
                train_data_loader.dataset.tokenizer = tokenizer
                test_data_loader.dataset.tokenizer = tokenizer
                optimizer = torch.optim.AdamW(
                    peft_model.parameters(), 
                    lr=lr, 
                    weight_decay=0.01
                )
                scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * len(train_data_loader), eta_min=1e-6)
                inital_val_loss = eval(peft_model, test_data_loader)
                mlflow.log_metric('eval_loss', inital_val_loss, step=0)
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
                mlflow.set_tag('LoraType', 'AdaLora')
                mlflow.log_param('base_model', model_name.split('/')[-1])
                mlflow.log_param('init_r', init_r)
                mlflow.log_param('target_r', target_r)
                mlflow.log_param('lora_alpha', lora_alpha)
                mlflow.log_param('number_of_parameters', sum(p.numel() for p in peft_model.parameters()))
                mlflow.log_param('LoraType', 'rslora')
                mlflow.log_param("lora_dropout", lora_dropout)
                

if __name__ == '__main__':
    run()
