import mlflow
from sentence_transformers import SentenceTransformer
import torch
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from tqdm import tqdm
import sys
sys.path.append("/content/rag-financial-assistant")
from src.utils.losses import info_nce_loss
from src.utils.base_embedding import PEFTEmbeddingModel
from src.evaluation.evaluate_models import evaluate_lr
from src.data.indexing_data import create_index
from src.data.fiqa_dataset import FiqaDataset, collate_fn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import os   
import warnings
import torch.nn as nn
import copy
warnings.filterwarnings(
    "ignore",
    message="`encoder_attention_mask` is deprecated and will be removed in version 4.55.0 for `BertSdpaSelfAttention.forward`.",
    category=FutureWarning,
    module="torch.nn.modules.module"
)



class MoELoRA(nn.Module):
    def __init__(self, n_experts, base_layer, hidden_size, lora_rank, lora_alpha, lora_dropout, name, device):
        super().__init__()
        self.n_experts = n_experts
        self.device = device
        self.name = name
        self.experts = nn.ModuleList([])
        for _ in range(n_experts):
            layer_copy = copy.deepcopy(base_layer)
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=['q', 'v'],
                bias='none',
            )
            expert = get_peft_model(layer_copy, lora_config).to(self.device)
            self.experts.append(expert)
        
        self.gate = nn.Linear(hidden_size, self.n_experts).to(self.device)
    
    def extra_repr(self):
        return f"name={self.name}, num_experts={self.n_experts}"

    def forward(self, hidden_states, attention_mask=None, position_ids=None, head_mask=None, output_attentions=False):
        scores = torch.softmax(self.gate(hidden_states), dim=-1)
        outputs = 0
        for i, expert in enumerate(self.experts):
            # Pass hidden_states as keyword argument to match MPNetLayer signature
            expert_output = expert(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask,
                output_attentions=output_attentions
            )
            outputs += scores[:, :, i].unsqueeze(-1) * expert_output[0]
        outputs += hidden_states
        outputs = outputs.unsqueeze(0)
        return outputs
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

def load_peft_model(model: SentenceTransformer, num_expert, lora_rank) -> PeftModel:
    """
    Load Peft Model for fine-tuning.
    """
    hidden_size = model[0].auto_model.config.hidden_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for idx, layer in enumerate(model[0].auto_model.encoder.layer):
            if idx in range(2, 12, 3):
                model[0].auto_model.encoder.layer[idx] = MoELoRA(
                    n_experts=num_expert,
                    base_layer=layer,
                    hidden_size=hidden_size,
                    lora_rank=lora_rank,
                    lora_alpha=2 * lora_rank,
                    lora_dropout=0.1,
                    name=f"MoELoRALayer-{idx}",
                    device=device
                )
    for name, p in model.named_parameters():
        if 'lora' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable_params / total_params if total_params > 0 else 0

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Trainable ratio: {ratio:.2%}")

    return model

def save_onnx_model(model):

    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    # Get the underlying transformer model
    transformer_model = model[0].auto_model.to(device)  # or peft_model._first_module().auto_model

    # Prepare dummy input as a tuple of tensors
    dummy_input = (
        torch.randint(0, transformer_model.config.vocab_size, (2, 32)).to(device),  # input_ids
        torch.ones(2, 32, dtype=torch.long).to(device)  # attention_mask
    )

    torch.onnx.export(
        transformer_model,
        dummy_input,
        "model_base.onnx",
        verbose=False,
        input_names=['input_ids', 'attention_mask'],
        output_names=['output']
    )



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
    transformer = peft_model._first_module().auto_model
    best_eval_loss = 1e9
    counter, patience = 0, 2
    step = 1
    accumulation_steps = 8
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
                loss /= accumulation_steps
                loss.backward()
                if (step+1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()                
                    scheduler.step()
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
    NUM_EPOCHS = 10
    BATCH_SIZE = 4
    NUM_EXPERT = 2
    LORA_RANK = 2
    # Optimized grid - focus on promising configurations first
    model_name = 'sentence-transformers/all-mpnet-base-v2'
    lr = 1e-4
    
    # Set environment variables for optimization
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    with mlflow.start_run(run_name=f"fine-tune-[{model_name.split('/')[-1]}]"):   
        train_data_loader = load_data('train', batch_size=BATCH_SIZE)  
        test_data_loader = load_data('test', batch_size=BATCH_SIZE)
        
        base_model = SentenceTransformer(model_name)


        tokenizer = base_model.tokenizer
        train_data_loader.dataset.tokenizer = tokenizer
        test_data_loader.dataset.tokenizer = tokenizer
        peft_model = load_peft_model(base_model, NUM_EXPERT, LORA_RANK)

#        save_onnx_model(peft_model)
        optimizer = torch.optim.AdamW(
            peft_model.parameters(), 
            lr=lr, 
            weight_decay=0.01
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * len(train_data_loader), eta_min=1e-6)
        #inital_val_loss = eval(peft_model, test_data_loader)
        #mlflow.log_metric('eval_loss', inital_val_loss, step=0)
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
        mlflow.set_tag('LoraType', 'MoELoRA')
        mlflow.log_param('base_model', model_name.split('/')[-1])
        mlflow.log_param('number_of_parameters', sum(p.numel() for p in peft_model.parameters()))
        mlflow.log_param('number_of_experts', 2)
        mlflow.log_param("lora_dropout", 0)
        

if __name__ == '__main__':
    run()
import mlflow
from sentence_transformers import SentenceTransformer
import torch
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from tqdm import tqdm
import sys
sys.path.append("/content/rag-financial-assistant")
from src.utils.losses import info_nce_loss
from src.utils.base_embedding import PEFTEmbeddingModel
from src.evaluation.evaluate_models import evaluate_lr
from src.data.indexing_data import create_index
from src.data.fiqa_dataset import FiqaDataset, collate_fn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import os   
import warnings
import torch.nn as nn
import copy
warnings.filterwarnings(
    "ignore",
    message="`encoder_attention_mask` is deprecated and will be removed in version 4.55.0 for `BertSdpaSelfAttention.forward`.",
    category=FutureWarning,
    module="torch.nn.modules.module"
)



class MoELoRA(nn.Module):
    def __init__(self, n_experts, base_layer, hidden_size, lora_rank, lora_alpha, lora_dropout, name, device):
        super().__init__()
        self.n_experts = n_experts
        self.device = device
        self.name = name
        self.experts = nn.ModuleList([])
        for _ in range(n_experts):
            layer_copy = copy.deepcopy(base_layer)
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=['q', 'v'],
                bias='none',
            )
            expert = get_peft_model(layer_copy, lora_config).to(self.device)
            self.experts.append(expert)
        
        self.gate = nn.Linear(hidden_size, self.n_experts).to(self.device)
    
    def extra_repr(self):
        return f"name={self.name}, num_experts={self.n_experts}"

    def forward(self, hidden_states, attention_mask=None, position_ids=None, head_mask=None, output_attentions=False):
        scores = torch.softmax(self.gate(hidden_states), dim=-1)
        outputs = 0
        for i, expert in enumerate(self.experts):
            # Pass hidden_states as keyword argument to match MPNetLayer signature
            expert_output = expert(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask,
                output_attentions=output_attentions
            )
            outputs += scores[:, :, i].unsqueeze(-1) * expert_output[0]
        outputs += hidden_states
        outputs = outputs.unsqueeze(0)
        return outputs
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

def load_peft_model(model: SentenceTransformer, num_expert, lora_rank) -> PeftModel:
    """
    Load Peft Model for fine-tuning.
    """
    hidden_size = model[0].auto_model.config.hidden_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for idx, layer in enumerate(model[0].auto_model.encoder.layer):
            if idx in range(2, 12, 3):
                model[0].auto_model.encoder.layer[idx] = MoELoRA(
                    n_experts=num_expert,
                    base_layer=layer,
                    hidden_size=hidden_size,
                    lora_rank=lora_rank,
                    lora_alpha=2 * lora_rank,
                    lora_dropout=0.1,
                    name=f"MoELoRALayer-{idx}",
                    device=device
                )
    for name, p in model.named_parameters():
        if 'lora' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable_params / total_params if total_params > 0 else 0

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Trainable ratio: {ratio:.2%}")

    return model

def save_onnx_model(model):

    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    # Get the underlying transformer model
    transformer_model = model[0].auto_model.to(device)  # or peft_model._first_module().auto_model

    # Prepare dummy input as a tuple of tensors
    dummy_input = (
        torch.randint(0, transformer_model.config.vocab_size, (2, 32)).to(device),  # input_ids
        torch.ones(2, 32, dtype=torch.long).to(device)  # attention_mask
    )

    torch.onnx.export(
        transformer_model,
        dummy_input,
        "model_base.onnx",
        verbose=False,
        input_names=['input_ids', 'attention_mask'],
        output_names=['output']
    )



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
    transformer = peft_model._first_module().auto_model
    best_eval_loss = 1e9
    counter, patience = 0, 2
    step = 1
    accumulation_steps = 8
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
                loss /= accumulation_steps
                loss.backward()
                if (step+1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()                
                    scheduler.step()
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
    BATCH_SIZE = 4
    NUM_EXPERT = 2
    LORA_RANK = 2
    # Optimized grid - focus on promising configurations first
    model_name = 'sentence-transformers/all-mpnet-base-v2'
    lr = 1e-4
    
    # Set environment variables for optimization
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    with mlflow.start_run(run_name=f"fine-tune-[{model_name.split('/')[-1]}]"):   
        train_data_loader = load_data('train', batch_size=BATCH_SIZE)  
        test_data_loader = load_data('test', batch_size=BATCH_SIZE)
        
        base_model = SentenceTransformer(model_name)


        tokenizer = base_model.tokenizer
        train_data_loader.dataset.tokenizer = tokenizer
        test_data_loader.dataset.tokenizer = tokenizer
        peft_model = load_peft_model(base_model, NUM_EXPERT, LORA_RANK)

#        save_onnx_model(peft_model)
        optimizer = torch.optim.AdamW(
            peft_model.parameters(), 
            lr=lr, 
            weight_decay=0.01
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * len(train_data_loader), eta_min=1e-6)
        #inital_val_loss = eval(peft_model, test_data_loader)
        #mlflow.log_metric('eval_loss', inital_val_loss, step=0)
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
        mlflow.set_tag('LoraType', 'MoELoRA')
        mlflow.log_param('base_model', model_name.split('/')[-1])
        mlflow.log_param('number_of_parameters', sum(p.numel() for p in peft_model.parameters()))
        mlflow.log_param('number_of_experts', 2)
        mlflow.log_param("lora_dropout", 0)
        

if __name__ == '__main__':
    run()
