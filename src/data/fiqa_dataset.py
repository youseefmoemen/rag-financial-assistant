from beir.datasets.data_loader import GenericDataLoader
from pathlib import Path
import random
from torch.utils.data import DataLoader
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class FiqaDataset:
    def __init__(self, split: str = "train", tokenizer=None):
        self.DATA_DIR = BASE_DIR / "data" / "fiqa"
        self.split = split
        self.corpus, self.queries, self.qrels = GenericDataLoader(
            self.DATA_DIR, 
        ).load(split=split)
        self.index = list(self.qrels.keys())
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.qrels)

    def __getitem__(self, idx):
        query_id = self.index[idx]
        relevances = self.qrels[query_id]
        corpus_id = [context for context, score in relevances.items() if score > 0]
        query_text = self.queries[str(query_id)]
        corpus_text = [self.corpus[id]['text'] for id in corpus_id]
        corpus_text = corpus_text[:4] + ['0'] * (4 - len(corpus_text))
        corpus_id = corpus_id[:4] + [-1] * (4 - len(corpus_id))
        query_inputs, corpus_inputs = None, None
        if self.tokenizer:
            query_inputs = self.tokenizer(query_text, padding='max_length', truncation=True, max_length=256, return_tensors="pt")['input_ids'][0]
            corpus_inputs = [self.tokenizer(text, padding='max_length', truncation=True, max_length=256, return_tensors="pt")['input_ids'][0] for text in corpus_text]
        return {
        "query_id": query_id,
        "corpus_id": corpus_id,
        "query_text": query_text,
        "corpus_text": corpus_text,
        'query_inputs': query_inputs,
        'corpus_inputs': corpus_inputs,
        }

def preprocess_batch(self, batch):
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

def collate_fn(batch):
    return batch

if __name__ == "__main__":
    fiqa_data = FiqaDataset("test")
    dataloader = DataLoader(fiqa_data, batch_size=32, collate_fn=collate_fn)
    for batch in dataloader:
        print(batch)
        break
    print(f"Number of samples in FiQA train dataset: {len(fiqa_data)}")
