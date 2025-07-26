from datasets import load_dataset
from beir.datasets.data_loader import GenericDataLoader
from pathlib import Path
import random
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class FiqaDataset:
    def __init__(self, split: str = "train"):
        self.DATA_DIR = BASE_DIR / "data" / "fiqa"
        self.split = split
        self.corpus, self.queries, self.qrels = GenericDataLoader(
            self.DATA_DIR, 
        ).load(split=split)
        self.index = list(self.qrels.keys())

    def __len__(self):
        return len(self.qrels)

    def __getitem__(self, idx):
        query_id = self.index[idx]
        relevances = self.qrels[query_id]
        context_id = [context for context, score in relevances.items() if score > 0]
        query_text = self.queries[str(query_id)]
        corpus_text = [self.corpus[id]['text'] for id in context_id]
        return {
            "query_text": query_text,
            "corpus_text": corpus_text,
        }

if __name__ == "__main__":
    fiqa_data = FiqaDataset("test")
    print(f"Number of samples in FiQA train dataset: {len(fiqa_data)}")
    print(f"Sample data: {fiqa_data.__getitem__(58)}")
