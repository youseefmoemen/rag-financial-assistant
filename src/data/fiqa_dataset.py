from beir.datasets.data_loader import GenericDataLoader
from pathlib import Path
from torch.utils.data import DataLoader
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
        corpus_id = [context for context, score in relevances.items() if score > 0]
        query_text = self.queries[str(query_id)]
        corpus_text = [self.corpus[id]['text'] for id in corpus_id]
        corpus_text = corpus_text[:4] + ['0'] * (4 - len(corpus_text))
        corpus_id = corpus_id[:4] + [-1] * (4 - len(corpus_id))
        return {
            "query_id": query_id,
            "corpus_id": corpus_id,
            "query_text": query_text,
            "corpus_text": corpus_text,
        }

def collate_fn(batch):
    return batch

if __name__ == "__main__":
    fiqa_data = FiqaDataset("test")
    dataloader = DataLoader(fiqa_data, batch_size=32, collate_fn=collate_fn)
    for batch in dataloader:
        print(batch)
        break
    print(f"Number of samples in FiQA train dataset: {len(fiqa_data)}")
