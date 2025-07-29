from llama_index.core import VectorStoreIndex, Document
from tqdm import tqdm
from llama_index.core import Settings


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

def create_index(data_loader, model_name: str, embed_model=None) -> VectorStoreIndex:
    if embed_model is not None:
        Settings.embed_model = embed_model
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
