from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import PrivateAttr
import torch

class PEFTEmbeddingModel(BaseEmbedding):
    _model: torch.nn.Module = PrivateAttr()
    _tokenizer = PrivateAttr()
    _device: str = PrivateAttr()

    def __init__(self, sentence_transformer, model_name=None, device=None):
        super().__init__()
        # Extract underlying transformer and tokenizer
        self._model = sentence_transformer.model._first_module().auto_model
        self._tokenizer = sentence_transformer.model.tokenizer
        self._device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = self._model.to(self._device)
        self.model_name = model_name

    def _get_text_embedding(self, text: str) -> list[float]:
        inputs = self._tokenizer([text], return_tensors="pt", truncation=True, padding=True).to(self._device)
        with torch.no_grad():
            output = self._model(**inputs)
            if hasattr(output, "pooler_output") and output.pooler_output is not None:
                embedding = output.pooler_output
            else:
                embedding = output.last_hidden_state.mean(dim=1)
            embedding = embedding.squeeze().cpu().tolist()
        return embedding

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)
