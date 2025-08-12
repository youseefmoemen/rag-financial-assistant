from sentence_transformers import SentenceTransformer
from peft import PeftModel
from src.utils.base_embedding import PEFTEmbeddingModel
from llama_index.llms.huggingface import HuggingFaceLLM
import warnings
from llama_index.core.chat_engine.types import BaseChatEngine
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn as uv
import time
from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage
import logging

warnings.filterwarnings(
    "ignore",
    message="`encoder_attention_mask` is deprecated and will be removed in version 4.55.0 for `BertSdpaSelfAttention.forward`.",
    category=FutureWarning,
    module="torch.nn.modules.module"
)


class Query(BaseModel):
    input_msg: str


class Response(BaseModel):
    answer: str
    elapsed_time: float


app = FastAPI(
    title="Financial RAG",
    description="RAG pipeline for financial document QA",
)

def set_embedding_model(model_name: str, adapter_path: str) -> bool:
    try:
        base_model = SentenceTransformer(model_name)
        peft_model = PeftModel.from_pretrained(base_model, adapter_path)
        embbed_model = PEFTEmbeddingModel(peft_model, model_name)
        Settings.embed_model = embbed_model
        return True
    except Exception:
        return False 

def load_chat_engine(generator_name: str, index_path: str) -> BaseChatEngine:
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context=storage_context)
    llm = HuggingFaceLLM(model_name=generator_name, tokenizer_name=generator_name)
    chat_engine = index.as_chat_engine(llm=llm)
    return chat_engine


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}



@app.post('/')
async def respond(query: Query) -> Response:
    t0 = time.time()
    response = chat_engine.chat(query.input_msg)
    t1 = time.time()
    dt = t1 - t0
    return Response(answer=str(response), elapsed_time=dt)

if __name__ == '__main__':
    index_path = "data/index_store"
    generator_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_name = "sentence-transformers/all-mpnet-base-v2"
    adapterh_path = "models/all-mpnet-base-v2-adalora-best/adapter"
    retrival_statue = set_embedding_model(
        model_name=model_name,
        adapter_path=adapterh_path
    )
    if retrival_statue:
        logging.log('RetrivalLoadedSucessfully')
    else:
        logging.log("SomwthingWentWrong")
    chat_engine = load_chat_engine(
        generator_name=generator_name,
        index_path=index_path
    )
    uv.run(app, host='0.0.0.0', port=8000)



