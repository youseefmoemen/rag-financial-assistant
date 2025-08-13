# rag-financial-assistant

A **Retrieval-Augmented Generation (RAG)** system tailored for financial question answering, fine-tuned using **PEFT techniques** (LoRA, AdaLoRA) and tracked with **MLflow** for experiment management.  
This project integrates **LlamaIndex** for retrieval and Hugging Face transformer models for generation.

The **Financial RAG Assistant** answers domain-specific questions using:
- A document retriever (vector search via LlamaIndex)
- A fine-tuned Retrival model (LoRA / AdaLoRA)
- Experiment tracking with MLflow for all model configurations

**Key Features:**
- Supports multiple PEFT types (**LoRA**, **AdaLoRA**)
- MLflow integration for training, evaluation, and comparison
- Dockerized for easy deployment
- API built with FastAPI for real-time inference


# Training Retrival
