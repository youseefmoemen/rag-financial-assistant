from sentence_transformers import SentenceTransformer


list_of_models = [
    "intfloat/e5-small",
    "intfloat/e5-base",
    'BAAI/bge-small-en-v1.5',
    'BAAI/bge-base-en-v1.5',
    'all-MiniLM-L6-v2',
    'all-MiniLM-L12-v2',
]




if __name__ == '__main__':
    for model_name in list_of_models:
        print(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name)
        print(f"Model {model_name} loaded successfully.")
        # You can add additional code here to test the model or perform inference