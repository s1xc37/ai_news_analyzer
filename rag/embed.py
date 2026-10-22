import json
import numpy as np


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('intfloat/multilingual-e5-large')

def load_chunks(path):
    content = []
    with open(path) as f:
        json_list = list(f)
        for json_str in json_list:
            content.append(json.loads(json_str))
    return content

chunks = load_chunks("./rag/chunks.jsonl")

texts = ["passage: " + c["text"] for c in chunks]
embeddings = model.encode(texts, normalize_embeddings=True)

np.save("rag/embeddings.npy", embeddings)
