from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text):
    return model.encode([text])[0]


def build_job_index(jobs):

    texts = []
    for job in jobs:
        text = f"{job['title']} {job['company']} {job['description']}"
        texts.append(text)

    embeddings = model.encode(texts)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, embeddings


def search_jobs(cv_text, jobs, index, top_k=5):

    cv_vector = embed_text(cv_text)

    distances, indices = index.search(
        np.array([cv_vector]), top_k
    )

    results = []

    for idx in indices[0]:
        if idx < len(jobs):
            results.append(jobs[idx])

    return results
