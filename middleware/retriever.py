"""
Minimal vector-store retriever wrapper (Chroma) for the Arabic RAG demo.
Falls back to an in-memory mock corpus if no Chroma collection is populated,
so the app runs out-of-the-box without requiring a pre-built index.
"""

from __future__ import annotations

from typing import List, Optional

MOCK_CORPUS: List[str] = [
    "تعتبر جامعة الملك سعود من أقدم الجامعات في المملكة العربية السعودية وتضم كليات علمية متعددة وطاقم بحثي مميز.",
    "تأسست الجامعة في عام 1957 وتقع في مدينة الرياض وتساهم بشكل كبير في نشر أوراق الذكاء الاصطناعي.",
    "الطقس في مدينة الرياض حار جداً خلال فصل الصيف وتتجاوز درجات الحرارة في العادة 45 درجة مئوية.",
    "يركز قسم علوم الحاسب بالجامعة على أبحاث معالجة اللغة العربية والأنظمة الموزعة لتسريع خوادم الاستدلال للنماذج الضخمة.",
    "يضم الحرم الجامعي مكتبة مركزية ضخمة تحتوي على أكثر من مليوني كتاب ومرجع علمي في شتى المجالات.",
    "أطلقت الجامعة مؤخراً مركزاً متخصصاً لأبحاث الذكاء الاصطناعي التوليدي بالتعاون مع شركات تقنية عالمية.",
]


class SimpleRetriever:
    """
    Wraps a Chroma collection if provided; otherwise serves from MOCK_CORPUS
    using a naive keyword-overlap ranking so the demo works without a DB.
    """

    def __init__(self, chroma_collection=None):
        self.collection = chroma_collection

    def retrieve(self, query: str, top_k: int = 4) -> List[str]:
        if self.collection is not None:
            results = self.collection.query(query_texts=[query], n_results=top_k)
            docs = results.get("documents", [[]])[0]
            return docs

        # Fallback: naive overlap scoring over the mock corpus.
        q_tokens = set(query.split())
        scored = []
        for doc in MOCK_CORPUS:
            overlap = len(q_tokens & set(doc.split()))
            scored.append((overlap, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [doc for _, doc in scored[:top_k]] if any(s for s, _ in scored) else MOCK_CORPUS[:top_k]
        return top


def build_chroma_collection(persist_dir: str = "./chroma_db", collection_name: str = "arabic_docs"):
    """
    Optional helper: build/load a real Chroma collection from a directory of
    .txt files (one document per file). Requires `chromadb` installed.
    """
    import glob
    import os

    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=persist_dir)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-large"
    )
    collection = client.get_or_create_collection(name=collection_name, embedding_function=ef)

    docs_dir = os.path.join(persist_dir, "..", "corpus")
    if os.path.isdir(docs_dir) and collection.count() == 0:
        files = glob.glob(os.path.join(docs_dir, "*.txt"))
        ids, texts = [], []
        for i, f in enumerate(files):
            with open(f, "r", encoding="utf-8") as fh:
                texts.append(fh.read())
            ids.append(f"doc_{i}")
        if texts:
            collection.add(documents=texts, ids=ids)

    return collection
