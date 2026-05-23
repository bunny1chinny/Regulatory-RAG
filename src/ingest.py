import os, uuid, json, openai
DOC_STORE = "docs"
INDEX_FILE = os.path.join(DOC_STORE, "index.json")
EMB_MODEL = "text-embedding-3-small"
openai.api_key = os.getenv("OPENAI_API_KEY", "")

def _ensure():
    os.makedirs(DOC_STORE, exist_ok=True)
    if not os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump({"items": []}, f)

def _chunks(text, size=800):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for p in paras:
        for i in range(0, len(p), size):
            chunks.append(p[i:i+size])
    return chunks

def ingest_document(uploaded_file):
    _ensure()
    if uploaded_file.type not in ("text/plain",):
        raise ValueError("Only plain .txt uploads supported in this demo. Convert PDFs to text first.")
    doc_id = str(uuid.uuid4())
    path = os.path.join(DOC_STORE, f"{doc_id}.txt")
    text = uploaded_file.getvalue().decode("utf-8")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    chunks = _chunks(text)
    items = []
    for c in chunks:
        emb = None
        if openai.api_key:
            resp = openai.Embedding.create(input=c, model=EMB_MODEL)
            emb = resp["data"][0]["embedding"]
        items.append({"doc_id": doc_id, "chunk": c, "embedding": emb})
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        idx = json.load(f)
    idx["items"].extend(items)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False)
    return doc_id
