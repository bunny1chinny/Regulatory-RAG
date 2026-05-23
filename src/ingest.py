import os, uuid, json
import pdfplumber, openai
EMB_MODEL = "text-embedding-3-small"
DOC_STORE = "docs"
INDEX_FILE = os.path.join(DOC_STORE,"index.json")
openai.api_key = os.getenv("OPENAI_API_KEY","")
def _ensure():
    os.makedirs(DOC_STORE, exist_ok=True)
    if not os.path.exists(INDEX_FILE):
        with open(INDEX_FILE,"w",encoding="utf-8") as f:
            json.dump({"items":[]}, f)
def _chunks(text,size=800):
    paras=[p.strip() for p in text.split("\n\n") if p.strip()]
    chunks=[]
    for p in paras:
        for i in range(0,len(p),size):
            chunks.append(p[i:i+size])
    return chunks
def ingest_document(uploaded_file):
    _ensure()
    doc_id=str(uuid.uuid4())
    path=os.path.join(DOC_STORE,f"{doc_id}.txt")
    if uploaded_file.type=="application/pdf" or uploaded_file.name.endswith(".pdf"):
        with pdfplumber.open(uploaded_file) as pdf, open(path,"w",encoding="utf-8") as out:
            for p in pdf.pages:
                out.write((p.extract_text() or "") + "\n\n")
    else:
        txt = uploaded_file.getvalue().decode("utf-8")
        open(path,"w",encoding="utf-8").write(txt)
    text=open(path,"r",encoding="utf-8").read()
    chunks=_chunks(text)
    items=[]
    for c in chunks:
        emb=None
        if openai.api_key:
            resp=openai.Embedding.create(input=c, model=EMB_MODEL)
            emb=resp["data"][0]["embedding"]
        items.append({"doc_id":doc_id,"chunk":c,"embedding":emb})
    idx=json.load(open(INDEX_FILE,"r",encoding="utf-8"))
    idx["items"].extend(items)
    with open(INDEX_FILE,"w",encoding="utf-8") as f:
        json.dump(idx,f,ensure_ascii=False)
    return doc_id
