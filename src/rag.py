import os, json, math, heapq
import openai
EMB_MODEL="text-embedding-3-small"
INDEX_FILE=os.path.join("docs","index.json")
openai.api_key = os.getenv("OPENAI_API_KEY","")
def _dot(a,b): return sum(xy for x,y in zip(a,b))
def _norm(a): return math.sqrt(sum(xx for x in a))
def _cos(a,b):
    if a is None or b is None: return -1.0
    return _dot(a,b)/(_norm(a)*_norm(b)+1e-12)
def _retrieve(q,k=4):
    if not os.path.exists(INDEX_FILE): return []
    items=json.load(open(INDEX_FILE,"r",encoding="utf-8"))["items"]
    if not openai.api_key:
        return [{"source":it["doc_id"],"snippet":it["chunk"]} for it in items[:k]]
    q_emb=openai.Embedding.create(input=q, model=EMB_MODEL)["data"][0]["embedding"]
    heap=[]
    for it in items:
        sim=_cos(q_emb,it.get("embedding"))
        heapq.heappush(heap,(-sim,it))
    res=[]
    for _ in range(min(k,len(heap))):
        _,it=heapq.heappop(heap)
        res.append({"source":it["doc_id"],"snippet":it["chunk"]})
    return res
def answer_query(q):
    sources=_retrieve(q)
    context="\n\n".join([s["snippet"] for s in sources])
    if not openai.api_key:
        return {"answer":"OPENAI_API_KEY not set. Set in Space secrets to enable LLM.", "sources":sources}
    prompt=f"You are a Regulatory QA Assistant. Use context:\n{context}\nQuestion:\n{q}\nAnswer with bullets and cite sources."
    resp=openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=800)
    return {"answer":resp["choices"][0]["message"]["content"], "sources":sources}
