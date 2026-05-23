import streamlit as st
from src import ingest, rag, ecm_export, utils
import os
st.set_page_config(page_title="Regulatory RAG Demo", layout="wide")
st.title("Regulatory RAG Prototype — eCTD Scaffold & QA")
st.markdown("Upload a small SOP or dossier, ingest, ask RAG Q&A, generate scaffold.")
with st.sidebar:
    uploaded = st.file_uploader("Upload TXT file", type=["txt"])
    if st.button("Ingest"):
        if uploaded:
            try:
                doc_id = ingest.ingest_document(uploaded)
                st.success(f"Ingested: {doc_id}")
            except Exception as e:
                st.error(f"Ingest failed: {e}")
        else:
            st.error("Upload a file first")
st.header("RAG Q&A")
query = st.text_area("Ask a question", height=140)
if st.button("Get Answer"):
    if not query.strip():
        st.warning("Enter a question.")
    else:
        with st.spinner("Retrieving..."):
            ans = rag.answer_query(query)
        st.subheader("Answer")
        st.write(ans.get("answer","No answer"))
        st.subheader("Sources")
        for s in ans.get("sources",[]):
            st.markdown(f"- {s.get('source','')[:8]} — {s.get('snippet','')[:300]}")
st.header("eCTD Scaffold")
if st.button("Generate scaffold"):
    scaffold = ecm_export.generate_scaffold()
    st.download_button("Download scaffold.json", data=utils.to_json(scaffold), file_name="scaffold.json")
st.sidebar.markdown("### Workspace status")
st.sidebar.write("Index exists:", os.path.exists("docs/index.json"))
