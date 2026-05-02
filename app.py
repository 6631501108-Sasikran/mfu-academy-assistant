import os
import re
from typing import Any, List, Optional
import numpy as np
import streamlit as st
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


class HFAPIEmbeddings(Embeddings):
    """Embeddings via HuggingFace Inference API using InferenceClient.feature_extraction."""

    def __init__(self, model_name: str, token: str):
        self._client = InferenceClient(model=model_name, token=token)

    def _embed(self, text: str):
        out = self._client.feature_extraction(text)
        arr = np.asarray(out, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        return arr.tolist()

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


class HFAPILLM(LLM):
    """LLM via HuggingFace Inference API using InferenceClient.text_generation."""

    model_name: str
    token: str
    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9

    @property
    def _llm_type(self) -> str:
        return "hf_inference_api"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        client = InferenceClient(model=self.model_name, token=self.token)
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return response.choices[0].message.content

# --- UI Setup ---
st.set_page_config(page_title="MFU Academy Assistant", layout="centered")
st.title("🎓 BDA_Project2_Group4 ")
st.markdown("""
สอบถามข้อมูลคอร์สเรียน MFU Academy ได้ที่นี่ครับ <br>
**Group member:**
1. 6631501034 - Nataporn Dibdee
2. 6631501064 - Nawamol Nuanyai
3. 6631501107 - Withara Tangchai
4. 6631501108 - Sasikran Sawangtem
5. 6631501112 - Sawitree Mekao
6. 6631501114 - Soontharee Chaichompoo
7. 6631501126 - Araya Logniyom
8. 6631501163 - Parichat Phojan
<hr>
""", unsafe_allow_html=True)

HF_TOKEN = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    st.error("❌ ไม่พบ HF_TOKEN — กรุณาตั้งค่าใน Streamlit Secrets (Settings → Secrets) หรือ environment variable")
    st.stop()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

@st.cache_resource
def load_rag_bot():
    loader = TextLoader("mfu_academy_data.txt", encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    embeddings = HFAPIEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        token=HF_TOKEN,
    )
    vectorstore = FAISS.from_documents(docs, embeddings)

    llm = HFAPILLM(
        model_name="Qwen/Qwen2.5-72B-Instruct",
        token=HF_TOKEN,
        max_new_tokens=512,
        temperature=0.3,
        top_p=0.9,
    )

    template = """คุณคือผู้ช่วยของ MFU Academy ตอบคำถามเป็นภาษาไทยจาก CONTEXT ที่ให้มาเท่านั้น
- ห้ามพิมพ์ Q: หรือ A: นำหน้าคำตอบ
- ห้ามทวนคำถามของผู้ใช้
- ถ้าไม่มีข้อมูลใน CONTEXT ให้ตอบว่า "ขออภัย ไม่มีข้อมูลในระบบครับ"

CONTEXT:
{context}

คำถาม: {question}
คำตอบ:"""

    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        chain_type_kwargs={"prompt": PROMPT},
    )

qa_chain = load_rag_bot()

# --- Chat Interface Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("พิมพ์คำถามเกี่ยวกับคอร์สเรียนที่นี่..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาข้อมูล..."):
            result = qa_chain.invoke(prompt)
            answer = result["result"]
            answer = re.sub(r"^(assistant|A\s*:|Answer\s*:)\s*", "", answer.strip(), flags=re.IGNORECASE)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
