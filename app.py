import streamlit as st
import torch
import re
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- UI Setup ---
st.set_page_config(page_title="MFU Academy Assistant", layout="centered")
st.title("🎓 MFU Academy Course Assistant")
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

@st.cache_resource
def load_rag_bot():
    # Load and Split Data
    loader = TextLoader("mfu_academy_data.txt", encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    # Create Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Load Typhoon2 3B Model
    model_id = "scb10x/llama3.2-typhoon2-3b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=512, 
        temperature=0.3,
        top_p=0.9
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # System Prompt
    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    คุณคือผู้ช่วยของ MFU Academy ตอบคำถามจาก CONTEXT เท่านั้น 
    - ห้ามพิมพ์ Q: หรือ A: นำหน้าคำตอบ
    - ห้ามทวนคำถามของผู้ใช้
    CONTEXT: {context}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    
    return RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        chain_type_kwargs={"prompt": PROMPT}
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
