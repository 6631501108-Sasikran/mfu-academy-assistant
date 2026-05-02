import os
import re
from typing import Any, List, Optional
import streamlit as st
from huggingface_hub import InferenceClient
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun


class HFAPILLM(LLM):
    """LLM via HuggingFace Inference API using InferenceClient.chat_completion."""

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
st.title("🎓 MFU BDA_Project2_Group4")
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
def load_knowledge_base():
    with open("mfu_academy_data.txt", "r", encoding="utf-8") as f:
        return f.read()


@st.cache_resource
def load_llm():
    return HFAPILLM(
        model_name="Qwen/Qwen2.5-72B-Instruct",
        token=HF_TOKEN,
        max_new_tokens=768,
        temperature=0.2,
        top_p=0.9,
    )


SYSTEM_INSTRUCTION = """คุณคือผู้ช่วยของ MFU Academy ตอบคำถามเป็นภาษาไทยจากข้อมูลที่ให้มาเท่านั้น
- ตอบให้ครบถ้วนและตรงคำถาม โดยอ้างอิงจากข้อมูลใน KNOWLEDGE BASE ด้านล่าง
- ห้ามพิมพ์ Q:, A:, "คำตอบ:" นำหน้าคำตอบ
- ห้ามทวนคำถามของผู้ใช้
- ถ้าผู้ใช้ถามภาพรวม (เช่น มีกี่คอร์ส, คอร์สอะไรบ้าง, คอร์สไหนฟรี) ให้ตอบโดยอ้างอิงจาก section "รายชื่อคอร์สทั้งหมด"
- ถ้าไม่มีข้อมูลที่เกี่ยวข้อง ให้ตอบว่า "ขออภัย ไม่มีข้อมูลในระบบครับ"
"""


def build_prompt(knowledge: str, question: str) -> str:
    return f"""{SYSTEM_INSTRUCTION}

KNOWLEDGE BASE:
{knowledge}

คำถามผู้ใช้: {question}"""


knowledge = load_knowledge_base()
llm = load_llm()

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
            full_prompt = build_prompt(knowledge, prompt)
            answer = llm.invoke(full_prompt)
            answer = re.sub(r"^(assistant|A\s*:|Answer\s*:|คำตอบ\s*:)\s*", "", answer.strip(), flags=re.IGNORECASE)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
