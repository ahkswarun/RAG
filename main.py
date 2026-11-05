import os
import streamlit as st

from src.rag_setup import get_rag


st.set_page_config(
    page_title="Document Intelligence System",
    page_icon="📚",
    layout="wide"
)


@st.cache_resource(show_spinner=False)
def load_rag(pdf_path):
    return get_rag(pdf_path)


def init_session(pdf_name: str):
    if (
        "chat_pdf" not in st.session_state
        or st.session_state.chat_pdf != pdf_name
    ):
        st.session_state.chat_pdf = pdf_name
        st.session_state.messages = []


def render_debug(result: dict):
    retrieved = result.get("retrieved_chunks", [])
    search_query = result.get("search_query")

    with st.expander("🔍 Retrieval debug", expanded=False):
        if search_query:
            st.caption(f"Search query used: {search_query}")

        if not retrieved:
            st.warning(
                "No chunks passed the distance filter — retrieval found nothing useful."
            )
        else:
            st.success(
                f"{len(retrieved)} chunk(s) retrieved and sent to the LLM."
            )
            for i, chunk in enumerate(retrieved, start=1):
                distance = chunk.get("distance")
                label = f"Chunk {i} — Page {chunk['page']}"
                if distance is not None:
                    label += f" (distance={distance:.4f})"
                with st.expander(label):
                    st.write(chunk["text"])

        if result.get("sources"):
            st.markdown("**Sources**")
            for source in result["sources"]:
                st.markdown(f"- Page {source['page']}")


st.title("📚 Document Intelligence System")

st.markdown(
    "Upload a PDF and chat with it. Follow-up questions stay in this session."
)

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file is None:
    st.info("Please upload a PDF to begin.")
    st.stop()


upload_dir = "uploads"
os.makedirs(upload_dir, exist_ok=True)

pdf_path = os.path.join(upload_dir, uploaded_file.name)

with open(pdf_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

init_session(uploaded_file.name)

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

try:
    with st.spinner("Loading document..."):
        rag = load_rag(pdf_path)
except Exception as exc:
    st.error("Failed to initialize the RAG system.")
    st.exception(exc)
    st.stop()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and message.get("result"):
            render_debug(message["result"])


prompt = st.chat_input("Ask a question about the document")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.write(prompt)

    history_for_rag = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = rag.ask(prompt, chat_history=history_for_rag)

        st.write(result["answer"])
        render_debug(result)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "result": result,
        }
    )
