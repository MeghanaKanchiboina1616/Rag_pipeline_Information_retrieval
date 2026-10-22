import streamlit as st

from src.answer_langchain import generate_answer


st.set_page_config(
    page_title="DevOps RCA - LangChain",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Context-Aware DevOps RCA")
st.caption(
    "LangChain · Hybrid RAG · BM25 · BGE · ChromaDB · "
    "RRF · Cross-Encoder · Qwen2.5"
)

with st.sidebar:
    st.header("About this project")
    st.write(
        "An AI-assisted Kubernetes troubleshooting tool that "
        "retrieves relevant documentation and generates "
        "evidence-grounded diagnostic guidance."
    )

    st.divider()
    st.metric("Knowledge base documents", "338")
    st.metric("Indexed chunks", "2,128")

    st.markdown("**Pipeline**")
    st.write("1. BM25 sparse retrieval")
    st.write("2. BGE dense retrieval")
    st.write("3. Reciprocal Rank Fusion")
    st.write("4. BGE Cross-Encoder reranking")
    st.write("5. LangChain prompt and LLM pipeline")
    st.write("6. Qwen2.5 through Ollama")

    st.caption(
        "Uses stored documentation. No live Kubernetes "
        "cluster is connected."
    )

st.subheader("Ask a troubleshooting question")

examples = [
    "Why is my Kubernetes pod in CrashLoopBackOff?",
    "How do I troubleshoot an admission webhook x509 error?",
    "Why is my Kubernetes ingress controller crashing?",
    "How do I troubleshoot ImagePullBackOff?",
]

selected_question = st.selectbox(
    "Choose an example or enter your own question below",
    ["Choose an example..."] + examples,
)

default_question = (
    selected_question
    if selected_question != "Choose an example..."
    else ""
)

question = st.text_area(
    "Troubleshooting question",
    value=default_question,
    placeholder="Describe the Kubernetes issue...",
    height=100,
)

if st.button("Analyze incident", type="primary"):
    if not question.strip():
        st.warning("Please enter a troubleshooting question.")
    else:
        try:
            with st.spinner(
                "Retrieving evidence and generating analysis..."
            ):
                answer, results = generate_answer(question)

            st.subheader("Diagnostic analysis")
            st.markdown(answer)

            st.subheader("Retrieved evidence")
            st.caption(f"Retrieved evidence: {len(results)} chunks")

            if not results:
                st.info("No relevant documents were retrieved.")

            for index, item in enumerate(results, start=1):
                source = (
                    item.get("path")
                    or item.get("source")
                    or "Unknown source"
                )
                score = item.get("score")

                heading = f"{index}. {source}"

                if score is not None:
                    heading += f" · Score: {score}"

                document = (
                    item.get("document")
                    or item.get("text")
                    or item.get("content")
                    or "No document text available."
                )

                with st.expander(heading):
                    st.text(document.strip())

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.info(
                "Check that Ollama is running, the model is available, "
                "and the existing retrieval index is accessible."
            )