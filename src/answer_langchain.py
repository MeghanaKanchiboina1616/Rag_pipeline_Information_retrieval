
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

try:
    from .retrieve import retrieve
except ImportError:
    from retrieve import retrieve


OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a Kubernetes troubleshooting assistant for DevOps engineers.

Your task is to analyze the user's question using the supplied
Kubernetes documentation as evidence.

Follow these rules carefully:

1. Explain what the reported Kubernetes issue means.

2. Identify plausible root causes supported by the retrieved
   documentation. Prioritize causes directly relevant to the question.

3. Give safe, read-only diagnostic commands where relevant.

4. Explain what the diagnostic results could indicate.

5. Cite the supporting source filenames provided in the context.

6. Never invent pod names, namespaces, logs, events, or actual
   cluster findings.

7. Clearly distinguish documented facts, possible causes, and
   confirmed causes. A possible cause is not confirmed without
   supporting diagnostic evidence.

8. If the retrieved evidence is insufficient to determine the
   root cause, explicitly state what additional information is needed.

9. Never claim that you accessed or inspected a live Kubernetes cluster.

10. IMPORTANT: CrashLoopBackOff and ImagePullBackOff are distinct
    Kubernetes container status conditions.

    - CrashLoopBackOff indicates that a container repeatedly fails
      and Kubernetes is delaying another restart attempt.
    - ImagePullBackOff indicates that Kubernetes cannot successfully
      pull a required container image and is backing off before
      retrying.

    Do not present ImagePullBackOff as a direct cause of
    CrashLoopBackOff. If the question concerns CrashLoopBackOff,
    prioritize relevant causes such as application startup failures,
    process exit codes, configuration errors, missing dependencies,
    resource problems, and failed health checks when supported by
    the retrieved evidence.

    Mention ImagePullBackOff only when the retrieved evidence or
    reported symptoms indicate a separate image-pull problem or
    another relevant image-related diagnostic condition.

11. Do not claim that a possible cause is confirmed without
    supporting logs, Kubernetes events, exit codes, or other
    diagnostic evidence.

12. Do not list unrelated Kubernetes failure states merely because
    they appear in the retrieved documentation. Explain their
    relevance before mentioning them.

13. Prefer a small number of relevant, evidence-supported causes
    over a long list of speculative possibilities.

14. If different retrieved documents describe different failure
    conditions, keep those conditions separate and explain their
    differences accurately.

Retrieved documentation:
{context}
""",
        ),
        (
            "human",
            "Troubleshooting question:\n{question}\n\n"
            "Provide a clear, structured diagnostic analysis.",
        ),
    ]
)


def format_context(results, max_docs=4, max_chars=1800):
    """Format retrieved documents as evidence for the LLM."""

    if not results:
        return "No relevant documentation was retrieved."

    sections = []
    seen_paths = set()

    for item in results:
        path = (
            item.get("path")
            or item.get("source")
            or "Unknown source"
        )

        if path in seen_paths:
            continue

        document = (
            item.get("document")
            or item.get("text")
            or item.get("content")
            or ""
        ).strip()

        if not document:
            continue

        seen_paths.add(path)

        sections.append(
            f"Source: {path}\n"
            f"Documentation:\n{document[:max_chars]}"
        )

        if len(sections) >= max_docs:
            break

    if not sections:
        return "Retrieved results contained no readable document text."

    return "\n\n---\n\n".join(sections)


def generate_answer(question):
    """Generate an evidence-grounded answer using LangChain and Ollama."""

    question = question.strip()

    if not question:
        raise ValueError("Please enter a troubleshooting question.")

    # Reuse the existing hybrid retrieval pipeline:
    # BM25 + BGE dense retrieval + RRF + BGE reranking.
    results = retrieve(question)

    # Prepare the retrieved documentation for the LLM.
    context = format_context(results)

    # Connect LangChain to the local Ollama model.
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
    )

    # LangChain chain:
    # Prompt -> ChatOllama -> String output parser.
    chain = PROMPT | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    return answer, results
