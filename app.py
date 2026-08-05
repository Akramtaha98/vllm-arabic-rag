"""
Streamlit demo UI for:
Semantic-Driven Context Pruning for Optimizing Arabic RAG Systems
in Memory-Constrained vLLM Deployments

Run locally:
    streamlit run app.py

Configure via environment variables (see .env.example):
    VLLM_API_URL, VLLM_MODEL_NAME, VLLM_API_KEY, VLLM_METRICS_URL, PRUNER_MODEL_NAME
"""

import os
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from middleware import (
    SemanticPruner,
    DynamicRatioController,
    DynamicRatioConfig,
    VLLMClient,
)
from middleware.retriever import SimpleRetriever, MOCK_CORPUS

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
VLLM_API_URL = os.getenv("VLLM_API_URL", "http://localhost:8000/v1/chat/completions")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
VLLM_METRICS_URL = os.getenv("VLLM_METRICS_URL", "http://localhost:8000/metrics")
VLLM_API_KEY = os.getenv("VLLM_API_KEY")  # required for NVIDIA NIM (build.nvidia.com), unused for local vLLM

PRUNER_MODELS = {
    "⚡ Fast (multilingual MiniLM)": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "🎯 Accurate (BGE-reranker-v2-m3)": "BAAI/bge-reranker-v2-m3",
}
DEFAULT_PRUNER_LABEL = "⚡ Fast (multilingual MiniLM)"

EXAMPLE_QUESTIONS = [
    "متى تأسست جامعة الملك سعود؟",
    "ما هي اهتمامات قسم الحاسب في الجامعة؟",
    "كيف يكون الطقس في الرياض؟",
]

st.set_page_config(
    page_title="Arabic RAG Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', 'IBM Plex Sans Arabic', sans-serif;
    }

    .rtl-text {
        direction: rtl;
        text-align: right;
        font-family: 'IBM Plex Sans Arabic', sans-serif;
        font-size: 1.05rem;
        line-height: 1.9;
    }

    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #6C4CF1 0%, #9B5DE5 45%, #22D3EE 100%);
        margin-bottom: 1.4rem;
    }
    .hero h1 {
        color: white;
        font-size: 1.7rem;
        margin: 0 0 0.35rem 0;
        font-weight: 700;
    }
    .hero p {
        color: rgba(255,255,255,0.92);
        margin: 0;
        font-size: 0.95rem;
    }
    .badge-row { margin-top: 0.7rem; }
    .badge {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        color: white;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        margin-right: 6px;
        border: 1px solid rgba(255,255,255,0.3);
    }

    .card {
        background: rgba(127,127,127,0.06);
        border: 1px solid rgba(127,127,127,0.15);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        opacity: 0.65;
        margin-bottom: 0.6rem;
    }

    .answer-card {
        background: linear-gradient(135deg, rgba(108,76,241,0.10), rgba(34,211,238,0.10));
        border: 1px solid rgba(108,76,241,0.30);
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
    }

    .stat-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        background: rgba(34,211,238,0.15);
        color: #22D3EE;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }

    div[data-testid="stButton"] button {
        border-radius: 10px;
        font-weight: 600;
    }

    .chip button {
        border-radius: 999px !important;
        font-size: 0.8rem !important;
        padding: 2px 14px !important;
        opacity: 0.85;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_pruner(model_name: str):
    return SemanticPruner(model_name=model_name)


@st.cache_resource(show_spinner=False)
def load_client():
    return VLLMClient(api_url=VLLM_API_URL, model_name=VLLM_MODEL_NAME, api_key=VLLM_API_KEY)


@st.cache_resource(show_spinner=False)
def load_retriever():
    return SimpleRetriever()


client = load_client()
retriever = load_retriever()
dynamic_controller = DynamicRatioController(DynamicRatioConfig(metrics_url=VLLM_METRICS_URL))

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    pruner_label = st.selectbox(
        "Pruning model",
        list(PRUNER_MODELS.keys()),
        index=list(PRUNER_MODELS.keys()).index(DEFAULT_PRUNER_LABEL),
        help="Fast model = sub-second pruning. Accurate model = stronger relevance ranking, slower on CPU.",
    )
    pruner_model_name = PRUNER_MODELS[pruner_label]

    mode = st.radio("Compression mode", ["Fixed ratio", "Dynamic (vLLM load-aware)"], index=0)

    if mode == "Fixed ratio":
        comp_ratio = st.slider("Compression ratio", 0.1, 1.0, 0.5, step=0.1)
    else:
        comp_ratio = None
        st.caption("Ratio auto-computed from vLLM's `/metrics` (`vllm:gpu_cache_usage_perc`).")
        c1, c2 = st.columns(2)
        min_r = c1.slider("Min (high load)", 0.05, 0.5, 0.2, step=0.05)
        max_r = c2.slider("Max (low load)", 0.5, 1.0, 0.8, step=0.05)
        dynamic_controller.config.min_ratio = min_r
        dynamic_controller.config.max_ratio = max_r

    top_k = st.slider("Top-K retrieved documents", 1, len(MOCK_CORPUS), 4)

    with st.expander("Advanced"):
        max_tokens = st.slider("Max answer tokens", 64, 1024, 300, step=32)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, step=0.05)
        use_streaming = st.toggle("Stream answer live", value=True)

    st.markdown("---")
    st.caption("Backend")
    st.code(VLLM_API_URL, language=None)
    st.caption(f"Model: `{VLLM_MODEL_NAME}`")
    st.caption(f"Pruner: `{pruner_model_name.split('/')[-1]}`")

pruner = load_pruner(pruner_model_name)

# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>⚡ Arabic RAG Optimizer</h1>
        <p>Semantic-driven context pruning that shrinks retrieved Arabic documents before they hit the LLM —
        less KV-cache pressure, faster answers, same accuracy.</p>
        <div class="badge-row">
            <span class="badge">vLLM-ready</span>
            <span class="badge">Cross-encoder pruning</span>
            <span class="badge">Arabic-first</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Query input
# --------------------------------------------------------------------------
st.markdown("**جرّب سؤالاً:**")
chip_cols = st.columns(len(EXAMPLE_QUESTIONS))
for i, q in enumerate(EXAMPLE_QUESTIONS):
    with chip_cols[i]:
        st.markdown('<div class="chip">', unsafe_allow_html=True)
        if st.button(q, key=f"chip_{i}", use_container_width=True):
            st.session_state.query_input = q
        st.markdown("</div>", unsafe_allow_html=True)

user_query = st.text_input(
    "أدخل سؤالك هنا:",
    key="query_input",
    placeholder="اكتب سؤالك بالعربية...",
    label_visibility="collapsed",
)

run = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
if run:
    if not user_query.strip():
        st.warning("الرجاء إدخال سؤال.")
        st.stop()

    step_progress = st.progress(0, text="Retrieving documents...")

    # 1) Retrieval
    t_retrieve = time.perf_counter()
    docs = retriever.retrieve(user_query, top_k=top_k)
    retrieve_ms = (time.perf_counter() - t_retrieve) * 1000
    step_progress.progress(25, text="Pruning context semantically...")

    # 2) Determine compression ratio
    effective_ratio = comp_ratio if comp_ratio is not None else dynamic_controller.get_ratio()

    # 3) Semantic pruning
    result = pruner.prune(user_query, docs, compression_ratio=effective_ratio)
    step_progress.progress(55, text="Generating answer...")

    reduction_pct = (
        100 * (1 - result.pruned_char_count / result.original_char_count)
        if result.original_char_count
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sentences kept", f"{result.kept_sentence_count}/{result.original_sentence_count}")
    col2.metric("Context reduced", f"{reduction_pct:.0f}%")
    col3.metric("Pruning time", f"{result.latency_ms:.0f} ms")
    col4.metric(
        "Ratio used",
        f"{effective_ratio:.2f}",
        help="Fixed slider value, or auto-resolved from vLLM load in Dynamic mode.",
    )

    with st.expander("📄 Retrieved documents & pruning detail"):
        st.markdown("**Raw retrieved chunks:**")
        for d in docs:
            st.markdown(f'<div class="rtl-text">• {d}</div>', unsafe_allow_html=True)
        st.markdown("**Dropped sentences:**")
        for d in result.dropped_sentences:
            st.markdown(f'<div class="rtl-text" style="opacity:0.55;">✗ {d}</div>', unsafe_allow_html=True)
        st.caption(f"Retrieval latency: {retrieve_ms:.1f} ms")

    st.markdown('<div class="card"><div class="card-title">Optimized context sent to the model</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="rtl-text">{result.pruned_text}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 4) Call the model
    system_instruction = "أنت مساعد ذكي تجيب بدقة بالاعتماد فقط على السياق المتاح لك وبشكل مختصر."
    full_prompt = f"السياق: {result.pruned_text}\n\nالسؤال: {user_query}"

    st.markdown('<div class="answer-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">💬 Answer</div>', unsafe_allow_html=True)

    t_gen = time.perf_counter()
    if use_streaming:
        def _stream():
            for chunk in client.chat_stream(
                system_instruction, full_prompt, temperature=temperature, max_tokens=max_tokens
            ):
                yield chunk

        answer_text = st.write_stream(_stream())
        gen_ms = (time.perf_counter() - t_gen) * 1000
        answer_error = isinstance(answer_text, str) and answer_text.startswith("[vLLM")
    else:
        with st.spinner("Awaiting response..."):
            response = client.chat(
                system_instruction, full_prompt, temperature=temperature, max_tokens=max_tokens
            )
        st.markdown(f'<div class="rtl-text">{response.text}</div>', unsafe_allow_html=True)
        gen_ms = response.total_latency_ms
        answer_error = response.text.startswith("[vLLM")

    st.markdown("</div>", unsafe_allow_html=True)
    step_progress.progress(100, text="Done")
    step_progress.empty()

    total_pipeline_ms = retrieve_ms + result.latency_ms + gen_ms
    st.markdown(
        f"""
        <div style="margin-top:0.6rem;">
            <span class="stat-pill">⏱️ Total: {total_pipeline_ms/1000:.1f}s</span>
            <span class="stat-pill">🔍 Retrieval: {retrieve_ms:.0f}ms</span>
            <span class="stat-pill">✂️ Pruning: {result.latency_ms:.0f}ms</span>
            <span class="stat-pill">🤖 Generation: {gen_ms:.0f}ms</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if answer_error:
        st.error(
            "Could not reach the model backend. Check `VLLM_API_URL` / `VLLM_API_KEY` in `.env` "
            "(see README for local vLLM or NVIDIA NIM setup)."
        )
else:
    st.markdown(
        """
        <div class="card" style="text-align:center; padding: 2.4rem 1rem; opacity:0.75;">
            Pick an example above or type your own Arabic question, then hit
            <b>Run Pipeline</b> to see retrieval → pruning → generation end to end.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div style="text-align:center; opacity:0.5; font-size:0.78rem; margin-top:2rem;">
        Semantic-Driven Context Pruning for Arabic RAG · built for vLLM-based deployments
    </div>
    """,
    unsafe_allow_html=True,
)
