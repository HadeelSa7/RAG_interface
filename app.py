"""
app.py - Arabic RAG Research Prototype (Streamlit).

Section 1: ask a question and choose the retrieval methods.
Section 2: compare the retrieved chunks and the generated answers.

Run with:   streamlit run app.py
"""
import html
import json
import os
import time

import streamlit as st

import retrieval

METHODS = ["BM25", "BGE-M3", "Hybrid", "Hybrid + Reranking"]

# The 50 evaluation questions used in the experiments (copy of evaluation_queries_final.json)
QUESTIONS_FILE = os.path.join(os.path.dirname(__file__), "data", "evaluation_queries_final.json")
with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    evaluation_questions = json.load(f)

OWN_QUESTION = "— Write my own question —"

# What the "score" number means for each method
SCORE_NAMES = {
    "BM25": "BM25 score",
    "BGE-M3": "cosine similarity",
    "Hybrid": "RRF score",
    "Hybrid + Reranking": "reranker score",
}

st.set_page_config(page_title="Arabic RAG Research Prototype", layout="wide")

# Small style rule: Arabic text boxes and answers are written right-to-left
st.markdown(
    """
    <style>
    textarea, input[type="text"] { direction: rtl; text-align: right; }
    .arabic { direction: rtl; text-align: right; line-height: 1.9; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_models_once():
    """Load the chunks and models only once (not on every click)."""
    retrieval.load_models()
    return True


def copy_chosen_question():
    """When a question is chosen from the list, write it into the question box."""
    choice = st.session_state["question_choice"]
    if choice != OWN_QUESTION:
        question_id = choice.split(" · ")[0]
        for item in evaluation_questions:
            if item["query_id"] == question_id:
                st.session_state["question_text"] = item["query"]


def show_arabic(text):
    """Show Arabic text right-to-left."""
    safe_text = html.escape(text).replace("\n", "<br>")
    st.markdown(f'<div class="arabic">{safe_text}</div>', unsafe_allow_html=True)


# =====================================================================
# Title
# =====================================================================
st.title("Arabic Information Retrieval Assistant")
st.caption("Intelligent Arabic Information Retrieval & Question Answering")

# =====================================================================
# Section 1 - Ask a question
# =====================================================================
st.header("Ask a question")

question_options = [OWN_QUESTION]
for item in evaluation_questions:
    question_options.append(item["query_id"] + " · " + item["query"])

st.selectbox(
    "Choose one of the 50 evaluation questions (optional)",
    question_options,
    key="question_choice",
    on_change=copy_chosen_question,
)

question = st.text_area(
    "Question (Arabic)",
    placeholder="ما شروط القبول في برامج الدكتوراه؟",
    height=100,
    key="question_text",
)
st.caption("The chosen question appears here and can still be edited.")

st.write("Retrieval methods:")
run_all = st.checkbox("Run all four methods", value=True)

selected_methods = []
for i, method_name in enumerate(METHODS):
    if i % 2 == 0:
        method_columns = st.columns(2)  # a new row for every two methods
    with method_columns[i % 2]:
        if run_all:
            st.checkbox(method_name, value=True, disabled=True, key="all_" + method_name)
            selected_methods.append(method_name)
        else:
            if st.checkbox(method_name, key="box_" + method_name):
                selected_methods.append(method_name)

generate_answers = st.checkbox("Generate answers with Gemini", value=True)

# The API key: a key typed here is used first;
# if the box is empty, the key saved in GEMINI_API_KEY is used.
saved_key = os.environ.get("GEMINI_API_KEY", "")
typed_key = st.text_input("Gemini API key", type="password")
if saved_key:
    st.caption("Leave empty to use the key saved in GEMINI_API_KEY. A key typed here replaces it.")
else:
    st.caption("Needed only for generating answers. The key is not saved.")

if typed_key.strip() != "":
    api_key = typed_key.strip()
else:
    api_key = saved_key

submit = st.button("Submit question", type="primary")

# =====================================================================
# Run retrieval (and generation) when the button is pressed
# =====================================================================
if submit:
    if question.strip() == "":
        st.warning("Please write a question first.")
    elif len(selected_methods) == 0:
        st.warning("Please choose at least one retrieval method.")
    else:
        with st.spinner("Loading the models (only the first time, this can take a few minutes)..."):
            load_models_once()

        # 1) Retrieval for each selected method
        results_by_method = {}
        for method_name in selected_methods:
            with st.spinner(f"Retrieving with {method_name}..."):
                results_by_method[method_name] = retrieval.run_method(method_name, question.strip())

        # 2) Answer generation for each method (same model and prompt for all)
        answers_by_method = {}
        if generate_answers and api_key:
            import generation

            generation.create_client(api_key)
            for method_name in selected_methods:
                with st.spinner(f"Generating the answer for {method_name}..."):
                    try:
                        answer, used_ids = generation.generate_rag_answer(
                            question.strip(), results_by_method[method_name]
                        )
                    except Exception as error:
                        answer = "ERROR: " + str(error)
                    answers_by_method[method_name] = answer
                    time.sleep(generation.SLEEP_BETWEEN_CALLS)  # same 1-second pause as the notebook
        elif generate_answers:
            st.info("Add a Gemini API key to generate answers. Retrieval results are shown below.")

        # Keep the results so they stay on screen when the page refreshes
        st.session_state["question"] = question.strip()
        st.session_state["results"] = results_by_method
        st.session_state["answers"] = answers_by_method

# =====================================================================
# Section 2 - Compare results
# =====================================================================
if "results" in st.session_state:
    st.header("Compare results")
    st.write("Question:")
    show_arabic(st.session_state["question"])

    results_by_method = st.session_state["results"]
    answers_by_method = st.session_state["answers"]

    result_columns = st.columns(len(results_by_method))
    for column, method_name in zip(result_columns, results_by_method):
        with column:
            with st.container(border=True):
                st.subheader(method_name)

                # Generated answer
                st.markdown("**Generated answer**")
                if method_name in answers_by_method:
                    answer = answers_by_method[method_name]
                    if answer.startswith("ERROR:"):
                        st.error(answer)  # for example, a wrong or expired API key
                    else:
                        show_arabic(answer)
                else:
                    st.caption("No answer generated.")

                # Retrieved chunks (top-10), each one in an expandable area
                st.markdown(f"**Retrieved chunks** (score = {SCORE_NAMES[method_name]})")
                for result in results_by_method[method_name]:
                    # For reranking, the rank and score come from the reranker
                    if method_name == "Hybrid + Reranking":
                        rank = result["rerank_rank"]
                        score = result["reranker_score"]
                    else:
                        rank = result["rank"]
                        score = result["score"]

                    label = f"#{rank}  {result['chunk_id']}  |  {result['document_id']}  |  {score:.4f}"
                    with st.expander(label):
                        st.caption("Section: " + result["section"])
                        show_arabic(result["text"])

    st.caption("All ten chunks shown for each method are the context given to Gemini (TOP_K_CHUNKS = 10).")
