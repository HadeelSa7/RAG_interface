"""
test_app.py - simple checks for the prototype.  Run with:   pytest -q

1. The functions in retrieval.py and generation.py are identical to the notebook.
2. The tokenizer gives the same tokens the notebook printed.
3. Each of the four methods runs and returns 10 ranked results
   (small fake models are used here, so the test is fast and needs no download).
4. Gemini receives the original prompt, model and settings.
"""
import ast
import json
import os
import sys
import types

import numpy as np

# ---------------------------------------------------------------------
# Fake "google.genai" package, so generation.py can be imported without
# internet or an API key. It is only used by this test file.
# ---------------------------------------------------------------------
fake_google = types.ModuleType("google")
fake_genai = types.ModuleType("google.genai")
fake_errors = types.ModuleType("google.genai.errors")
fake_errors.APIError = type("APIError", (Exception,), {})
fake_errors.ClientError = type("ClientError", (fake_errors.APIError,), {})
fake_errors.ServerError = type("ServerError", (fake_errors.APIError,), {})
fake_genai.errors = fake_errors
fake_genai.Client = lambda api_key: None
fake_google.genai = fake_genai
sys.modules.setdefault("google", fake_google)
sys.modules["google.genai"] = fake_genai
sys.modules["google.genai.errors"] = fake_errors

import generation  # noqa: E402
import retrieval  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def read_notebook_code(notebook_path):
    """Return all code cells of a notebook as one list of Python statements."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
    statements = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # remove "!pip install ..." lines (they are not Python)
        lines = [line for line in source.splitlines() if not line.strip().startswith("!")]
        source = "\n".join(lines)
        try:
            for node in ast.parse(source).body:
                statements.append(ast.get_source_segment(source, node))
        except SyntaxError:
            pass
    return statements

def read_notebook_code_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    return [ast.get_source_segment(source, node) for node in ast.parse(source).body]


def find_definition(statements, name):
    for statement in statements:
        if statement.startswith("def " + name + "(") or statement.startswith(name + " ="):
            return statement
    raise AssertionError(name + " not found")


def test_retrieval_functions_are_unchanged():
    notebook_code = read_notebook_code(os.path.join(HERE, "data", "Retrival_Methods_updated1.ipynb"))
    our_code = read_notebook_code_from_file(os.path.join(HERE, "retrieval.py"))
    for name in ["normalize_arabic_keep_lines", "remove_prefix", "tokenize_arabic",
                 "bm25_search", "dense_search", "hybrid_search", "rerank_results",
                 "ARABIC_STOPWORDS", "ARABIC_PREFIXES", "DIGIT_TABLE", "LIST_NUMBER_PATTERN"]:
        ours = find_definition(our_code, name)
        assert ours in notebook_code, name + " is different from the notebook"


def test_generation_code_is_unchanged():
    notebook_code = read_notebook_code(os.path.join(HERE, "data", "Generated_Answer3.ipynb"))
    our_code = read_notebook_code_from_file(os.path.join(HERE, "generation.py"))
    for name in ["TOP_K_CHUNKS", "GEMINI_MODEL", "GENERATION_CONFIG", "MAX_RETRIES",
                 "RETRY_BACKOFF_SECONDS", "SLEEP_BETWEEN_CALLS", "PROMPT_TEMPLATE", "generate_rag_answer"]:
        ours = find_definition(our_code, name)
        assert ours in notebook_code, name + " is different from the notebook"


def test_tokenizer_gives_notebook_tokens():
    with open(retrieval.CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    # Notebook cell 29 printed these first tokens of the first chunk:
    printed = ['لغة', 'دراسة', 'برامج', 'تدرس', 'لغة', 'عربية', 'محتوي', 'لغة', 'انجليزية', 'بكالوريوس']
    assert retrieval.tokenize_arabic(chunks[0]["text"])[:10] == printed
    # ... and "Total tokens: 5705"
    assert sum(len(retrieval.tokenize_arabic(c["text"])) for c in chunks) == 5705


# ---------------------------------------------------------------------
# Small fake models (only for testing the four methods quickly)
# ---------------------------------------------------------------------
class FakeBM25:
    def __init__(self, corpus):
        self.corpus = corpus

    def get_scores(self, query_tokens):
        return np.array([sum(doc.count(t) for t in query_tokens) + i * 1e-6
                         for i, doc in enumerate(self.corpus)])


class FakeEmbeddingModel:
    def encode(self, texts, batch_size=8, show_progress_bar=False, normalize_embeddings=True):
        vectors = []
        for text in texts:
            rng = np.random.default_rng(len(text) * 7919 + sum(map(ord, text[:50])))
            v = rng.normal(size=16)
            vectors.append(v / np.linalg.norm(v))
        return np.array(vectors)


class FakeLogits:
    def __init__(self, values):
        self.values = np.array(values, dtype=float)

    def view(self, *args):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeTokenizer:
    def __call__(self, pairs, padding=True, truncation=True, return_tensors="pt"):
        self.number_of_pairs = len(pairs)
        return {"pairs": pairs}


class FakeReranker:
    def __call__(self, pairs):
        return types.SimpleNamespace(logits=FakeLogits([len(text) % 13 for q, text in pairs]))


class FakeTorch:
    class no_grad:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False


def load_fake_models():
    with open(retrieval.CHUNKS_FILE, "r", encoding="utf-8") as f:
        retrieval.all_chunks = json.load(f)
    retrieval.corpus = [retrieval.tokenize_arabic(c["text"]) for c in retrieval.all_chunks]
    retrieval.bm25 = FakeBM25(retrieval.corpus)
    retrieval.model = FakeEmbeddingModel()
    retrieval.chunk_embeddings = retrieval.model.encode([c["text"] for c in retrieval.all_chunks])
    retrieval.tokenizer = FakeTokenizer()
    retrieval.reranker_model = FakeReranker()
    retrieval.torch = FakeTorch()


QUESTION = "ما شروط القبول في برامج الدكتوراه؟"


def test_all_four_methods_return_ten_ranked_results():
    load_fake_models()
    for method_name in ["BM25", "BGE-M3", "Hybrid", "Hybrid + Reranking"]:
        results = retrieval.run_method(method_name, QUESTION)
        assert len(results) == 10, method_name
        for result in results:
            assert result["chunk_id"].startswith("DOC_")
            assert result["document_id"].startswith("DOC_")


def test_reranking_uses_20_candidates_and_keeps_10():
    load_fake_models()
    results = retrieval.run_method("Hybrid + Reranking", QUESTION)
    assert retrieval.tokenizer.number_of_pairs == 20
    assert [r["rerank_rank"] for r in results] == list(range(1, 11))


def test_gemini_gets_original_prompt_and_settings():
    calls = []

    class FakeInteractions:
        def create(self, model, input, generation_config):
            calls.append((model, input, generation_config))
            return types.SimpleNamespace(output_text=" الإجابة ")

    generation.client = types.SimpleNamespace(interactions=FakeInteractions())
    results = [{"chunk_id": f"C{i}", "text": f"نص {i}"} for i in range(15)]
    answer, used_ids = generation.generate_rag_answer("سؤال؟", results)

    model, prompt, config = calls[0]
    assert model == "gemini-3.5-flash-lite"
    assert config == {"temperature": 0.0}
    assert "سؤال؟" in prompt and "Chunk ID: C9" in prompt and "Chunk ID: C10" not in prompt
    assert answer == "الإجابة"
    assert used_ids == [f"C{i}" for i in range(10)]


def test_notebook_builds_the_same_app_py():
    """app.ipynb writes app.py with %%writefile cells; the result must equal app.py."""
    with open(os.path.join(HERE, "app.ipynb"), "r", encoding="utf-8") as f:
        notebook = json.load(f)
    written = ""
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        cell_text = "".join(cell["source"])
        if not cell_text.endswith("\n"):
            cell_text += "\n"                      # Jupyter adds this newline when running a cell
        first_line, cell_body = cell_text.split("\n", 1)
        if first_line == "%%writefile app.py":
            written = cell_body                    # first cell creates the file
        elif first_line == "%%writefile -a app.py":
            written += cell_body                   # next cells append to it
    with open(os.path.join(HERE, "app.py"), "r", encoding="utf-8") as f:
        assert written == f.read()
