# Arabic RAG Research Prototype

A simple Streamlit prototype for the master's thesis. You ask an Arabic question, run the four
retrieval methods (BM25, BGE-M3, Hybrid, Hybrid + Reranking), and compare the retrieved chunks
and the Gemini answers side by side.

## Files

```
simple_rag_app/
├── app.py            the Streamlit page (Section 1: ask, Section 2: compare)
├── app.ipynb         the same app.py code as a Jupyter notebook, explained part by part
├── retrieval.py      the notebook retrieval functions (copied unchanged) + model loading
├── generation.py     the notebook Gemini settings, prompt and function (copied unchanged)
├── test_app.py       simple tests
├── requirements.txt
├── .streamlit/config.toml
└── data/             copies of midocean_chunks_final.json, evaluation_queries_final.json and the two notebooks
```

## Run it (Anaconda Prompt)

```bat
cd /d "C:\Users\hp\Desktop\shd\midocean\RAG\اكواد جديدة\simple_rag_app"
pip install streamlit
streamlit run app.py
```

The browser opens at http://localhost:8501. The first **Submit question** downloads and loads
the models (about 4.5 GB, a few minutes). After that, each question takes seconds.

**Questions:** type your own question, or pick one of the 50 evaluation questions from the list
(it is copied into the question box and can still be edited).

**Gemini API key:** a key typed in the *Gemini API key* box is always used first. If the box is empty,
the key saved in the `GEMINI_API_KEY` environment variable is used. If a key is wrong or expired,
each result column shows the error in red.

To skip typing the Gemini key every time, set it once before starting:

```bat
set GEMINI_API_KEY=your-key-here
streamlit run app.py
```

## Run it from Jupyter (app.ipynb)

1. Open `app.ipynb` in Jupyter (it must stay in this folder).
2. Menu **Kernel → Restart & Run All**.
   Each `%%writefile` cell writes its part of the code into `app.py`; the last cell starts the app.
3. The app opens at http://localhost:8501. Stop it with the **■ (Interrupt)** button.

Always use *Run All*: running one `%%writefile -a` cell twice adds its code to `app.py` twice.

## Tests

```bat
pytest -q
```

The tests check that the copied functions are identical to the notebooks, that the tokenizer gives
the notebook's printed tokens, that all four methods return 10 ranked results, that reranking uses
20 candidates and keeps 10, that Gemini gets the original prompt and settings, and that `app.ipynb` builds exactly the same `app.py`.
