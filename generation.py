"""
generation.py - answer generation with Gemini.

Everything in PART 1 is copied UNCHANGED from the notebook "Generated_Answer3.ipynb":
the same model, the same prompt, the same settings and the same function.
"""
import time

from google import genai
from google.genai import errors as genai_errors

# Created by create_client() with the user's API key (the notebook used getpass).
client = None


# =====================================================================
# PART 1 - ORIGINAL NOTEBOOK CODE (copied without any change)
# =====================================================================

# ---- From notebook cell 6 (shared settings) ----
TOP_K_CHUNKS = 10
GEMINI_MODEL = "gemini-3.5-flash-lite"
GENERATION_CONFIG = {
    "temperature": 0.0,
}

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5
SLEEP_BETWEEN_CALLS = 1

PROMPT_TEMPLATE = """أنت مساعد أكاديمي يجيب عن أسئلة الطلاب في جامعة ميداوشن.

أجب عن السؤال اعتمادًا فقط على المعلومات الموجودة في السياق المرفق.
لا تضف معلومات من خارج السياق.
إذا لم تكن المعلومات الكافية للإجابة موجودة في السياق، قل:
"لا توجد معلومات كافية في السياق للإجابة عن هذا السؤال."

اكتب إجابة واضحة ومباشرة باللغة العربية.

السؤال:
{question}

السياق:
{context}

الإجابة:
"""


# ---- From notebook cell 12 (unified answer generation function) ----
def generate_rag_answer(question, results, top_k=TOP_K_CHUNKS):
    """دالة موحّدة تُستخدم لكل من Non-reranked و Reranked بلا أي اختلاف."""
    top_results = results[:top_k]
    used_chunk_ids = [r["chunk_id"] for r in top_results]

    context = "\n\n".join(
        f"Chunk ID: {r['chunk_id']}\n{r['text']}"
        for r in top_results
    )

    prompt = PROMPT_TEMPLATE.format(question=question, context=context)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            interaction = client.interactions.create(
                model=GEMINI_MODEL,
                input=prompt,
                generation_config=GENERATION_CONFIG,
            )

            answer = interaction.output_text
            if not answer or not answer.strip():
                raise ValueError("استجابة Gemini فارغة (output_text خالٍ).")

            return answer.strip(), used_chunk_ids

        except genai_errors.ClientError as e:
            last_error = f"ClientError {e.code}: {e.message}"
            if e.code == 429 and attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue
            raise RuntimeError(last_error) from e

        except genai_errors.ServerError as e:
            last_error = f"ServerError {e.code}: {e.message}"
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue
            raise RuntimeError(last_error) from e

        except genai_errors.APIError as e:
            last_error = f"APIError: {e}"
            raise RuntimeError(last_error) from e

        except ValueError as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue
            raise

    raise RuntimeError(last_error or "فشل غير معروف.")


# =====================================================================
# PART 2 - CREATE THE GEMINI CLIENT
# =====================================================================

def create_client(api_key):
    """Same as the notebook: client = genai.Client(api_key=api_key).
    The key is only kept in memory, never written to a file."""
    global client
    client = genai.Client(api_key=api_key)
