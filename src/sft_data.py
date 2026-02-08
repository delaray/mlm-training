
#  ************************************************************************
#  Generate Supervised Fine-Tuning Data From A Pdf Book
#  ************************************************************************

# Usage: python make_sft_from_pdf.py path/to/book.pdf --model llama3.1:8b
#        --pairs-per-paragraph 4 --out book_sft.jsonl

# *************************************************************************

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

import requests
from pypdf import PdfReader
from tqdm import tqdm

load_dotenv(override=True)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")


# ----------------------------------------------------------------------------
# Text extraction & cleaning
# ----------------------------------------------------------------------------

def extract_pdf_text(pdf_path: str, max_pages: Optional[int] = None) -> str:
    reader = PdfReader(pdf_path)
    pages = reader.pages[:max_pages] if max_pages else reader.pages
    parts = []
    for p in pages:
        txt = p.extract_text() or ""
        parts.append(txt)
    return "\n".join(parts)


def normalize_text(text: str) -> str:
    # Remove common header/footer noise patterns (best-effort; customize
    # for your PDFs)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Fix hyphenation across line breaks: "exam-\nple" -> "example"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Join single newlines that look like line-wrapping within paragraphs
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str, min_chars: int = 300, max_chars: int = 1800
                     ) -> List[str]:
    raw_paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    paras = []
    buf = ""

    def flush_buf():
        nonlocal buf
        b = buf.strip()
        if b:
            paras.append(b)
        buf = ""

    for p in raw_paras:
        # If paragraph too short, accumulate
        if len(p) < min_chars:
            buf = (buf + " " + p).strip()
            if len(buf) >= min_chars:
                flush_buf()
            continue

        # If we already have buffered short text, flush it first
        if buf:
            flush_buf()

        # If paragraph too long, split by sentences roughly
        if len(p) > max_chars:
            sentences = re.split(r"(?<=[.!?])\s+", p)
            chunk = ""
            for s in sentences:
                if len(chunk) + len(s) + 1 <= max_chars:
                    chunk = (chunk + " " + s).strip()
                else:
                    if len(chunk) >= min_chars:
                        paras.append(chunk)
                        chunk = s
                    else:
                        # force add to reach min_chars
                        chunk = (chunk + " " + s).strip()
            if chunk:
                paras.append(chunk)
        else:
            paras.append(p)

    if buf:
        flush_buf()

    # Deduplicate exact duplicates
    seen = set()
    uniq = []
    for p in paras:
        key = re.sub(r"\s+", " ", p).strip().lower()
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


# ----------------------------------------------------------------------------
# Ollama helpers
# ----------------------------------------------------------------------------
def ollama_chat(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: Optional[int] = None,
    num_ctx: int = 8192,
    timeout_s: int = 180,
) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_ctx": num_ctx,
        },
    }
    if seed is not None:
        payload["options"]["seed"] = seed

    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload,
                      timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]


def extract_json_from_text(text: str) -> Any:
    """
    Best-effort JSON extraction: supports responses that wrap JSON in
    markdown fences.
    """
    text = text.strip()
    # Try fenced code block
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.S)
    if m:
        return json.loads(m.group(1))

    # Try direct JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to locate first {...} or [...]
    m2 = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
    if m2:
        return json.loads(m2.group(1))

    raise ValueError("Could not parse JSON from model output.")


# ----------------------------------------------------------------------------
# Data generation prompts
# ----------------------------------------------------------------------------

QA_GEN_SYSTEM = (
    """You generate high-quality supervised fine-tuning examples
    grounded in a provided excerpt. """
    """Never invent facts not present in the excerpt.
    Prefer assistant-like helpful answers."""
)


def qa_gen_user(excerpt: str, n_pairs: int) -> str:
    return f"""
Create {n_pairs} diverse question/answer pairs that are ANSWERABLE using
ONLY the excerpt.

Rules:
- The answer must be fully supported by the excerpt (no outside knowledge).
- Mix question types: definition, why/how, comparison, application, edge case,
  step-by-step (when possible).
- Avoid trivial copy/paste questions.
- Keep answers helpful and clear; quote short phrases only when necessary.
- Return STRICT JSON only, no commentary.

JSON schema:
{{
  "pairs": [
    {{
      "question": "...",
      "answer": "...",
      "type": "definition|how|why|comparison|application|edge_case|summary|procedure",
      "difficulty": "easy|medium|hard",
      "evidence": ["short supporting quote 1", "short supporting quote 2"]
    }}
  ]
}}

EXCERPT:
\"\"\"{excerpt}\"\"\"
""".strip()


SUPPORT_CHECK_SYSTEM = (
    """You are a strict verifier. You check whether an answer is fully
    supported by an excerpt."""
)


def support_check_user(excerpt: str, question: str, answer: str) -> str:
    return f"""
Decide if the ANSWER is fully supported by the EXCERPT.

Return STRICT JSON only:
{{
  "verdict": "SUPPORTED" | "NOT_SUPPORTED",
  "reason": "short reason",
  "fix": "If NOT_SUPPORTED, provide a corrected answer that IS supported,
          otherwise empty string."
}}

QUESTION: {question}
ANSWER: {answer}

EXCERPT:
\"\"\"{excerpt}\"\"\"
""".strip()


# ----------------------------------------------------------------------------
# Dataset writing
# ----------------------------------------------------------------------------
def to_chat_example(excerpt: str, question: str, answer: str
                    ) -> Dict[str, Any]:
    user_content = (
        "Using ONLY the excerpt below, answer the question.\n\n"
        f"EXCERPT:\n{excerpt}\n\n"
        f"QUESTION:\n{question}"
    )
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ]
    }


@dataclass
class GenConfig:
    model: str = DEFAULT_MODEL
    pairs_per_paragraph: int = 3
    temperature: float = 0.7
    top_p: float = 0.9
    seed: Optional[int] = None
    verify_support: bool = True
    max_paragraphs: Optional[int] = None
    out_path: str = "dataset.jsonl"


def generate_for_paragraph(paragraph: str, cfg: GenConfig
                           ) -> List[Dict[str, Any]]:
    # 1) Generate pairs
    raw = ollama_chat(
        model=cfg.model,
        messages=[
            {"role": "system",
             "content": QA_GEN_SYSTEM},
            {"role": "user",
             "content": qa_gen_user(paragraph, cfg.pairs_per_paragraph)},
        ],
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        seed=cfg.seed,
    )
    data = extract_json_from_text(raw)
    pairs = data.get("pairs", [])
    results = []

    # 2) Optional verification pass (reduces hallucinations a lot)
    for p in pairs:
        q = (p.get("question") or "").strip()
        a = (p.get("answer") or "").strip()
        if not q or not a:
            continue

        if cfg.verify_support:
            check_raw = ollama_chat(
                model=cfg.model,
                messages=[
                    {"role": "system",
                     "content": SUPPORT_CHECK_SYSTEM},
                    {"role": "user",
                     "content": support_check_user(paragraph, q, a)},
                ],
                temperature=0.0,  # deterministic checking
                top_p=1.0,
                seed=cfg.seed,
            )
            check = extract_json_from_text(check_raw)
            verdict = check.get("verdict", "NOT_SUPPORTED")
            if verdict == "NOT_SUPPORTED":
                fixed = (check.get("fix") or "").strip()
                if not fixed:
                    continue
                a = fixed

        results.append(to_chat_example(paragraph, q, a))

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="Path to PDF book")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--pairs-per-paragraph", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-verify", action="store_true",
                    help="Disable support verification pass")
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--max-paragraphs", type=int, default=None)
    ap.add_argument("--out", default="dataset.jsonl")
    args = ap.parse_args()

    cfg = GenConfig(
        model=args.model,
        pairs_per_paragraph=args.pairs_per_paragraph,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        verify_support=not args.no_verify,
        max_paragraphs=args.max_paragraphs,
        out_path=args.out,
    )

    raw_text = extract_pdf_text(args.pdf, max_pages=args.max_pages)
    text = normalize_text(raw_text)
    paragraphs = split_paragraphs(text)

    if cfg.max_paragraphs:
        paragraphs = paragraphs[: cfg.max_paragraphs]

    # Shuffle to increase topical diversity early in the file (optional)
    random.shuffle(paragraphs)

    n_written = 0
    with open(cfg.out_path, "w", encoding="utf-8") as f:
        for para in tqdm(paragraphs, desc="Generating"):
            # Basic guard: skip very low-signal paragraphs
            if len(para) < 200:
                continue

            # Retry wrapper (Ollama occasionally returns malformed JSON)
            for attempt in range(4):
                try:
                    examples = generate_for_paragraph(para, cfg)
                    for ex in examples:
                        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                        n_written += 1
                    break
                except Exception as e:
                    e = e.message if hasattr(e, "message") else str(e)
                    print(f"⚠️  Warning: generation error (attempt "
                          f"{attempt + 1}/4): {e}")
                    if attempt == 3:
                        # give up on this paragraph
                        # you could log e somewhere if desired
                        pass
                    time.sleep(1.5 * (attempt + 1))

    print(f"Done. Wrote {n_written} chat examples to {cfg.out_path}")


if __name__ == "__main__":
    main()
