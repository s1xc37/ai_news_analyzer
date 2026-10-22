import argparse
import datetime as dt
import json
import re
import statistics
from pathlib import Path

import frontmatter

ROOT = Path(__file__).resolve().parent.parent  # корень репо (скрипт лежит в rag/)
DEFAULT_DOCS = ROOT / "docs" / "md"
DEFAULT_OUT = ROOT / "rag" / "chunks.jsonl"

E5_MODEL = "intfloat/multilingual-e5-base"
E5_MAX_TOKENS = 512

# Граница предложения: знак конца + пробел + начало нового предложения.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[«\"(\[A-ZА-ЯЁ0-9])")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)


# ---------- загрузка ----------

def to_iso(value):
    """YAML превращает 2026-09-11 в datetime.date, а json.dumps его не умеет."""
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return str(value) if value is not None else None


def load_docs(docs_dir: Path):
    docs = []
    for path in sorted(docs_dir.glob("*.md")):
        post = frontmatter.load(path)
        meta = post.metadata
        if meta.get("outdated"):
            print(f"SKIP  {path.name} (outdated)")
            continue
        docs.append(
            {
                "stem": path.stem,
                "title": meta.get("title") or path.stem,
                "date": to_iso(meta.get("date")),
                "author": meta.get("author"),
                "source_file": meta.get("source_file"),
                "tags": list(meta.get("tags") or []),
                "content": post.content,
            }
        )
    return docs


# ---------- разбор markdown на блоки ----------

def clean(text: str) -> str:
    text = text.replace("**", "").replace("__", "").replace("\\*", "")  # жирный шрифт эмбеддингам не нужен
    return re.sub(r"\s+", " ", text).strip()


def iter_blocks(content: str):
    """Отдаёт (kind, text): kind = heading | figure | para."""
    content = HTML_COMMENT.sub("", content)
    for raw in re.split(r"\n\s*\n", content):
        block = raw.strip()
        if not block:
            continue
        if block.startswith("#"):
            # заголовок может оказаться в одном блоке с текстом без пустой строки
            first, _, rest = block.partition("\n")
            yield "heading", (len(first) - len(first.lstrip("#")), clean(first.lstrip("#")))
            if rest.strip():
                yield "para", clean(rest)
        elif block.startswith("[Рис"):
            yield "figure", clean(block)
        else:
            yield "para", clean(block)


def split_long(text: str, max_words: int):
    """Длинный абзац -> примерно равные куски по границам предложений.

    Делим не жадно (иначе в конце остаётся огрызок из одного предложения),
    а на n равных частей, где n = минимум, при котором часть <= max_words.
    """
    total = len(text.split())
    if total <= max_words:
        return [text]
    n_parts = -(-total // max_words)  # ceil
    target = total / n_parts
    parts, cur, cur_words = [], [], 0
    for sent in SENTENCE_SPLIT.split(text):
        cur.append(sent)
        cur_words += len(sent.split())
        if cur_words >= target and len(parts) < n_parts - 1:
            parts.append(" ".join(cur))
            cur, cur_words = [], 0
    if cur:
        parts.append(" ".join(cur))
    return parts


# ---------- нарезка документа ----------

def chunk_doc(doc, min_words: int, max_words: int):
    pieces = []          # [(section, text)]
    section = None
    buf = []             # абзацы, которые копятся до min_words

    def flush():
        nonlocal buf
        if not buf:
            return
        text = " ".join(buf)
        buf = []
        # хвост короче min_words приклеиваем к предыдущему куску той же секции
        if len(text.split()) < min_words and pieces and pieces[-1][0] == section:
            prev_section, prev_text = pieces.pop()
            text = prev_text + " " + text
        for part in split_long(text, max_words):
            pieces.append((section, part))

    for kind, value in iter_blocks(doc["content"]):
        if kind == "heading":
            level, title = value
            flush()
            if level >= 2:          # "# ..." — это заголовок документа, он уже в title
                section = title
        elif kind == "figure":
            if buf:
                buf.append(value)
            elif pieces and pieces[-1][0] == section:
                s, t = pieces.pop()
                pieces.append((s, t + " " + value))
            # подпись без текста рядом — выкидываем
        else:
            buf.append(value)
            if len(" ".join(buf).split()) >= min_words:
                flush()
    flush()

    chunks = []
    for i, (sec, body) in enumerate(pieces, 1):
        header = f"Документ: {doc['title']} ({doc['date']})."
        if sec:
            header += f" Раздел: {sec}."
        chunks.append(
            {
                "chunk_id": f"{doc['stem']}_{i:02d}",
                "doc_title": doc["title"],
                "date": doc["date"],
                "author": doc["author"],
                "source_file": doc["source_file"],
                "section": sec,
                "tags": doc["tags"],
                "text": f"{header}\n\n{body}",
            }
        )
    return chunks


# ---------- проверка длины в токенах e5 ----------

def get_tokenizer(enabled: bool):
    if not enabled:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(E5_MODEL)
    except Exception as exc:  # нет transformers или нет сети — считаем по словам
        print(f"WARN  токенизатор e5 недоступен ({exc.__class__.__name__}), считаю по словам")
        return None


def count_tokens(tokenizer, text: str) -> int:
    if tokenizer is None:
        return int(len(text.split()) * 1.8)  # грубая оценка для русского
    return len(tokenizer("passage: " + text)["input_ids"])


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--min-words", type=int, default=40)
    ap.add_argument("--max-words", type=int, default=220)
    ap.add_argument("--no-tokenizer", action="store_true", help="не грузить токенизатор e5")
    args = ap.parse_args()

    docs = load_docs(args.docs)
    if not docs:
        raise SystemExit(f"Нет документов в {args.docs}")

    tokenizer = get_tokenizer(not args.no_tokenizer)
    all_chunks = []
    for doc in docs:
        chunks = chunk_doc(doc, args.min_words, args.max_words)
        for c in chunks:
            c["n_tokens"] = count_tokens(tokenizer, c["text"])
            if c["n_tokens"] > E5_MAX_TOKENS:
                print(f"WARN  {c['chunk_id']}: {c['n_tokens']} токенов > {E5_MAX_TOKENS}, e5 обрежет хвост")
        sizes = [c["n_tokens"] for c in chunks]
        print(f"OK    {doc['stem']}: {len(chunks)} чанков, токены min/avg/max = "
              f"{min(sizes)}/{int(statistics.mean(sizes))}/{max(sizes)}")
        all_chunks.extend(chunks)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nВсего {len(all_chunks)} чанков -> {args.out}")


if __name__ == "__main__":
    main()