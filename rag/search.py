"""Поиск по аналитике Альфа-Банка (RAG, шаг Retrieval).

Нужны файлы из предыдущих шагов (лежат рядом со скриптом, в rag/):
    chunks.jsonl      — чанки (chunk.py)
    embeddings.npy    — векторы чанков (embed.py), строка i <-> строка i в chunks.jsonl
    index_meta.json   — {"model": ..., "n_chunks": ..., "dim": ...}

Запуск:
    python rag/search.py "что будет с ключевой ставкой"
    python rag/search.py "цены на бензин" -k 5
    python rag/search.py --test          # прогон проверочных запросов

Из кода:
    from search import Searcher
    s = Searcher()
    hits = s.search("ключевая ставка", k=3)
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

RAG_DIR = Path(__file__).resolve().parent

# Ниже этого скора считаем, что аналитики по теме нет.
# Подбери по --test: у e5 скоры сжаты (условно 0.75–0.9), смотри на мусорный запрос.
MIN_SCORE = 0.80

TEST_QUERIES = [
    ("что будет с ключевой ставкой", "Ключевая ставка"),
    ("цены на бензин растут", "Инфляция"),
    ("спрос населения замедляется", "Потребление"),
    ("доходность ОФЗ", "Бонд Бриф / FI Strategy"),
    ("золото дорожает", "золота"),
    ("рост зарплат и безработица", "7М26 / 1П26"),
    ("манулу Тимофею в зоопарке нашли самку", "НИЧЕГО (мусорный запрос)"),
]


class Searcher:
    def __init__(self, rag_dir: Path = RAG_DIR):
        meta = json.loads((rag_dir / "index_meta.json").read_text(encoding="utf-8"))
        with open(rag_dir / "chunks.jsonl", encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f if line.strip()]
        self.emb = np.load(rag_dir / "embeddings.npy")

        # Защита от рассинхрона: перегенерировал чанки и забыл пересчитать векторы
        n = len(self.chunks)
        if not (n == meta["n_chunks"] == self.emb.shape[0]):
            raise RuntimeError(
                f"Рассинхрон: chunks.jsonl={n}, embeddings.npy={self.emb.shape[0]}, "
                f"index_meta={meta['n_chunks']}. Перезапусти embed.py"
            )

        # Модель грузим ОДИН раз — это самое долгое (секунды)
        self.model = SentenceTransformer(meta["model"])

    def search(self, query: str, k: int = 3, min_score: float = MIN_SCORE):
        """Топ-k чанков по смыслу. Чанки ниже min_score отбрасываются."""
        q = self.model.encode("query: " + query, normalize_embeddings=True)
        scores = self.emb @ q                    # векторы нормированы -> это косинус
        top = np.argsort(-scores)[:k]            # индексы по убыванию скора
        hits = []
        for i in top:
            s = float(scores[i])
            if s < min_score:
                continue
            hits.append({**self.chunks[i], "score": round(s, 4)})
        return hits


def show(searcher, query, k, min_score):
    # min_score=0, чтобы видеть скоры даже у мусора, а порог показываем пометкой
    hits = searcher.search(query, k=k, min_score=0.0)
    print(f"\n=== {query}")
    for h in hits:
        mark = "  " if h["score"] >= min_score else "✗ "   # ✗ = ниже порога, в промпт не пойдёт
        body = h["text"].split("\n\n", 1)[-1]               # без шапки "Документ: ..."
        print(f"{mark}{h['score']:.3f}  [{h['chunk_id']}]  {h['doc_title'][:60]}")
        print(f"         {body[:160]}...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", help="текст запроса")
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--min-score", type=float, default=MIN_SCORE)
    ap.add_argument("--test", action="store_true", help="прогнать проверочные запросы")
    args = ap.parse_args()

    if not args.query and not args.test:
        ap.error("укажи запрос или --test")

    s = Searcher()
    if args.test:
        for q, expected in TEST_QUERIES:
            print(f"\n(ожидаем: {expected})", end="")
            show(s, q, args.k, args.min_score)
        print(f"\nПорог min_score = {args.min_score}. Строки с ✗ в промпт не попадут.")
    else:
        show(s, args.query, args.k, args.min_score)


if __name__ == "__main__":
    main()