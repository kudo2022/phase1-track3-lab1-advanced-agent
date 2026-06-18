from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen


DEFAULT_SOURCE_URL_TEMPLATE = (
    "https://datasets-server.huggingface.co/rows?"
    "dataset=hotpotqa/hotpot_qa&config=distractor&split=validation&offset=0&length={length}"
)


def download_json(url: str, destination: Path) -> list[dict]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if isinstance(payload, dict) and "rows" in payload:
        rows = [item["row"] for item in payload["rows"]]
    else:
        rows = payload
    destination.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def convert_record(item: dict) -> dict:
    level = item.get("level")
    difficulty = level if level in {"easy", "medium", "hard"} else "medium"
    raw_context = item.get("context", [])
    if isinstance(raw_context, dict):
        pairs = zip(raw_context.get("title", []), raw_context.get("sentences", []))
    else:
        pairs = raw_context
    context = [
        {
            "title": title,
            "text": " ".join(sentence.strip() for sentence in sentences if sentence.strip()),
        }
        for title, sentences in pairs
    ]
    return {
        "qid": item.get("_id", item["id"]),
        "difficulty": difficulty,
        "question": item["question"],
        "gold_answer": item["answer"],
        "context": context,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and convert HotpotQA for Step 4.")
    parser.add_argument("--source-url")
    parser.add_argument("--raw-out", default="data/raw/hotpot_dev_distractor_v1.json")
    parser.add_argument("--dataset-out", default="data/hotpot_dev_distractor_100.json")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    raw_path = Path(args.raw_out)
    dataset_path = Path(args.dataset_out)
    source_url = args.source_url or DEFAULT_SOURCE_URL_TEMPLATE.format(length=args.limit)

    if raw_path.exists():
        raw_items = json.loads(raw_path.read_text(encoding="utf-8"))
        if len(raw_items) < args.limit:
            raw_items = download_json(source_url, raw_path)
    else:
        raw_items = download_json(source_url, raw_path)

    converted = [convert_record(item) for item in raw_items[: args.limit]]
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved raw dataset to {raw_path}")
    print(f"Saved converted subset to {dataset_path}")
    print(f"Converted {len(converted)} examples")


if __name__ == "__main__":
    main()
