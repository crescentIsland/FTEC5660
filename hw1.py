#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
from collections import Counter
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """

    model = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a precise Hong Kong supermarket receipt parser. Read the receipt image carefully "
         "and extract these fields into ONE JSON object:\n\n"
         '- "subtotal_before_discount": goods total before any discounts (null if not shown)\n'
         '- "discount_lines": array of objects, each {{"label": "...", "amount": <positive number>}}; '
         "list EVERY discount/promotion/coupon/offer line separately, do not omit or merge any\n"
         '- "discount_total": sum of all discount_lines amounts, as a positive number\n'
         '- "subtotal_after_discount": the SUBTOTAL line after discounts but BEFORE rounding\n'
         '- "rounding": the ROUNDING amount (e.g. -0.01), use 0 if absent. ROUNDING IS NOT A DISCOUNT; '
         "never add it to discount_total or discount_lines.\n"
         '- "amount_paid": the final amount the customer actually paid AFTER rounding '
         "(Octopus/cash/card)\n\n"
         "Rules:\n"
         "1. Read every line twice before answering.\n"
         "2. discount_total MUST equal the exact sum of discount_lines; recompute it yourself.\n"
         "3. Cross-check: subtotal_after_discount + rounding should equal amount_paid (within 0.01). "
         "If not, re-read the receipt and fix the numbers.\n"
         "4. If unsure about any amount, re-read that region of the image before finalizing.\n\n"
         "5. Item lines often contain a promotional hint like 'Buy 2 Save $5'; the number in that "
         "hint may DIFFER from the actual discount printed in the amount column (e.g. -$6.00). "
         "ALWAYS take the discount amount from the right-hand amount column, never from the hint "
         "text. Read every discount amount from its printed value.\n\n"
         "Output ONLY a valid JSON object with exactly these keys: "
         '"subtotal_before_discount", "discount_lines", "discount_total", '
         '"subtotal_after_discount", "rounding", "amount_paid". '
         "No markdown, no explanation, no extra text."
        ),
        ("human", [
            {"type": "image_url", "image_url": {"url": "{image_url}"}}
        ]),
    ])

    return prompt | model

def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """

    K = 3  # 每张收据采样 3 次，取多数答案，提高鲁棒性

    inputs = [{"image_url": image_data_url(img)} for img in images for _ in range(K)]
    results = chain.batch(inputs)
    per_receipt = [results[i * K:(i + 1) * K] for i in range(len(images))]

    total_paid = Decimal("0")
    total_without_discount = Decimal("0")

    def extract_json(text: str) -> dict:
        text = text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        start = text.find("{")
        end = text.rfind("}")
        return json.loads(text[start:end + 1])

    for group in per_receipt:
        paid_votes, subtotal_votes, discount_votes = [], [], []
        for result in group:
            text = result.content if hasattr(result, "content") else str(result)
            try:
                data = extract_json(text)
                lines = data.get("discount_lines") or []
                discount = (
                    sum(float(ln["amount"]) for ln in lines)
                    if lines
                    else float(data["discount_total"])
                )
                paid_votes.append(str(data["amount_paid"]))
                subtotal_votes.append(str(data["subtotal_after_discount"]))
                discount_votes.append(f"{discount:.2f}")
            except Exception:
                continue

        if not paid_votes:
            print(f"WARNING: no parseable output for a receipt: {text!r}")
            continue

        # 三个字段各自取出现最多的值（众数）
        paid = Counter(paid_votes).most_common(1)[0][0]
        subtotal = Counter(subtotal_votes).most_common(1)[0][0]
        discount = Counter(discount_votes).most_common(1)[0][0]

        total_paid += Decimal(paid)
        total_without_discount += Decimal(subtotal) + Decimal(discount)

    return {
        QUERY_1: f"HK${total_paid:.2f}",
        QUERY_2: f"HK${total_without_discount:.2f}",
    }

    _ = (chain, images)
    return {QUERY_1: DUMMY_RESPONSE, QUERY_2: DUMMY_RESPONSE}


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
