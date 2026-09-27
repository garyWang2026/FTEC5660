#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


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


EXTRACTION_PROMPT = """You are an expert at reading Hong Kong supermarket receipts (labels may be English or Traditional Chinese).

Look at the receipt image and extract exactly these numbers as ONE JSON object:

{{
  "item_total": <number>,                    // sum of ALL positive item prices (right-hand column) printed BEFORE the subtotal
  "discounts": [<positive number>, ...],    // one entry per discount line printed BEFORE the subtotal
  "subtotal": <number>,                      // amount on the "SUBTOTAL" / "小計" line (after discounts, before rounding)
  "rounding": <number>,                      // amount on the "ROUNDING" line (small, usually negative)
  "total_paid": <number>                     // amount actually PAID: the payment line printed IMMEDIATELY AFTER the ROUNDING line (e.g. OCTOPUS / VISA / CASH / EPS), as a positive number
}}

Rules:
- A "discount line" is a negative amount (-$X.XX) before the subtotal next to words like 包裝變形/包裝損壞, "Buy N Save", "% OFF", "MB PRICE", coupon, member or app promotion. Record each discount as its POSITIVE value.
- The ROUNDING line is NOT a discount. Do not put it in "discounts".
- EXCLUDE everything else: the CHANGE/找續 ($0.00) line; the card section (Amount Deducted/扣除金額, Remaining Value/餘額, card no.); the points section (Point Balance/Points Earned/積分); a duplicate payment printed later (e.g. GP.VISA); dates, times, card numbers, phone numbers, register/ticket numbers. Positive item prices and plastic-bag charges are NOT discounts.
- Plain decimals in HKD, no $ signs or commas.
- Output the JSON object ONLY, no prose.
"""

RETRY_PROMPT = """You are an expert at reading Hong Kong supermarket receipts.

Extract ONE JSON object from this receipt with keys:
  "item_total" (sum of all positive item prices before subtotal),
  "discounts" (POSITIVE amount of every discount/promotion line before the subtotal),
  "subtotal" (SUBTOTAL/小計 line),
  "rounding" (ROUNDING line),
  "total_paid" (the payment amount immediately AFTER the ROUNDING line, positive).

Your previous attempt FAILED a consistency check: {feedback}
Remember: item_total - sum(discounts) must equal subtotal. ROUNDING is not a discount.
Exclude card balance/deducted, points, change, duplicate payment copies, dates and numbers.
Plain decimals, HKD, no $ or commas. Output the JSON object ONLY.
"""


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Uses deepseek-v4-flash-vision-exp. The returned object holds two vision
    runnables: "extract" (first pass) and "retry" (consistency-check retry).
    """
    import os

    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek

    llm = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0.0,
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
    )

    def _vision_chain(template: str):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "user",
                    [
                        {"type": "text", "text": template},
                        {"type": "image_url", "image_url": {"url": "{image_url}"}},
                    ],
                )
            ]
        )
        return prompt | llm | JsonOutputParser()

    return {"extract": _vision_chain(EXTRACTION_PROMPT), "retry": _vision_chain(RETRY_PROMPT)}


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run the chain on every receipt, sum the results in Python.

    Q1 = sum of total_paid (after rounding) over all receipts.
    Q2 = sum of (subtotal + sum of discounts) over all receipts (rounding NOT added back).
    """
    extract = chain["extract"]
    retry = chain["retry"]

    def _clean_number(value: Any, default: float = 0.0) -> float:
        try:
            return float(str(value).replace("$", "").replace(",", "").strip())
        except (TypeError, ValueError):
            return default

    def _read_one(url: str) -> dict:
        try:
            record = extract.invoke({"image_url": url})
        except Exception:
            # First pass returned unparseable output: one retry.
            record = retry.invoke({"image_url": url, "feedback": "the reply was not valid JSON"})

        subtotal = _clean_number(record.get("subtotal"))
        discounts = [_clean_number(d) for d in (record.get("discounts") or [])]
        item_total = _clean_number(record.get("item_total"), default=None)

        # Consistency gate: item_total - sum(discounts) should match subtotal.
        feedback = None
        if item_total is not None:
            expected = item_total - sum(discounts)
            if abs(expected - subtotal) > 0.5:
                feedback = (
                    f"item_total={item_total}, sum(discounts)={sum(discounts):.2f}, "
                    f"but subtotal={subtotal}; they do not reconcile."
                )
        if feedback is not None:
            fixed = retry.invoke({"image_url": url, "feedback": feedback})
            subtotal = _clean_number(fixed.get("subtotal"), subtotal)
            discounts = [_clean_number(d) for d in (fixed.get("discounts") or [])]

        return {
            "total_paid": _clean_number(record.get("total_paid")),
            "subtotal": subtotal,
            "discount_total": sum(discounts),
        }

    urls = [image_data_url(path) for path in images]
    records = [_read_one(url) for url in urls]

    q1 = sum(r["total_paid"] for r in records)
    q2 = sum(r["subtotal"] + r["discount_total"] for r in records)
    return {QUERY_1: f"HK${q1:.2f}", QUERY_2: f"HK${q2:.2f}"}


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
