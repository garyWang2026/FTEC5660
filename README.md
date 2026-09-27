# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
  response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The starter intentionally returns the dummy
response `please design your chain to answer these two queries.` so it runs
before you add any API code.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.


## Homework 1 solution

**Approach.** Following the prompt-chaining idea from Tutorial 1, the vision model is only asked to *transcribe* each receipt, never to sum anything. For every receipt image, `deepseek-v4-flash-vision-exp` returns one small JSON object: the gross item total, the list of discount amounts before the subtotal, the SUBTOTAL line, the ROUNDING line, and the payment line immediately after ROUNDING (which is the amount actually paid). The prompt explicitly tells the model what NOT to touch — ROUNDING, change, card balance / amount deducted, points, and duplicate payment copies. A consistency gate then checks that `item_total − sum(discounts) ≈ subtotal`; on mismatch the receipt is extracted a second time with the discrepancy fed back. All aggregation is done in Python, exactly as in the expense-ledger tutorial: query 1 sums `total_paid` over every receipt, query 2 sums `subtotal + sum(discounts)` (rounding is not added back). Each final response is formatted as `HK$<amount>` so the response string contains exactly one number. On the public 7-receipt set the chain scores 2/2 across repeated runs (`HK$1974.30` and `HK$2348.20`).

```mermaid
flowchart TD
    A["Receipt images in a folder"] --> B["Per-receipt extraction<br/>deepseek-v4-flash-vision-exp"]
    B --> C["JSON per receipt:<br/>item_total · discounts[] · subtotal · rounding · total_paid"]
    C --> G{"Consistency gate<br/>item_total − Σdiscounts ≈ subtotal ?"}
    G -- "mismatch" --> R["Retry extraction<br/>with the discrepancy as feedback"]
    G -- "pass" --> P["Python aggregation"]
    R --> P
    P --> Q1["Q1 = Σ total_paid"]
    P --> Q2["Q2 = Σ (subtotal + Σ discounts)<br/>(ROUNDING not added back)"]
    Q1 --> OUT['HK$1974.30']
    Q2 --> OUT2['HK$2348.20']
```

