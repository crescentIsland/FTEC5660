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


## Homework 1 solution: 
> to students: please fill your solution description here.
### Chain Design

```mermaid
flowchart LR
    A["Receipt images<br/>(folder/*.jpg)"] --> B["base64 encode<br/>image_data_url()"]
    B --> C["Sample K=3 times per receipt<br/>(chain.batch parallel)"]
    C --> D["ChatPromptTemplate +<br/>ChatDeepSeek vision model<br/>deepseek-v4-flash-vision-exp"]
    D --> E["Parse JSON per call:<br/>amount_paid, subtotal_after_discount,<br/>discount_lines[], discount_total,<br/>subtotal_before_discount, rounding"]
    E --> F["Per receipt majority vote<br/>(3 calls -> most common value<br/>per field)"]
    F --> G["Decimal aggregation:<br/>total_paid = sum(amount_paid)<br/>total_without = sum(subtotal + discount)"]
    G --> H["Output<br/>Query 1: HK$1974.30<br/>Query 2: HK$2348.20"]
```

### Solution Description

My chain processes every receipt independently and in parallel. For each receipt image, a multimodal prompt asks the DeepSeek vision model (`deepseek-v4-flash-vision-exp`, temperature 0) to extract structured fields as JSON: the final amount paid after rounding, the SUBTOTAL line after discounts but before rounding, every discount/promotion/coupon line listed individually (as `discount_lines`), the goods subtotal before discounts, and the rounding amount. The prompt instructs the model to read discount amounts from the printed amount column rather than from promotional hint text such as "Buy 2 Save $5", whose number can differ from the actual discount, and to cross-check that `subtotal_after_discount + rounding ≈ amount_paid`. To reduce occasional vision misreads, each receipt is sampled three times and a majority vote is taken per field; the discount total is recomputed in Python by summing the listed discount lines instead of trusting the model's own sum. All arithmetic is done with `Decimal` to avoid floating-point drift: `without_discount = subtotal_after_discount + discount_total`, then summed across all receipts. The chain is built as `ChatPromptTemplate | ChatDeepSeek` and run with `chain.batch()` for parallelism. The final response contains exactly one HKD amount per query, matching the grader's parser.

