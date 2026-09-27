"""Robustness test for HW1: run the chain on many receipt combinations."""
import json
import random
import shutil
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path.cwd()
SRC = ROOT / "public_test"
TRUTH = json.loads((SRC / "ground_truth.json").read_text(encoding="utf-8"))
RECEIPTS = sorted(SRC.glob("*.jpg"))
PER_RECEIPT = TRUTH["receipts"]
Q1 = "How much money did I spend in total for these bills?"
Q2 = "How much would I have had to pay without the discount?"

OUT = ROOT / "robust_test"
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir()
random.seed(42)

# 测试组合：7 种单张 + 全量 + 8 个随机组合
folds = []
for img in RECEIPTS:
    folds.append([img])
folds.append(RECEIPTS)
for _ in range(8):
    k = random.randint(2, min(5, len(RECEIPTS)))
    folds.append(random.sample(RECEIPTS, k))

def make_gt(folder, combo):
    total1, total2 = Decimal("0"), Decimal("0")
    for img in combo:
        rec = PER_RECEIPT[img.name]
        total1 += Decimal(str(rec["amount_paid_after_rounding"]))
        total2 += Decimal(str(rec["amount_without_discounts"]))
    gt = {
        "currency": "HKD",
        "answers": {Q1: str(total1), Q2: str(total2)},
        "receipts": {img.name: PER_RECEIPT[img.name] for img in combo},
    }
    (folder / "ground_truth.json").write_text(
        json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")

def run_fold(folder, label):
    folder.mkdir()
    combo = [RECEIPTS] if label == "repeat" else None
    for img in (combo or []):
        shutil.copy2(img, folder / img.name)
    print(f"=== {label}: {len(combo or [])} receipt(s) ===")
    result = subprocess.run(
        [sys.executable, str(ROOT / "hw1.py"), "--image-folder", str(folder)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        print("ERROR:", result.stderr[-2000:])
        return
    for line in (ROOT / "results.csv").read_text(encoding="utf-8").strip().splitlines()[1:]:
        print("   ", line)
    shutil.copy2(ROOT / "results.csv", folder / "results.csv")

for i, combo in enumerate(folds):
    folder = OUT / f"fold{i}"
    folder.mkdir()
    for img in combo:
        shutil.copy2(img, folder / img.name)
    make_gt(folder, combo)
    print(f"=== fold{i}: {len(combo)} receipt(s) -> {[p.name for p in combo]}")
    result = subprocess.run(
        [sys.executable, str(ROOT / "hw1.py"), "--image-folder", str(folder)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        print("ERROR:", result.stderr[-2000:])
        continue
    for line in (ROOT / "results.csv").read_text(encoding="utf-8").strip().splitlines()[1:]:
        print("   ", line)
    shutil.copy2(ROOT / "results.csv", folder / "results.csv")

# 稳定性：全量重复跑 3 次
for rep in range(3):
    folder = OUT / f"repeat{rep}"
    folder.mkdir()
    for img in RECEIPTS:
        shutil.copy2(img, folder / img.name)
    make_gt(folder, RECEIPTS)
    print(f"=== repeat{rep}: full folder again ===")
    result = subprocess.run(
        [sys.executable, str(ROOT / "hw1.py"), "--image-folder", str(folder)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        print("ERROR:", result.stderr[-2000:])
        continue
    for line in (ROOT / "results.csv").read_text(encoding="utf-8").strip().splitlines()[1:]:
        print("   ", line)
    shutil.copy2(ROOT / "results.csv", folder / "results.csv")

print("\nAll folds done. Results kept in robust_test/fold*/results.csv")
