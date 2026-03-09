"""
chart_anomalies.py
Investigates the specific anomalies visible in the analyze.py charts:
  1. Inuit/NU inflated scores (small denominator)
  2. First_Nations/Metis exactly 0.00 (missing data or key mismatch)
  3. Hallucination uniformity
  4. Overall visibility distribution

Run: python chart_anomalies.py
"""
import sqlite3, sys
from pathlib import Path
from collections import Counter

try:
    from config import DB_PATH
except ImportError:
    print("ERROR: run from project root.")
    sys.exit(1)

SEP = "─" * 65
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

# ── 1. Program counts per population value ────────────────────────────────────
print(SEP)
print("1. PROGRAM COUNT PER POPULATION VALUE IN classifications TABLE")
print("   (First_Nations/Metis = 0 means either missing or key mismatch)")
rows = conn.execute("""
    SELECT value, COUNT(DISTINCT entity_id) as n
    FROM classifications
    WHERE dimension = 'population'
    GROUP BY value
    ORDER BY n DESC
""").fetchall()
for r in rows:
    print(f"   {r['value']:<20}  {r['n']:>4} programs")

# ── 2. What population values do questions use? ───────────────────────────────
print(SEP)
print("2. POPULATION VALUES USED IN questions TABLE")
rows = conn.execute("""
    SELECT target_population, COUNT(*) as n
    FROM questions
    GROUP BY target_population
    ORDER BY n DESC
""").fetchall()
for r in rows:
    print(f"   '{r['target_population']}'  →  {r['n']} questions")

# ── 3. Key mismatch check ─────────────────────────────────────────────────────
print(SEP)
print("3. KEY MISMATCH CHECK")
print("   Values in classifications vs values used in questions:")
clf_values = {r["value"] for r in conn.execute(
    "SELECT DISTINCT value FROM classifications WHERE dimension='population'"
).fetchall()}
q_values = {r["target_population"] for r in conn.execute(
    "SELECT DISTINCT target_population FROM questions"
).fetchall() if r["target_population"]}

in_clf_not_q = clf_values - q_values
in_q_not_clf = q_values - clf_values
matched = clf_values & q_values

print(f"   Matched (exist in both)       : {sorted(matched)}")
print(f"   In classifications, not in Q  : {sorted(in_clf_not_q)}")
print(f"   In questions, not in classif  : {sorted(in_q_not_clf)}")
if in_q_not_clf:
    print("   ⚠ These question targets will NEVER match any program!")

# ── 4. Inuit/First_Nations relevant program count per question ────────────────
print(SEP)
print("4. RELEVANT PROGRAM COUNT FOR Inuit / First_Nations / Metis QUESTIONS")
special_pops = ["Inuit", "First_Nations", "Metis", "First Nations"]
for pop in special_pops:
    questions = conn.execute(
        "SELECT question_id, question_text FROM questions WHERE target_population = ?",
        (pop,)
    ).fetchall()
    if not questions:
        print(f"   {pop}: no questions found")
        continue
    for q in questions:
        n = conn.execute("""
            SELECT COUNT(DISTINCT p.entity_id) FROM programs p
            WHERE p.is_active = 1
            AND EXISTS (
                SELECT 1 FROM classifications c
                WHERE c.entity_id = p.entity_id
                AND c.dimension = 'population'
                AND c.value = ?
            )
        """, (pop,)).fetchone()[0]
        print(f"   {pop} | {q['question_id']} | {n} relevant programs | {q['question_text'][:55]}")

# ── 5. Visibility score distribution (not just averages) ─────────────────────
print(SEP)
print("5. VISIBILITY SCORE DISTRIBUTION (are scores bimodal 0 vs high?)")
rows = conn.execute("""
    SELECT platform,
           SUM(CASE WHEN visibility_score = 0 THEN 1 ELSE 0 END) as zero_count,
           SUM(CASE WHEN visibility_score > 0 AND visibility_score < 0.3 THEN 1 ELSE 0 END) as low_count,
           SUM(CASE WHEN visibility_score >= 0.3 AND visibility_score < 0.6 THEN 1 ELSE 0 END) as mid_count,
           SUM(CASE WHEN visibility_score >= 0.6 THEN 1 ELSE 0 END) as high_count,
           COUNT(*) as total
    FROM evaluations
    GROUP BY platform
""").fetchall()
print(f"   {'platform':<12} {'=0':>6} {'0–0.3':>6} {'0.3–0.6':>8} {'≥0.6':>6} {'total':>6}")
for r in rows:
    print(f"   {r['platform']:<12} {r['zero_count']:>6} {r['low_count']:>6} {r['mid_count']:>8} {r['high_count']:>6} {r['total']:>6}")

# ── 6. Hallucination breakdown ────────────────────────────────────────────────
print(SEP)
print("6. HALLUCINATION: how many candidates extracted per platform on average?")
print("   (uniform rate = same number of candidates failing threshold)")
rows = conn.execute("""
    SELECT platform,
           AVG(has_hallucination) as rate,
           SUM(has_hallucination) as flagged,
           COUNT(*) as total,
           SUM(CASE WHEN hallucination_note = '' OR hallucination_note IS NULL THEN 1 ELSE 0 END) as empty_notes
    FROM evaluations
    GROUP BY platform
""").fetchall()
for r in rows:
    print(f"   {r['platform']:<12}  rate={r['rate']:.2f}  flagged={r['flagged']}/{r['total']}  empty_notes={r['empty_notes']}")

# ── 7. Sample hallucination notes to inspect quality ─────────────────────────
print(SEP)
print("7. SAMPLE hallucination_note VALUES (first 3 flagged per platform)")
for platform in ["claude", "gemini", "chatgpt", "perplexity"]:
    notes = conn.execute("""
        SELECT hallucination_note FROM evaluations
        WHERE platform = ? AND has_hallucination = 1
        AND hallucination_note IS NOT NULL AND hallucination_note != ''
        LIMIT 3
    """, (platform,)).fetchall()
    print(f"\n   [{platform}]")
    for n in notes:
        print(f"   {n['hallucination_note'][:120]}")

print(SEP)
conn.close()
print("Done.")