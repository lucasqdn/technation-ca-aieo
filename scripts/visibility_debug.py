"""
visibility_debug.py
Diagnoses why visibility scores are low by showing exactly what's happening
for a single response: relevant programs, their similarities to the response,
and what the threshold is cutting off.

Run: python visibility_debug.py
"""
import sqlite3, sys
from pathlib import Path

try:
    from config import DB_PATH
    from embed import ProgramEmbedder
except ImportError as e:
    print(f"ERROR: {e} — run from your project root.")
    sys.exit(1)

SIMILARITY_THRESHOLD = 0.45
RELEVANCE_TOP_K = 20
CHROMA_FETCH_K = 300

SEP = "─" * 70

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
embedder = ProgramEmbedder()

# ── Pick one response to inspect ─────────────────────────────────────────────
row = conn.execute("""
    SELECT r.response_id, r.question_id, r.platform,
           r.response_text,
           q.question_text,
           q.target_population, q.target_sector, q.target_region
    FROM ai_responses r
    JOIN questions q ON r.question_id = q.question_id
    LIMIT 1
""").fetchone()

print(SEP)
print(f"Question : {row['question_id']} — {row['question_text']}")
print(f"Platform : {row['platform']}")
print(f"Dims     : pop={row['target_population']}  sec={row['target_sector']}  reg={row['target_region']}")
print(f"Response (first 400 chars):\n  {row['response_text'][:400]}")

# ── Stage 1: classification-matched programs ──────────────────────────────────
print(SEP)
print("STAGE 1 — Classification-matched programs (SQL filter)")

pop, sec, reg = row['target_population'] or '', row['target_sector'] or '', row['target_region'] or ''
clauses, params = [], []
if pop and pop != "general_public":
    clauses.append("EXISTS (SELECT 1 FROM classifications c WHERE c.entity_id=p.entity_id AND c.dimension='population' AND c.value=?)")
    params.append(pop)
if sec and sec != "public_services":
    clauses.append("EXISTS (SELECT 1 FROM classifications c WHERE c.entity_id=p.entity_id AND c.dimension='sector' AND c.value=?)")
    params.append(sec)
if reg and reg != "national":
    clauses.append("EXISTS (SELECT 1 FROM classifications c WHERE c.entity_id=p.entity_id AND c.dimension='region' AND c.value IN (?, 'national'))")
    params.append(reg)

base = "SELECT DISTINCT p.entity_id, p.name_en FROM programs p WHERE p.is_active = 1"
query = (base + " AND " + " AND ".join(clauses)) if clauses else base + " LIMIT 200"
stage1 = conn.execute(query, params).fetchall()
candidate_ids = {r["entity_id"]: r["name_en"] for r in stage1}
print(f"  Programs after SQL filter: {len(candidate_ids)}")

# ── Stage 2: semantic re-rank by question ─────────────────────────────────────
print(SEP)
print("STAGE 2 — Semantic re-ranking by question text")

question_hits = embedder.query(row['question_text'], n_results=min(len(candidate_ids) + 50, 200))
scored = []
seen = set()
for h in question_hits:
    eid = h['entity_id']
    if eid in candidate_ids and eid not in seen:
        scored.append({'entity_id': eid, 'name_en': candidate_ids[eid], 'relevance_sim': h['similarity']})
        seen.add(eid)
for eid, name in candidate_ids.items():
    if eid not in seen:
        scored.append({'entity_id': eid, 'name_en': name, 'relevance_sim': 0.0})

scored.sort(key=lambda x: x['relevance_sim'], reverse=True)
relevant = scored[:RELEVANCE_TOP_K]

print(f"  Top {RELEVANCE_TOP_K} relevant programs (by question similarity):")
for p in relevant:
    print(f"    [{p['entity_id']:>5}] q_sim={p['relevance_sim']:.3f}  {p['name_en'][:70]}")

# ── Stage 3: response vs relevant programs ────────────────────────────────────
print(SEP)
print("STAGE 3 — Response similarity against relevant programs")

relevant_id_set = {p['entity_id'] for p in relevant}
response_hits = embedder.query(row['response_text'], n_results=CHROMA_FETCH_K)

# Map entity_id → similarity for ALL top-300 hits
all_hit_sims = {h['entity_id']: h['similarity'] for h in response_hits}

print(f"  {'entity_id':>9}  {'q_sim':>6}  {'r_sim':>6}  {'covered':>8}  name")
covered_count = 0
for p in relevant:
    eid = p['entity_id']
    r_sim = all_hit_sims.get(eid, None)
    in_top300 = r_sim is not None
    covered = in_top300 and r_sim >= SIMILARITY_THRESHOLD
    if covered:
        covered_count += 1
    flag = "✓" if covered else ("(below thresh)" if in_top300 else "NOT IN TOP-300")
    r_sim_str = f"{r_sim:.3f}" if r_sim is not None else "  N/A"
    print(f"  [{eid:>7}]  {p['relevance_sim']:.3f}   {r_sim_str}   {flag:>14}  {p['name_en'][:55]}")

print(f"\n  visibility = {covered_count}/{len(relevant)} = {covered_count/len(relevant):.3f}")

# ── Threshold sensitivity ────────────────────────────────────────────────────
print(SEP)
print("THRESHOLD SENSITIVITY — covered count at different thresholds")
r_sims = [all_hit_sims[p['entity_id']] for p in relevant if p['entity_id'] in all_hit_sims]
print(f"  Similarities of relevant programs that DO appear in top-{CHROMA_FETCH_K}:")
print(f"  {sorted(r_sims, reverse=True)}")
print()
for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    n = sum(1 for s in r_sims if s >= thresh)
    print(f"  threshold={thresh:.2f}  →  covered={n}/{len(relevant)}  visibility={n/len(relevant):.3f}")

# ── What are the TOP response hits that are NOT in relevant set? ──────────────
print(SEP)
print("TOP-10 RESPONSE HITS THAT ARE NOT IN THE RELEVANT SET")
print("(These are what the response is actually about — may reveal topic mismatch)")
non_relevant_hits = [h for h in response_hits if h['entity_id'] not in relevant_id_set][:10]
for h in non_relevant_hits:
    name = conn.execute("SELECT name_en FROM programs WHERE entity_id=?", (h['entity_id'],)).fetchone()
    name_str = name['name_en'][:65] if name else "unknown"
    print(f"  sim={h['similarity']:.3f}  [{h['entity_id']:>5}]  {name_str}")

print(SEP)
conn.close()