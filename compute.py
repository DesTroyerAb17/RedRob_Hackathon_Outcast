# ============================================================
# CELL 2 — RANK + GENERATE SUBMISSION CSV
# Run after Cell 1. Loads pkl, scores, writes top-100 CSV.
# Must finish in < 5 min on CPU. Will take ~2 seconds.
# ============================================================

import pickle, csv

FEATURES_PKL   = "precomputed_features.pkl"
SUBMISSION_CSV = "submission.csv"   # rename to your team ID before upload

# ─── SCORING WEIGHTS ─────────────────────────────────────────
W_CAREER   = 0.35
W_PRODUCT  = 0.25   # up from 0.20
W_YOE      = 0.15
W_LOCATION = 0.10
W_GITHUB   = 0.05   # down from 0.10
# python_bonus max 0.05 + nth_bonus max 0.05 = remaining 0.10

BEHAVIORAL_CAP = 0.35   # 0.70 base + behavioral * 0.35

# ─── SCORE ───────────────────────────────────────────────────

def score(feat):
    base = (
        W_CAREER   * feat["career_signal"] +
        W_PRODUCT  * feat["product_score"] +
        W_YOE      * feat["yoe_score"] +
        W_LOCATION * feat["location_score"] +
        W_GITHUB   * feat["github_score"]
    )
    base += feat["python_bonus"]
    base += feat["nth_bonus"]

    beh_mult  = 1 - BEHAVIORAL_CAP + feat["behavioral"] * BEHAVIORAL_CAP
    score_val = base * beh_mult
    score_val *= feat["hopping_penalty"]
    score_val *= feat["hp_penalty"]

    return float(max(0.0, score_val))


# ─── REASONING ───────────────────────────────────────────────

def _availability_summary(m):
    last_a = m["last_active"]
    open_w = m["open_to_work"]
    notice = m["notice"]
    rr     = m["rr"]
    parts  = []

    parts.append("open to work" if open_w else "not marked open to work")

    if last_a <= 7:       parts.append("active this week")
    elif last_a <= 30:    parts.append(f"active {last_a}d ago")
    elif last_a <= 90:    parts.append(f"last active {last_a}d ago")
    else:                 parts.append(f"inactive ~{last_a // 30} months")

    if notice <= 30:      parts.append(f"{notice}d notice")
    elif notice <= 60:    parts.append(f"{notice}d notice (buyout possible)")
    else:                 parts.append(f"{notice}d notice (long)")

    if rr >= 0.7:         parts.append("highly responsive")
    elif rr >= 0.4:       parts.append(f"{rr:.0%} response rate")
    else:                 parts.append(f"low response rate ({rr:.0%})")

    return ", ".join(parts)


def _career_summary(feat):
    m                    = feat["meta"]
    retrieval_cos        = feat.get("retrieval_companies", [])
    retrieval_role_count = feat.get("retrieval_role_count", 0)
    n_l1                 = feat["level1_norm"]
    n_l2a                = feat["level2a_norm"]
    n_l2b                = feat["level2b_norm"]
    n_l2c                = feat["level2c_norm"]
    companies            = m["companies"]
    titles               = m["titles"]
    yoe                  = m["yoe"]

    strongest = max(
        [("embedding and semantic search systems", n_l2a),
         ("vector search and retrieval infrastructure", n_l2b),
         ("ranking evaluation and offline/online metrics", n_l2c)],
        key=lambda x: x[1]
    )[0]

    if retrieval_cos:
        co_str = " and ".join(retrieval_cos[:2])
        where  = f"at {co_str}"
    elif companies:
        where  = f"at {companies[0]}"
    else:
        where  = "across career"

    current_title   = titles[0]    if titles    else m["title"]
    current_company = companies[0] if companies else m["company"]

    if n_l1 >= 0.70 and retrieval_role_count >= 3:
        return (f"Currently {current_title} at {current_company} ({yoe:.0f} YOE); "
                f"retrieval/ranking work confirmed across {retrieval_role_count} roles "
                f"{where}, strongest signal in {strongest}")
    elif n_l1 >= 0.55 and retrieval_role_count >= 2:
        return (f"{current_title} at {current_company} ({yoe:.0f} YOE); "
                f"production {strongest} confirmed {where} across {retrieval_role_count} roles")
    elif n_l1 >= 0.40:
        return (f"{current_title} at {current_company} ({yoe:.0f} YOE); "
                f"retrieval/ranking signal confirmed {where} — "
                f"strongest match: {strongest}")
    elif n_l1 >= 0.20:
        return (f"{current_title} at {current_company} ({yoe:.0f} YOE); "
                f"adjacent {strongest} background {where} — partial match to JD")
    else:
        return (f"{current_title} at {current_company} ({yoe:.0f} YOE); "
                f"limited direct retrieval signal — scored on adjacent ML/engineering background")


def _strengths(feat, scored_list, rank):
    curr_beh  = feat["behavioral"]
    curr_prod = feat["product_score"]
    curr_loc  = feat["location_score"]
    curr_yoe  = feat["yoe_score"]
    curr_gh   = feat["github_score"]
    curr_dep  = feat.get("retrieval_role_count", 0)
    points    = []

    if curr_beh >= 0.90:   points.append("fully available and highly responsive")
    elif curr_beh >= 0.75: points.append("good availability signals")

    if curr_loc == 1.0:    points.append("Pune/Noida based")
    elif curr_loc >= 0.95: points.append("India-based, willing to relocate")

    if curr_yoe == 1.0:    points.append("YOE in JD sweet spot (6–8 years)")

    if curr_gh >= 0.60:    points.append(f"strong open-source activity (GitHub {int(curr_gh*100)})")
    elif curr_gh >= 0.30:  points.append(f"active on GitHub (score {int(curr_gh*100)})")

    if curr_dep >= 3:      points.append(f"{curr_dep} roles with confirmed retrieval/ranking work")
    elif curr_dep == 2:    points.append("retrieval work confirmed across 2 roles")

    if curr_prod >= 0.85:  points.append("strong product company background throughout career")
    elif curr_prod >= 0.70: points.append("solid product company history")

    idx = rank - 1
    if idx + 1 < len(scored_list):
        _, _, _, _, next_feat = scored_list[idx + 1]
        edges = []
        if curr_beh  - next_feat["behavioral"]     > 0.08: edges.append("better availability")
        if curr_prod - next_feat["product_score"]   > 0.10: edges.append("stronger product company mix")
        if feat["career_signal"] - next_feat["career_signal"] > 0.05: edges.append("deeper retrieval signal")
        if curr_loc  - next_feat["location_score"]  > 0.10: edges.append("better location fit")
        if edges:
            points.append(f"edges rank {rank+1} on: {', '.join(edges[:2])}")

    return points[:3]


def _concerns(feat):
    m        = feat["meta"]
    last_a   = m["last_active"]
    rr       = m["rr"]
    notice   = m["notice"]
    int_r    = m["int_rate"]
    hp       = feat["hp_penalty"]
    hop      = feat["hopping_penalty"]
    mode     = m["work_mode"]
    country  = m["country"]
    relocate = m["relocate"]
    concerns = []

    if last_a > 180:      concerns.append(f"inactive ~{last_a // 30} months — reachability uncertain")
    elif last_a > 90:     concerns.append(f"last active {last_a}d ago")

    if rr < 0.10:         concerns.append(f"very low recruiter response rate ({rr:.0%})")
    elif rr < 0.20:       concerns.append(f"low response rate ({rr:.0%})")

    if notice > 90:       concerns.append(f"{notice}d notice — high buyout cost")
    elif notice > 60:     concerns.append(f"{notice}d notice (JD prefers ≤30, can buy out up to 30)")

    if int_r < 0.40:      concerns.append(f"low interview completion ({int_r:.0%})")

    if hp < 0.60:         concerns.append("significant profile inconsistencies detected")
    elif hp < 0.85:       concerns.append("minor profile consistency issues")

    if hop < 0.85:        concerns.append("recent short-tenure hopping pattern")
    elif hop < 0.95:      concerns.append("one short stint in recent career")

    if country != "india" and not relocate:
        concerns.append(f"based in {country.title()}, not willing to relocate")
    elif country != "india":
        concerns.append("based outside India — relocation needed")

    if mode == "remote":  concerns.append("prefers remote; JD expects hybrid with quarterly offsites")
    if not m["open_to_work"]: concerns.append("not marked open to work")

    return concerns


def build_reasoning(feat, rank, score_val, scored_list):
    career_str = _career_summary(feat)
    avail_str  = _availability_summary(feat["meta"])
    strengths  = _strengths(feat, scored_list, rank)
    concerns   = _concerns(feat)

    parts = [career_str]
    if strengths:
        parts.append("; ".join(strengths))
    parts.append(f"availability: {avail_str}")
    if concerns:
        parts.append(f"concern: {concerns[0]}")
        if len(concerns) > 1 and score_val >= 0.60:
            parts.append(f"also: {concerns[1]}")

    return ". ".join(parts) + "."


# ─── RUN ─────────────────────────────────────────────────────

def main():
    print(f"Loading {FEATURES_PKL} ...")
    with open(FEATURES_PKL, "rb") as f:
        features = pickle.load(f)

    print(f"Scoring {len(features):,} candidates ...")
    scored = [
        (score(feat), feat["level1_raw"], feat["meta"]["last_active"],
         feat["candidate_id"], feat)
        for feat in features
    ]

    scored.sort(key=lambda x: (-x[0], -x[1], x[2], x[3]))

    total_qualified = len(scored)
    top100          = scored[:100]

    for i in range(len(top100) - 1):
        assert top100[i][0] >= top100[i+1][0], \
            f"Score order violated at rank {i+1}: {top100[i][0]:.4f} < {top100[i+1][0]:.4f}"

    print(f"Writing {SUBMISSION_CSV} ...")
    with open(SUBMISSION_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (s, l1raw, last_a, cid, feat) in enumerate(top100, start=1):
            r = build_reasoning(feat, rank, s, top100)
            writer.writerow([cid, rank, f"{s:.4f}", r])

    print(f"Done → {SUBMISSION_CSV}")

    print("\nTop 10:")
    for rank, (s, l1raw, last_a, cid, feat) in enumerate(top100[:10], start=1):
        m = feat["meta"]
        print(f"  #{rank:>2} {cid} score={s:.4f} | {m['title']} @ {m['company']} "
              f"| YOE={m['yoe']:.0f} | cs={feat['career_signal']:.3f} "
              f"| prod={feat['product_score']:.3f} | beh={feat['behavioral']:.3f} "
              f"| gh={feat['github_score']:.2f} | hp={feat['hp_penalty']:.2f} "
              f"| hop={feat['hopping_penalty']:.2f}")

    print(f"\nScore range: {top100[0][0]:.4f} → {top100[99][0]:.4f}")
    print(f"Total qualified: {total_qualified:,}")

    print("\nSample reasoning (ranks 1, 10, 50, 100):")
    for rank in [1, 10, 50, 100]:
        s, _, _, cid, feat = top100[rank - 1]
        r = build_reasoning(feat, rank, s, top100)
        print(f"\n  Rank {rank} ({cid}, score={s:.4f}):")
        print(f"  {r}")


if __name__ == "__main__":
    main()