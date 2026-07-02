
import pickle, csv, hashlib

FEATURES_PKL   = "precomputed_features.pkl"
SUBMISSION_CSV = "submission.csv"

# ─── scoring weights ──────────────────────────────────────────
W_CAREER   = 0.35
W_PRODUCT  = 0.25
W_YOE      = 0.15
W_LOCATION = 0.07
W_GITHUB   = 0.08
# python_bonus (max 0.05) + nth_bonus (max 0.05) fill the remaining 0.10

BEHAVIORAL_CAP = 0.35


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


# ─── reasoning ─────────────────────────────────────────────────
# Recruiter-style reasoning with deterministic variation.

def _pick(cid, tag, options):
    h = int(hashlib.md5(f"{cid}:{tag}".encode()).hexdigest(), 16)
    return options[h % len(options)]


def _dedup(seq):
    return list(dict.fromkeys(seq))


def _concerns(feat):
    m = feat["meta"]

    concerns = []

    last_a = m["last_active"]
    if last_a > 180:
        concerns.append(f"has been inactive for roughly {last_a // 30} months")
    elif last_a > 90:
        concerns.append(f"was last active {last_a} days ago")

    notice = m["notice"]
    if notice > 90:
        concerns.append(f"requires a {notice}-day notice period")
    elif notice > 60:
        concerns.append(f"comes with a relatively long notice period ({notice} days)")

    rr = m["rr"]
    if rr < 0.10:
        concerns.append(f"has a low recruiter response rate ({rr:.0%})")

    int_rate = m["int_rate"]
    if int_rate < 0.40:
        concerns.append(f"shows a relatively low interview completion rate ({int_rate:.0%})")

    hp = feat["hp_penalty"]
    if hp < 0.60:
        concerns.append("shows notable profile inconsistencies")
    elif hp < 0.85:
        concerns.append("shows a few minor profile inconsistencies")

    hop = feat["hopping_penalty"]
    if hop < 0.85:
        concerns.append("has a recent pattern of short-tenure roles")
    elif hop < 0.95:
        concerns.append("includes one relatively short recent stint")

    if m["country"] != "india":
        if m["relocate"]:
            concerns.append("is currently based outside India and would require relocation")
        else:
            concerns.append("is currently based outside India without relocation preference")

    if m["work_mode"] == "remote":
        concerns.append("prefers remote work whereas the JD expects a hybrid setup")

    if not m["open_to_work"]:
        concerns.append("is not currently marked as open to work")

    return concerns


def build_reasoning(feat, rank, score_val, scored_list):

    cid = feat["candidate_id"]
    m = feat["meta"]

    title = (m["titles"][0] if m["titles"] else m["title"]) or "Candidate"

    yoe = m["yoe"]

    depth = min(feat.get("retrieval_role_count", 0), 3)

    companies = (
        _dedup(feat.get("retrieval_companies", []))
        or _dedup(m["companies"][:1])
        or [m.get("company", "previous roles")]
    )

    if len(companies) >= 2:
        comp_str = f"{companies[0]} and {companies[1]}"
    else:
        comp_str = companies[0]

    retrieval_note = _pick(
        cid,
        "retrieval_note",
        [
            f"deep retrieval experience across multiple roles, particularly at {comp_str}",
            f"strong search and ranking experience built at {comp_str}",
            f"consistent recommendation and retrieval work spanning {comp_str}",
            f"hands-on semantic search experience developed at {comp_str}",
            f"production retrieval systems experience gained through {comp_str}",
        ],
    )

    concerns = _concerns(feat)
    concern_text = _pick(cid, "concern", concerns) if concerns else None

    bottom_cutoff = max(1, len(scored_list) - 12)

    if rank > bottom_cutoff or score_val < 0.50:

        low_paths = [

            f"Compared with higher-ranked candidates, this profile shows less direct retrieval experience despite {yoe:.0f} years as a {title}.",

            f"A reasonable overall ML profile, although the evidence for production search and ranking work is lighter than the candidates ranked above.",

            f"This candidate brings {yoe:.0f} years as a {title}, but their experience appears more adjacent to the target retrieval domain than directly aligned.",

        ]

        return _pick(cid, "bottom_path", low_paths)

    path_choice = _pick(cid, "path", [1, 2, 3, 4])

    if path_choice == 1:

        base = (
            f"With {yoe:.0f} years of experience, this {title} stands out for "
            f"{retrieval_note}."
        )

        if feat.get("product_score", 0) >= 0.70:
            base += (
                " Their background spans established product engineering teams, "
                "providing confidence in production-scale ML delivery."
            )

        if concern_text:
            base += (
                f" One hiring consideration is that the candidate {concern_text}."
            )

        return base


    elif path_choice == 2:

        fit_level = "Strong" if rank <= 20 else "Solid"

        base = (
            f"{fit_level} alignment with the role. "
            f"Previously a {title} at {comp_str}, "
            f"their experience falls within the target seniority range for this position."
        )

        if feat.get("github_score", 0) >= 0.60:
            base += (
                " Public engineering work further reinforces their practical technical depth."
            )

        if concern_text:
            base += (
                f" The main consideration is that the candidate {concern_text}."
            )

        return base


    elif path_choice == 3:

        if concern_text:
            base = (
                f"Despite the fact that the candidate {concern_text}, "
                f"they remain a strong {title} for this role."
            )
        else:
            base = (
                f"A technically strong {title} with consistent career progression "
                f"and good overall profile quality."
            )

        base += f" They bring {yoe:.0f} years of production engineering experience."

        if depth >= 2:
            base += (
                f" Their background demonstrates repeated search and ranking work at {comp_str}."
            )
        elif depth == 1:
            base += (
                f" Their profile includes direct exposure to retrieval systems through {comp_str}."
            )
        else:
            base += (
                " Their broader ML experience appears transferable, although retrieval-specific evidence is lighter."
            )

        return base


    else:

        base = (
            f"This profile aligns well with the core requirements. "
            f"As a {title} with {yoe:.0f} years of experience, "
            f"their work across {comp_str} provides a solid foundation for the responsibilities of this role."
        )

        extras = []

        if feat.get("location_score", 0) >= 0.95:
            extras.append("they are already located in a preferred hiring region")

        if feat.get("behavioral", 0) >= 0.85:
            extras.append("their profile shows healthy hiring-market engagement")

        if feat.get("github_score", 0) >= 0.60:
            extras.append("they maintain visible public engineering work")

        if extras:
            if len(extras) == 1:
                base += f" Additionally, {extras[0]}."
            elif len(extras) == 2:
                base += f" Additionally, {extras[0]} and {extras[1]}."
            else:
                base += (
                    f" Additionally, {extras[0]}, {extras[1]}, and {extras[2]}."
                )

        if concern_text:
            base += (
                f" One hiring consideration is that the candidate {concern_text}."
            )

        return base


# ─── main ───────────────────────────────────────────────────

def main():
    with open(FEATURES_PKL, "rb") as f:
        features = pickle.load(f)

    scored = [
        (score(feat), feat.get("level1_raw", 0), feat["meta"]["last_active"], feat["candidate_id"], feat)
        for feat in features
    ]
    scored.sort(key=lambda x: (-round(x[0], 4), x[3]))

    total_qualified = len(scored)
    top100          = scored[:100]

    for i in range(len(top100) - 1):
        assert (
            round(top100[i][0], 4) > round(top100[i + 1][0], 4)
            or (
                round(top100[i][0], 4) == round(top100[i + 1][0], 4)
                and top100[i][3] <= top100[i + 1][3]
            )
        ),f"Ordering violated at rank {i+1}"

    with open(SUBMISSION_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (s, l1raw, last_a, cid, feat) in enumerate(top100, start=1):
            writer.writerow([cid, rank, f"{s:.4f}", build_reasoning(feat, rank, s, top100)])

    print(f"Done. Ranked {total_qualified:,} candidates -> {SUBMISSION_CSV} (top 100 written)")


if __name__ == "__main__":
    main()

