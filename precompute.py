

import json, pickle, os, numpy as np
from datetime import date, datetime

CANDIDATES_FILE     = "candidates.jsonl"
OUTPUT_PKL          = "precomputed_features.pkl"
TODAY               = date(2026, 6, 21)
BGE_BATCH_SIZE      = 1024
RETRIEVAL_THRESHOLD = 0.52
MAX_DESC_WORDS      = 120

LOCAL_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "models", "bge-base-en-v1.5")

# ─── anchor sentences ─────────────────────────────────────────

LEVEL1_ANCHORS = [
    "Built and deployed a semantic search system serving real users in production",
    "Developed a recommendation engine ranking items for millions of users at a product company",
    "Owned end-to-end candidate matching pipeline deployed to production at a tech company",
    "Improved search relevance for an e-commerce or marketplace platform using ML in production",
    "Shipped a ranking system that personalised content feeds at scale for real users",
    "Designed and operated retrieval infrastructure handling query-document matching in production",
    "Built ML-based job-candidate matching system deployed and serving real hiring traffic",
    "Built vector search pipeline with real-time index updates and quality monitoring",
    "Owned the relevance layer of a marketplace or platform search product end to end",
    "Deployed dense retrieval model and managed embedding drift and index refresh in production",
    "Led migration from keyword search to embedding-based retrieval system at scale",
    "Built and maintained two-tower retrieval model for personalised recommendations",
    "Designed and shipped learning-to-rank model improving search quality in production",
    "Owned discovery feed ranking system at a consumer tech or food-delivery product company",
    "Built production retrieval system combining dense and sparse search for a real product",
    "Developed item or document ranking pipeline for a marketplace or platform product",
    "Maintained and improved ranking model serving real-time product search to millions of users",
    "Built search and retrieval system for a SaaS or internet product used by real customers",
    "Worked closely with product managers to define ranking objectives and success metrics",
    "Collaborated with PM to translate recruiter and user feedback into ranking model improvements",
    "Owned the ML side of a product feature end to end including PM collaboration and metric definition",
    "Shipped a working ranker quickly then iterated based on real user feedback and engagement data",
    "Balanced ML quality with product shipping speed to deliver working systems under tight timelines",
]

EMBEDDING_ANCHORS = [
    "Trained and served sentence embedding models in production with drift monitoring",
    "Deployed text embedding pipeline handling index refresh and retrieval quality regression",
    "Built dense vector representations for document retrieval deployed to real users",
    "Fine-tuned transformer model for semantic similarity and retrieval tasks in production",
    "Managed embedding model versioning and retrieval quality monitoring at scale",
    "Migrated keyword search system to dense embedding-based retrieval in production",
    "Built bi-encoder model for semantic search deployed to serve real user queries",
    "Operated embedding pipeline with automated reindexing after model version updates",
    "Developed cross-encoder reranking layer on top of dense retrieval in production system",
    "Trained sentence transformer models for semantic text matching at production scale",
]

VECTORDB_ANCHORS = [
    "Operated vector database handling approximate nearest neighbor search at scale in production",
    "Built hybrid search combining dense vector retrieval with BM25 lexical search",
    "Maintained search index with real-time update pipelines and latency monitoring in production",
    "Deployed ANN index serving low-latency vector similarity search for real users",
    "Managed search infrastructure with relevance monitoring and quality regression alerts",
    "Built FAISS-based search system handling millions of vectors in production",
    "Operated Pinecone or Qdrant or Weaviate or Milvus index for production semantic search",
    "Designed vector storage and retrieval architecture for production ML system at scale",
    "Maintained Elasticsearch or OpenSearch cluster for hybrid keyword and semantic search",
    "Built and operated approximate nearest neighbor infrastructure serving real user traffic",
]

EVAL_ANCHORS = [
    "Designed offline evaluation pipeline measuring ranking quality with NDCG and precision metrics",
    "Built A/B testing framework to measure search or recommendation improvements in production",
    "Set up relevance labeling pipeline and human evaluation process for ranking system quality",
    "Measured ranking quality using NDCG MRR and MAP metrics offline and correlated with online",
    "Correlated offline evaluation metrics with online user engagement signals from A/B tests",
    "Owned the evaluation infrastructure for a production search or recommendation system",
    "Built offline-to-online correlation analysis for ranking model quality measurement",
    "Designed interleaving experiments and A/B tests to measure search quality improvements",
    "Set up click-through rate and engagement metrics to evaluate recommendation quality",
    "Built annotation pipeline and evaluation framework to measure retrieval improvement",
]

NTH_FINETUNING = [
    "Fine-tuned large language model using LoRA or QLoRA for domain-specific retrieval task",
    "Applied parameter-efficient fine-tuning to adapt foundation model for production use",
    "Used PEFT methods to fine-tune transformer model on custom domain dataset",
    "Fine-tuned LLM on domain data using LoRA adapters for production deployment",
    "Trained domain-adapted language model using parameter-efficient fine-tuning techniques",
    "Applied instruction tuning or RLHF to improve language model for production application",
]
NTH_L2R = [
    "Built learning-to-rank model using XGBoost or LightGBM for search ranking in production",
    "Trained gradient boosted tree model for ranking documents in search results at scale",
    "Implemented neural learning-to-rank model improving search relevance in production",
    "Designed pointwise or pairwise ranking model for information retrieval system",
    "Built LambdaMART or RankNet model for document ranking in production search system",
    "Trained learning-to-rank model combining multiple relevance signals for search product",
]
NTH_HRTECH = [
    "Built ML systems for job recommendation or candidate matching on a recruiting platform",
    "Developed ranking system for recruiting or HR technology product at a company",
    "Worked on marketplace matching candidates to job opportunities using machine learning",
    "Built search and recommendation features for a talent acquisition or hiring platform",
    "Owned ML features for candidate ranking or job matching in a human resources product",
    "Developed candidate scoring pipeline at an HR-tech or recruiting technology company",
]
NTH_DISTRIBUTED = [
    "Built distributed machine learning training system handling large-scale data",
    "Optimised ML inference pipeline for low-latency serving at high throughput in production",
    "Designed distributed feature computation pipeline for large-scale ML production system",
    "Scaled ML serving infrastructure to handle millions of real-time requests per day",
    "Optimised model serving latency and throughput for high-traffic production system",
    "Built large-scale distributed pipeline for ML feature engineering at production scale",
]
NTH_OPENSOURCE = [
    "Contributed code or bug fixes to open-source machine learning or NLP library",
    "Maintained open-source project in the AI or information retrieval ecosystem",
    "Published open-source implementation of a retrieval or ranking algorithm",
    "Active contributor to popular open-source ML or search framework on GitHub",
    "Released open-source tool used by the ML or NLP community on GitHub",
    "Contributed to open-source search or recommendation library used in production",
]

NTH_SETS = [
    ("finetuning",  NTH_FINETUNING,  0.04),
    ("l2r",         NTH_L2R,         0.05),
    ("hrtech",      NTH_HRTECH,      0.02),
    ("distributed", NTH_DISTRIBUTED, 0.04),
    ("opensource",  NTH_OPENSOURCE,  0.07),
]

CONSULTING = {
    "tcs", "infosys", "wipro", "capgemini", "hcl",
    "mindtree", "accenture", "cognizant", "tech mahindra", "mphasis"
}
FILLER = {
    "hooli", "initech", "pied piper",
    "dunder mifflin", "wayne enterprises",
    "acme corp", "globex inc", "stark industries",
}
CV_TITLES = ["computer vision", "vision engineer", "speech recogn", "robotics"]

# ─── helpers ────────────────────────────────────────────────

def days_since(date_str):
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        return max(0, (TODAY - d).days)
    except:
        return 9999

def is_consulting_only(career):
    if not career:
        return False
    return all(any(co in h.get("company", "").lower() for co in CONSULTING) for h in career)

def is_wrong_domain(career_text):
    WRONG = [
        "computer vision", "object detection", "image classification",
        "image segmentation", "speech recognition", "text-to-speech",
        "autonomous driving", "robotics", "convolutional neural",
    ]
    NLP_IR = [
        "retrieval", "ranking", "recommendation", "nlp", "natural language",
        "search relevance", "embedding", "information retrieval", "semantic search",
        "text matching", "query understanding",
    ]
    has_wrong = any(w in career_text for w in WRONG)
    has_nlp   = any(n in career_text for n in NLP_IR)
    return has_wrong and not has_nlp

def honeypot_check(candidate):
    profile     = candidate.get("profile", {})
    yoe         = profile.get("years_of_experience", 0)
    yoe_mo      = yoe * 12
    skills      = candidate.get("skills", [])
    career      = candidate.get("career_history", [])
    sigs        = candidate.get("redrob_signals", {})
    assessments = sigs.get("skill_assessment_scores", {})
    tolerance   = max(6, yoe_mo * 0.15)
    penalties   = []

    impossible = sum(1 for s in skills if s.get("duration_months", 0) > yoe_mo * 1.5 + tolerance)
    if impossible >= 7:   penalties.append(0.25)
    elif impossible >= 5: penalties.append(0.55)
    elif impossible >= 3: penalties.append(0.80)

    min_dur = max(12, yoe_mo * 0.1)
    fake_experts = sum(
        1 for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and s.get("duration_months", 99) < min_dur * 0.25
    )
    if fake_experts >= 3:   penalties.append(0.25)
    elif fake_experts >= 1: penalties.append(0.70)

    date_ranges = []
    for h in career:
        try:
            s   = datetime.strptime(h["start_date"][:10], "%Y-%m-%d").date()
            e_r = h.get("end_date")
            e   = datetime.strptime(e_r[:10], "%Y-%m-%d").date() if e_r else TODAY
            date_ranges.append((s, e))
        except:
            pass
    overlaps = 0
    for i in range(len(date_ranges)):
        for j in range(i + 1, len(date_ranges)):
            s1, e1 = date_ranges[i]
            s2, e2 = date_ranges[j]
            if (min(e1, e2) - max(s1, s2)).days > 30:
                overlaps += 1
    if overlaps >= 3:   penalties.append(0.25)
    elif overlaps >= 2: penalties.append(0.55)

    for s in skills:
        name = s.get("name", "")
        if name in assessments and s.get("proficiency") == "expert" and assessments[name] < 25:
            penalties.append(0.55)
            break

    descs = [h.get("description", "").strip() for h in career if h.get("description", "").strip()]
    if len(descs) >= 3:
        unique = len(set(descs))
        if unique == 1:
            penalties.append(0.15)
        elif unique < len(descs) * 0.5:
            penalties.append(0.55)

    return min(penalties) if penalties else 1.0

def compute_hopping_penalty(career):
    if len(career) < 3:
        return 1.0
    non_current = [h for h in career if not h.get("is_current", False)]
    recent      = non_current[:3]
    short_hops  = 0
    for i in range(len(recent) - 1):
        dur            = recent[i].get("duration_months", 99)
        company_change = recent[i].get("company", "").lower() != recent[i + 1].get("company", "").lower()
        if dur < 18 and company_change:
            short_hops += 1
    if short_hops >= 2: return 0.80
    if short_hops >= 1: return 0.92
    return 1.0

def compute_yoe_score(yoe):
    if yoe < 1:          return 0.0
    if 6 <= yoe <= 8:    return 1.0
    if 5 <= yoe < 6:     return 0.9
    if 8 < yoe <= 9:     return 0.9
    if 4 <= yoe < 5:     return 0.8
    if 9 < yoe <= 10:    return 0.8
    if 3 <= yoe < 4:     return 0.7
    if 10 < yoe <= 11:   return 0.7
    if 2 <= yoe < 3:     return 0.5
    if 11 < yoe <= 12:   return 0.5
    if 1 <= yoe < 2:     return 0.3
    if 12 < yoe <= 13:   return 0.3
    return max(0.1, 0.3 - (yoe - 13) * 0.1)

def compute_location_score(profile, sigs):
    loc      = profile.get("location", "").lower()
    country  = profile.get("country", "").lower()
    relocate = sigs.get("willing_to_relocate", False)
    mode     = sigs.get("preferred_work_mode", "")
    if country == "india":
        if any(c in loc for c in ["pune", "noida"]): score = 1.0
        elif relocate:                                score = 0.95
        else:                                         score = 0.75
    elif relocate:
        score = 0.35
    else:
        score = 0.0
    if mode == "remote":
        score *= 0.90
    return score

def compute_product_score(career):
    SIZE_SCORE = {
        "1-10": 0.40, "11-50": 0.75, "51-200": 0.85,
        "201-500": 0.90, "501-1000": 0.85, "1001-5000": 0.75,
        "5001-10000": 0.65, "10001+": 0.50,
    }
    role_scores, weights = [], []
    for i, h in enumerate(career):
        comp = h.get("company", "").lower()
        size = h.get("company_size", "1-10")
        base = SIZE_SCORE.get(size, 0.5)
        if any(co in comp for co in CONSULTING) or any(f in comp for f in FILLER):
            base = 0.05
        role_scores.append(base)
        weights.append(1.0 / (i + 1))
    if not role_scores:
        return 0.30
    return sum(r * w for r, w in zip(role_scores, weights)) / sum(weights)

def compute_python_bonus(skills):
    for s in skills:
        if "python" in s.get("name", "").lower() and s.get("proficiency") in ("intermediate", "advanced", "expert"):
            return 0.05
    return 0.0

def compute_behavioral(sigs):
    last_act = days_since(sigs.get("last_active_date", "2000-01-01"))
    open_w   = sigs.get("open_to_work_flag", False)
    rr       = sigs.get("recruiter_response_rate", 0.0)
    rt_h     = sigs.get("avg_response_time_hours", 999)
    notice   = sigs.get("notice_period_days", 90)
    int_rate = sigs.get("interview_completion_rate", 0.5)

    open_s   = 1.0 if open_w else 0.30
    act_s    = (1.0 if last_act <= 30  else 0.85 if last_act <= 60
                else 0.70 if last_act <= 90 else 0.50 if last_act <= 180 else 0.30)
    rr_s     = (1.0 if rr >= 0.7 else 0.80 if rr >= 0.4
                else 0.60 if rr >= 0.15 else 0.30)
    rt_s     = (1.0 if rt_h <= 24 else 0.90 if rt_h <= 72
                else 0.75 if rt_h <= 168 else 0.50)
    notice_s = (1.0 if notice <= 30 else 0.70 if notice <= 60
                else 0.50 if notice <= 90 else 0.30)
    int_s    = (1.0 if int_rate >= 0.8 else 0.85 if int_rate >= 0.6
                else 0.70 if int_rate >= 0.4 else 0.50)

    raw = open_s * 0.25 + act_s * 0.27 + rr_s * 0.20 + rt_s * 0.05 + notice_s * 0.15 + int_s * 0.08

    completeness = sigs.get("profile_completeness_score", 0)
    if completeness >= 80: raw += 0.02

    oar = sigs.get("offer_acceptance_rate", -1)
    if oar >= 0.7:       raw += 0.02
    elif 0 < oar < 0.3:  raw -= 0.02

    saved = sigs.get("saved_by_recruiters_30d", 0)
    if saved >= 10:  raw += 0.02
    elif saved >= 5: raw += 0.01

    return min(raw, 1.0)

def build_role_text(role):
    title   = role.get("title", "")
    company = role.get("company", "")
    desc    = role.get("description", "").strip()
    if desc:
        words = desc.split()
        if len(words) > MAX_DESC_WORDS:
            desc = " ".join(words[:MAX_DESC_WORDS])
        return f"{title} at {company}: {desc}"
    return f"{title} at {company}"

def norm(x, lo=0.55, hi=0.72):
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

def load_model():
    from sentence_transformers import SentenceTransformer
    import torch
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(os.cpu_count())
    return SentenceTransformer(LOCAL_MODEL_PATH, device="cpu")

def encode(model, texts, show_progress=False):
    return model.encode(
        texts,
        batch_size=BGE_BATCH_SIZE,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

def top_k_mean(sims, k=3):
    return float(np.sort(sims)[-k:].mean())

def passes_gate(c, career_text):
    profile  = c.get("profile", {})
    sigs     = c.get("redrob_signals", {})
    career   = c.get("career_history", [])
    yoe      = profile.get("years_of_experience", 0)

    if yoe < 1:
        return False, "yoe_too_low"
    if is_consulting_only(career):
        return False, "consulting_only"
    last_act = days_since(sigs.get("last_active_date", "2000-01-01"))
    rr       = sigs.get("recruiter_response_rate", 0.0)
    if last_act > 180 and rr < 0.05:
        return False, "not_available"
    if is_wrong_domain(career_text):
        return False, "wrong_domain"
    return True, "ok"

# ─── main ───────────────────────────────────────────────────

def main():
    with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
        raw_candidates = [json.loads(line) for line in f if line.strip()]

    all_role_texts, role_index, career_texts = [], [], []
    for ci, c in enumerate(raw_candidates):
        career    = c.get("career_history", [])
        full_text = " ".join(build_role_text(h) for h in career).lower()
        career_texts.append(full_text)
        for ri, h in enumerate(career):
            all_role_texts.append(build_role_text(h))
            role_index.append((ci, ri))

    model = load_model()

    unique_texts, inverse = np.unique(np.array(all_role_texts, dtype=object), return_inverse=True)
    unique_embs = encode(model, list(unique_texts), show_progress=True)
    role_embs   = unique_embs[inverse]

    anc_l1  = encode(model, LEVEL1_ANCHORS)
    anc_l2a = encode(model, EMBEDDING_ANCHORS)
    anc_l2b = encode(model, VECTORDB_ANCHORS)
    anc_l2c = encode(model, EVAL_ANCHORS)
    anc_nth = [(name, encode(model, sents), bonus) for name, sents, bonus in NTH_SETS]
    del model

    sim_l1  = role_embs @ anc_l1.T
    sim_l2a = role_embs @ anc_l2a.T
    sim_l2b = role_embs @ anc_l2b.T
    sim_l2c = role_embs @ anc_l2c.T
    sim_nth = [(name, role_embs @ enc.T, bonus) for name, enc, bonus in anc_nth]

    cand_role_indices = {}
    for flat_idx, (ci, ri) in enumerate(role_index):
        cand_role_indices.setdefault(ci, []).append(flat_idx)

    features = []
    for ci, c in enumerate(raw_candidates):
        career_text    = career_texts[ci]
        passed, reason = passes_gate(c, career_text)
        if not passed:
            continue

        hp      = honeypot_check(c)
        profile = c.get("profile", {})
        skills  = c.get("skills", [])
        sigs    = c.get("redrob_signals", {})
        career  = c.get("career_history", [])
        yoe     = profile.get("years_of_experience", 0)

        idxs = cand_role_indices.get(ci, [])

        if idxs:
            rs_l1  = sim_l1[idxs]
            rs_l2a = sim_l2a[idxs]
            rs_l2b = sim_l2b[idxs]
            rs_l2c = sim_l2c[idxs]

            per_role_l1 = [top_k_mean(rs_l1[ri], k=min(3, rs_l1.shape[1])) for ri in range(len(idxs))]

            best_l1  = rs_l1.max(axis=0)
            best_l2a = rs_l2a.max(axis=0)
            best_l2b = rs_l2b.max(axis=0)
            best_l2c = rs_l2c.max(axis=0)

            score_l1  = top_k_mean(best_l1,  k=min(3, len(best_l1)))
            score_l2a = top_k_mean(best_l2a, k=min(3, len(best_l2a)))
            score_l2b = top_k_mean(best_l2b, k=min(3, len(best_l2b)))
            score_l2c = top_k_mean(best_l2c, k=min(3, len(best_l2c)))

            nth_scores = {}
            nth_bonus  = 0.0
            for name, sim_mat, bonus in sim_nth:
                best_nth = sim_mat[idxs].max(axis=0)
                s        = top_k_mean(best_nth, k=min(2, len(best_nth)))
                nth_scores[name] = float(s)
                if s > 0.62:
                    nth_bonus += bonus
            nth_bonus = min(nth_bonus, 0.05)

            recency_boost        = norm(per_role_l1[0]) * 0.15 if per_role_l1 else 0.0
            retrieval_role_count = sum(1 for s in per_role_l1 if norm(s) > RETRIEVAL_THRESHOLD)
            depth_bonus          = min(retrieval_role_count * 0.04, 0.12)

            n_l1_raw   = norm(score_l1)
            avg_level2 = (norm(score_l2a) + norm(score_l2b) + norm(score_l2c)) / 3.0
            framework_penalty = 0.85 if (n_l1_raw > 0.65 and avg_level2 < 0.35) else 1.0
        else:
            per_role_l1 = []
            score_l1 = score_l2a = score_l2b = score_l2c = 0.0
            nth_scores = {}
            nth_bonus = recency_boost = depth_bonus = 0.0
            retrieval_role_count = 0
            framework_penalty = 1.0

        n_l1  = norm(score_l1)
        n_l2a = norm(score_l2a)
        n_l2b = norm(score_l2b)
        n_l2c = norm(score_l2c)

        level2        = n_l2a * 0.33 + n_l2b * 0.37 + n_l2c * 0.30
        career_signal = min((0.60 * n_l1 + 0.40 * level2) + recency_boost + depth_bonus, 1.0)
        career_signal *= framework_penalty

        current_title_l = profile.get("current_title", "").lower()
        if any(ttl in current_title_l for ttl in CV_TITLES):
            career_signal *= 0.70

        gh_raw   = sigs.get("github_activity_score", -1)
        gh_score = 0.0 if gh_raw == -1 else gh_raw / 100.0

        retrieval_companies = [
            career[ri].get("company", "")
            for ri, s in enumerate(per_role_l1)
            if norm(s) > RETRIEVAL_THRESHOLD and ri < len(career)
        ]

        features.append({
            "candidate_id":          c["candidate_id"],
            "career_signal":         career_signal,
            "level1_raw":            float(score_l1),
            "level1_norm":           n_l1,
            "level2a_norm":          n_l2a,
            "level2b_norm":          n_l2b,
            "level2c_norm":          n_l2c,
            "recency_boost":         recency_boost,
            "depth_bonus":           depth_bonus,
            "retrieval_role_count":  retrieval_role_count,
            "retrieval_companies":   retrieval_companies,
            "nth_bonus":             nth_bonus,
            "nth_scores":            nth_scores,
            "python_bonus":          compute_python_bonus(skills),
            "product_score":         compute_product_score(career),
            "yoe_score":             compute_yoe_score(yoe),
            "location_score":        compute_location_score(profile, sigs),
            "github_score":          gh_score,
            "behavioral":            compute_behavioral(sigs),
            "hopping_penalty":       compute_hopping_penalty(career),
            "hp_penalty":            hp,
            "meta": {
                "title":        profile.get("current_title", ""),
                "company":      profile.get("current_company", ""),
                "yoe":          yoe,
                "location":     profile.get("location", ""),
                "country":      profile.get("country", "").lower(),
                "notice":       sigs.get("notice_period_days", 90),
                "open_to_work": sigs.get("open_to_work_flag", False),
                "last_active":  days_since(sigs.get("last_active_date", "2000-01-01")),
                "rr":           sigs.get("recruiter_response_rate", 0.0),
                "rt_h":         sigs.get("avg_response_time_hours", 999),
                "github":       gh_raw,
                "relocate":     sigs.get("willing_to_relocate", False),
                "work_mode":    sigs.get("preferred_work_mode", ""),
                "int_rate":     sigs.get("interview_completion_rate", 0.5),
                "saved_30d":    sigs.get("saved_by_recruiters_30d", 0),
                "companies":    [h.get("company", "") for h in career[:4]],
                "titles":       [h.get("title", "") for h in career[:4]],
                "durations":    [h.get("duration_months", 0) for h in career[:4]],
                "skills_top": [s.get("name", "") for s in skills[:10]],
                "retrieval_role_evidence": [
                    {
                        "company": career[ri].get("company", ""),
                        "title":   career[ri].get("title", ""),
                        "snippet": " ".join(career[ri].get("description", "").split()[:25])
                    }
                    for ri, s in enumerate(per_role_l1)
                    if norm(s) > RETRIEVAL_THRESHOLD and ri < len(career)
                ][:3],
            }
        })

    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(features, f, protocol=4)

    print(f"Done. Precompute finished — {len(features):,} candidates ready in {OUTPUT_PKL}")


if __name__ == "__main__":
    main()