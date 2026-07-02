# RedRob Hackathon 2026
# Intelligent Candidate Discovery & Ranking

**Team:** Outcast

---
after installing requirements.txt [pip install -r requirements.txt]

we, have already provided the precomputed features in precomputed_features.pkl

So you can directly run the compute.py to generate the csv using  [ python compute.py]

else to replicate the entire process just execute , 
"python precompute.py
if ($LASTEXITCODE -eq 0) { python compute.py }"
---

# Overview

This repository contains our solution for the **RedRob AI Hiring Challenge**, where the objective is to identify and rank the **Top 100 candidates** from a dataset containing **100,000 candidate profiles** for a Retrieval, Search, Ranking and Applied Machine Learning engineering role.

Unlike traditional resume matching systems that rely primarily on keyword overlap, our solution attempts to approximate how an experienced technical recruiter evaluates candidates by combining semantic understanding, career trajectory, engineering context, behavioral hiring signals and recruiter-oriented reasoning.

The system is intentionally designed as a **two-stage pipeline**:

1. **Offline Feature Generation**
   - Parses the raw candidate dataset.
   - Extracts structured hiring signals.
   - Computes semantic similarity features.
   - Builds a reusable feature store.

2. **Deterministic Ranking**
   - Loads the precomputed feature store.
   - Applies a weighted scoring model.
   - Generates grounded recruiter-style reasoning.
   - Produces the final submission CSV.

Separating expensive feature extraction from ranking allows the scoring algorithm to remain lightweight, deterministic and easily reproducible while avoiding repeated semantic computation.

---

# Problem Statement

The challenge is significantly more complex than semantic search.

A strong hiring system must answer questions such as:

- Does the candidate possess production Retrieval/Search experience?
- Is the experience recent or outdated?
- Were those responsibilities sustained across multiple companies?
- Does the candidate come from product engineering or consulting?
- Is the candidate practically recruitable?
- Are there inconsistencies in the profile?
- Can the recruiter quickly understand *why* the candidate was ranked highly?

Our objective therefore became:

> Build a deterministic ranking engine that balances technical relevance, production experience, hiring practicality and profile quality while generating transparent recruiter-style explanations.

---

# Design Philosophy

The ranking pipeline follows five core principles.

## 1. Career matters more than keywords

Keyword matching alone cannot distinguish between:

- someone who briefly mentioned "RAG"
- someone who has spent six years building production retrieval systems

The solution therefore emphasizes **career evidence** rather than isolated keywords.

---

## 2. Production engineering matters

The target role is not purely research-oriented.

Candidates with sustained product engineering experience generally possess stronger deployment, ownership and operational exposure than equivalent consulting-heavy profiles.

Company history is therefore incorporated as a dedicated ranking signal.

---

## 3. Hiring signals should influence—not dominate—the ranking

Being easy to recruit does not automatically make someone technically stronger.

Behavioral metrics such as recruiter response rate, notice period and profile activity are therefore applied as bounded adjustments rather than replacing technical merit.

---

## 4. Penalize suspicious profiles

Even technically relevant profiles should lose confidence if they exhibit:

- inconsistent timelines
- suspicious career history
- excessive short-tenure roles
- profile anomalies

Rather than discarding candidates entirely, confidence is reduced using multiplicative penalties.

---

## 5. Every ranking should be explainable

Every shortlisted candidate receives recruiter-style reasoning constructed directly from measurable profile evidence.

The reasoning engine never relies on random generation.

Each explanation is deterministic and reproducible.

---

# System Architecture

```

                           candidates.jsonl
                                  │
                                  ▼
                    ┌────────────────────────┐
                    │    precompute.py       │
                    │------------------------│
                    │ JSON Parsing           │
                    │ Career Features        │
                    │ Product Features       │
                    │ Behavioral Features    │
                    │ Semantic Retrieval     │
                    │ Bonus Detection        │
                    │ Penalty Signals        │
                    └──────────┬─────────────┘
                               │
                               ▼
                 precomputed_features.pkl
                               │
                               ▼
                    ┌────────────────────────┐
                    │      compute.py        │
                    │------------------------│
                    │ Weighted Scoring       │
                    │ Behavioral Multiplier  │
                    │ Penalty Layer          │
                    │ Candidate Ranking      │
                    │ Recruiter Reasoning    │
                    └──────────┬─────────────┘
                               │
                               ▼
                        submission.csv

```

---

# Repository Structure

```
.
├── candidates.jsonl
├── sample_candidates.json
├── models/
│   └── bge-base-en-v1.5/
├── precompute.py
├── compute.py
├── precomputed_features.pkl
├── requirements.txt
└── README.md
```

---

# Why Two Separate Stages?

Semantic embedding generation is computationally expensive but independent of the ranking strategy.

Running semantic encoding every time scoring weights change would dramatically increase iteration time.

Instead, the pipeline is divided into two phases:

## Stage 1 — Feature Generation

Runs once.

Produces a reusable feature store containing every derived feature required by the ranking engine.

This includes semantic similarity features, structured hiring signals and penalty indicators.

Output:

```
precomputed_features.pkl
```

---

## Stage 2 — Candidate Ranking

Loads the feature store and performs only lightweight computation.

This stage:

- computes final scores
- applies behavioral adjustments
- applies penalties
- ranks candidates
- generates reasoning
- writes the submission CSV

Because all expensive computation has already been completed, ranking remains deterministic, lightweight and reproducible.

---

# End-to-End Pipeline

```

Candidate Profiles
        │
        ▼
JSON Parsing
        │
        ▼
Feature Engineering
        │
        ▼
Semantic Retrieval
        │
        ▼
Structured Signals
        │
        ▼
Weighted Scoring
        │
        ▼
Behavioral Adjustment
        │
        ▼
Penalty Layer
        │
        ▼
Ranking
        │
        ▼
Recruiter Reasoning
        │
        ▼
submission.csv

```

---

# High-Level Scoring Flow

The ranking engine evaluates every candidate through multiple independent signals rather than relying on a single similarity score.

These signals capture:

- technical specialization
- production engineering exposure
- hiring practicality
- profile quality
- recruiter confidence

Each signal contributes differently to the final score depending on its predictive value for the target Retrieval/Search role.

The following sections describe each component individually.

---------------------------------------------------

# Feature Engineering

Every candidate is transformed from an unstructured profile into a structured feature vector before ranking.

Rather than relying on a single semantic similarity score, the pipeline computes multiple independent hiring signals that collectively represent both technical relevance and recruitability.

The major feature groups are:

```
Career Evidence
        │
        ▼
Product Engineering
        │
        ▼
Years of Experience
        │
        ▼
Location Compatibility
        │
        ▼
GitHub Activity
        │
        ▼
Behavioral Signals
        │
        ▼
Domain Specialization
        │
        ▼
Penalty Signals
```

These signals are combined during the ranking stage.

---

# Career Signal (35%)

The Career Signal is the single most important feature in the ranking pipeline.

Its purpose is to answer one question:

> **How consistently has the candidate worked on Retrieval, Search, Ranking or Recommendation Systems throughout their career?**

Rather than searching for isolated keywords, the system evaluates sustained career evidence.

The score incorporates multiple observations extracted from a candidate's employment history, including:

- Semantic similarity between past work and the target role
- Retrieval-specific responsibilities
- Search and recommendation system exposure
- Engineering depth across multiple positions
- Career progression
- Consistency of specialization

Candidates demonstrating repeated production retrieval work naturally receive higher scores than candidates mentioning the same technologies only once.

Because long-term career specialization is the strongest predictor of success for this role, Career Signal receives the highest weight in the overall ranking model.

---

# Product Experience (25%)

Technical skills alone do not fully represent engineering maturity.

Candidates who have spent significant portions of their careers inside product organizations often gain stronger experience with:

- production deployments
- ownership of customer-facing systems
- iterative product development
- scalability
- operational excellence

The Product Score rewards candidates whose company history reflects sustained product engineering experience while slightly reducing confidence for consulting-heavy career trajectories.

This feature complements Career Signal by evaluating *where* the candidate gained experience rather than *what* they worked on.

---

# Years of Experience (15%)

Years of Experience is treated as a guidance signal rather than a dominant ranking factor.

The target role expects experienced engineers while avoiding excessive weighting toward seniority alone.

The score therefore rewards candidates whose experience falls within the desired hiring range while gradually reducing confidence outside that range.

This prevents:

- junior candidates from being over-ranked because of semantic similarity
- very senior candidates from dominating solely because of tenure

---

# Location Compatibility (7%)

Recruitability depends partly on geographical compatibility.

The Location Score considers:

- country
- preferred hiring region
- relocation willingness
- preferred work mode

Candidates already located in preferred hiring regions receive slightly higher scores, while candidates requiring relocation or preferring incompatible work arrangements receive proportionally lower confidence.

Location contributes to the final score without overwhelming technical relevance.

---

# GitHub Activity (8%)

Public engineering work provides additional evidence of practical software engineering capability.

GitHub is therefore used as a supporting confidence signal.

The score does **not** attempt to judge project quality directly.

Instead it rewards candidates who demonstrate sustained public technical activity.

GitHub contributes only a modest portion of the overall score to avoid disadvantaging experienced engineers who primarily work in private repositories.

---

# Behavioral Signals

Technical relevance alone does not determine hiring success.

Behavioral signals estimate how practical a candidate may be to recruit.

The pipeline incorporates signals including:

- Open-to-work status
- Recruiter response rate
- Recent profile activity
- Notice period
- Interview completion rate

Unlike technical features, behavioral metrics are **not added directly** into the score.

Instead they act as a bounded multiplier.

This design ensures:

- technically excellent candidates remain highly ranked
- highly active but technically weaker candidates cannot dominate the ranking

The behavioral multiplier therefore adjusts confidence rather than replacing technical evaluation.

---

# Specialized Technical Bonuses

The ranking engine rewards candidates demonstrating evidence of domain-specific expertise beyond general machine learning.

Specialization bonuses include areas such as:

- Learning-to-Rank
- Retrieval Engineering
- LLM Fine-tuning
- HR Technology
- Distributed Machine Learning
- Open Source Engineering
- Python Expertise

These bonuses provide additional separation between candidates with similar overall experience while remaining intentionally small relative to the primary scoring components.

---

# Penalty Layer

High semantic similarity alone does not guarantee profile quality.

The ranking engine therefore applies a dedicated penalty layer designed to reduce confidence in suspicious profiles.

Penalty sources include:

- profile inconsistencies
- suspicious career histories
- excessive short-tenure roles
- anomalous employment timelines

Unlike hard filtering, penalties are multiplicative.

This allows technically strong candidates to remain competitive while appropriately lowering confidence when profile quality decreases.

---

# Semantic Retrieval Engine

Semantic understanding forms the foundation of the ranking pipeline.

Rather than relying on keyword overlap, candidate work histories are embedded using a local Sentence Transformer model.

The embedding stage enables the system to recognize semantically related concepts such as:

- Search
- Retrieval
- Recommendation Systems
- Ranking
- Vector Databases
- Semantic Search
- Learning-to-Rank

even when exact wording differs between resumes.

Embedding generation occurs only during the offline preprocessing stage.

The ranking stage itself never performs semantic encoding, allowing deterministic and lightweight execution.

---

# Scoring Methodology

Each candidate receives a weighted base score computed from multiple independent feature groups.

The current weighting strategy is:

| Feature | Weight |
|----------|-------:|
| Career Signal | **35%** |
| Product Experience | **25%** |
| Years of Experience | **15%** |
| Location | **7%** |
| GitHub | **8%** |
| Python Bonus | up to **5%** |
| Domain Bonus | up to **5%** |

The weighted score is subsequently adjusted using:

- Behavioral multiplier
- Job hopping penalty
- Profile quality penalty

This layered approach allows strong technical candidates to remain competitive while appropriately accounting for practical hiring considerations.

---

# Why These Weights?

The weighting strategy reflects the hiring priorities for a Retrieval/Search engineering role.

**Career Signal (35%)**

Career history provides the strongest evidence of sustained retrieval expertise and therefore receives the highest weight.

---

**Product Experience (25%)**

Engineering context matters.

Candidates with significant product engineering exposure generally demonstrate stronger production ownership than equally skilled consulting profiles.

---

**Years of Experience (15%)**

Experience is valuable but should not dominate ranking.

This weight rewards maturity while preventing seniority alone from driving results.

---

**GitHub (8%)**

Public engineering work provides additional confidence but remains a supporting signal.

---

**Location (7%)**

Hiring practicality is important but secondary to technical ability.

---

**Specialization Bonuses**

These bonuses reward candidates who demonstrate expertise directly aligned with the target role while ensuring they cannot outweigh sustained career evidence.


----------------------------------------------------------------

# Recruiter Reasoning Engine

Ranking alone is often insufficient for hiring decisions.

Recruiters require a concise explanation describing **why** a candidate was ranked highly and what considerations remain before moving forward.

To support this, the ranking engine generates deterministic recruiter-style reasoning for every shortlisted candidate.

Unlike generative approaches, the reasoning engine does **not** synthesize new information.

Instead, each explanation is constructed directly from structured candidate evidence.

The reasoning process follows four stages:

```

Final Score
│
▼
Candidate Tier
│
▼
Template Selection
│
▼
Evidence Extraction
│
▼
Recruiter Note

```

The resulting explanations reference measurable profile attributes including:

- Current job title
- Years of experience
- Previous companies
- Retrieval/Search experience
- Behavioral observations
- Hiring considerations

Every statement in the generated reasoning is grounded in existing profile data.

---

# Deterministic Template Selection

One challenge when generating explanations is avoiding repetitive outputs.

Using random template selection would improve variety but reduce reproducibility.

Instead, template selection is deterministic.

Each candidate's unique identifier is hashed and mapped to one of several recruiter-style narrative paths.

This guarantees:

- identical output across runs
- reproducibility
- structural variation
- consistent wording

without introducing randomness.

---

# Ranking Pipeline

The complete ranking pipeline consists of the following stages:

```

Raw Candidate Profile
│
▼
JSON Parsing
│
▼
Feature Engineering
│
▼
Semantic Retrieval
│
▼
Structured Hiring Signals
│
▼
Weighted Base Score
│
▼
Behavioral Multiplier
│
▼
Penalty Layer
│
▼
Candidate Ranking
│
▼
Recruiter Reasoning
│
▼
submission.csv

```

Each stage contributes independently to the final ranking.

---

# Runtime Characteristics

The solution is intentionally divided into an offline feature generation stage and a lightweight ranking stage.

## Stage 1 – Feature Generation

Responsibilities:

- Parse candidate profiles
- Generate semantic embeddings
- Extract structured hiring signals
- Compute retrieval-specific features
- Serialize reusable feature store

Output:

```

precomputed_features.pkl

```

This stage is executed only once unless the source dataset changes.

---

## Stage 2 – Ranking

Responsibilities:

- Load precomputed features
- Compute weighted scores
- Apply behavioral adjustment
- Apply penalties
- Sort candidates
- Generate recruiter reasoning
- Produce submission CSV

Output:

```

submission.csv

```

Since semantic computation has already been completed, the ranking stage performs only lightweight numerical operations.

---

# Complexity Analysis

## Feature Generation

The preprocessing stage performs semantic encoding and feature extraction over the candidate dataset.

Dominant operations include:

- profile parsing
- embedding generation
- structured feature extraction

This stage is executed once.

---

## Ranking

For each candidate the ranking stage performs:

- weighted score calculation
- behavioral adjustment
- penalty application
- deterministic reasoning generation

Finally candidates are sorted by score.

Overall ranking complexity is dominated by sorting.

```

Feature Generation : Offline

Ranking : O(N log N)

Memory : O(N)

```

where **N** is the number of candidate profiles.

---

# Reproducibility

A primary design goal of this project is deterministic execution.

The following properties guarantee reproducibility:

- No stochastic ranking
- No random weight initialization
- No randomized template selection
- Stable sorting
- Deterministic reasoning generation
- Fixed scoring formula

Given identical inputs, the system always produces identical outputs.

---

# Installation

Clone the repository:

```bash
git clone <repository_url>
cd RedRob_Hackathon_Outcast
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Python 3.10 or newer is recommended.

---

# Running the Pipeline

## Step 1 – Feature Generation

```bash
python precompute.py
```

Produces:

```

precomputed_features.pkl

```

---

## Step 2 – Candidate Ranking

```bash
python compute.py
```

Produces:

```

submission.csv

```

---

# Output Format

The generated CSV follows the competition specification.

| Column | Description |
|---------|-------------|
| candidate_id | Candidate identifier |
| rank | Final ranking position |
| score | Deterministic ranking score |
| reasoning | Recruiter-style explanation |

The top 100 candidates are written to the output file.

---

# Example Candidate Evaluation

A simplified scoring flow is illustrated below.

```

Career Signal ............. 0.91

Product Experience ........ 0.84

Years of Experience ....... 1.00

Location .................. 0.95

GitHub .................... 0.78

Python Bonus .............. +0.05

Domain Bonus .............. +0.04

↓

Behavioral Adjustment

↓

Profile Quality Penalty

↓

Job Hopping Penalty

↓

Final Score

↓

Recruiter Reasoning

```

This layered evaluation enables the ranking engine to balance technical relevance, production experience and hiring practicality without relying on any single feature.

---

# Engineering Principles

The implementation follows several engineering principles throughout the pipeline:

- Deterministic execution
- Explainable ranking
- Modular feature extraction
- Separation of preprocessing and ranking
- CPU-only execution
- Reproducible outputs
- Transparent scoring
- Grounded recruiter reasoning

These principles ensure the system remains interpretable, maintainable and suitable for large-scale candidate ranking.

------------------------------------------------------

# Design Decisions

Several architectural decisions were made to balance ranking quality, reproducibility and computational efficiency.

---

## Why Separate Feature Generation and Ranking?

Generating semantic embeddings is significantly more expensive than computing ranking scores.

Rather than recomputing embeddings whenever the ranking strategy changes, the solution separates the pipeline into two independent stages.

This provides several advantages:

- Faster experimentation with ranking weights
- Reduced execution time during ranking
- Fully deterministic outputs
- Easier debugging
- Better scalability for larger candidate datasets

Only the feature generation stage performs semantic encoding, while the ranking stage operates entirely on structured features.

---

## Why Weighted Scoring Instead of Pure Semantic Similarity?

Semantic similarity is excellent for identifying technically relevant candidates but cannot capture many factors that influence hiring decisions.

For example, two candidates may have similar semantic similarity scores while differing substantially in:

- production engineering experience
- years of experience
- hiring availability
- company trajectory
- profile quality

A weighted scoring framework allows multiple independent hiring signals to contribute toward the final ranking.

---

## Why Behavioural Signals Are Multipliers

Behavioral signals estimate how recruitable a candidate is but should never outweigh technical ability.

Adding behavioral features directly into the base score risks over-rewarding highly active candidates while penalizing technically stronger engineers.

Applying behavioral signals as a bounded multiplier preserves technical ranking while adjusting confidence based on practical hiring considerations.

---

## Why Penalties Instead of Hard Filtering?

Candidate profiles often contain minor inconsistencies that should reduce confidence without completely excluding otherwise strong engineers.

Using multiplicative penalties instead of binary filters allows technically relevant candidates to remain competitive while appropriately accounting for profile quality.

---

## Why Deterministic Recruiter Reasoning?

The challenge requires reasoning that is:

- understandable
- reproducible
- grounded in profile evidence

Using deterministic template selection guarantees identical outputs across runs while avoiding repetitive explanations.

Every generated explanation references measurable candidate attributes rather than producing unsupported claims.

---

# Advantages of the Proposed Approach

Compared with ranking systems based solely on keyword search or semantic similarity, this solution offers several advantages.

## Multi-dimensional Evaluation

The ranking engine evaluates candidates using multiple independent signals rather than relying on a single embedding similarity score.

---

## Explainability

Every ranking decision can be traced back to measurable profile evidence.

This improves transparency and allows recruiters to quickly understand why a candidate appears in the shortlist.

---

## Reproducibility

No random operations influence the ranking.

Running the pipeline multiple times on the same dataset always produces identical outputs.

---

## Modularity

Feature generation and ranking are completely independent.

Future improvements can modify the scoring model without repeating expensive preprocessing.

---

## Scalability

The architecture naturally scales to significantly larger candidate collections because semantic computation occurs only once.

---

# Current Limitations

While the system performs well for the target Retrieval/Search engineering role, several opportunities remain for future improvement.

Current limitations include:

- Domain-specific weighting is optimized for Retrieval/Search hiring rather than arbitrary engineering roles.
- Company quality is represented through engineered signals rather than external market intelligence.
- GitHub scoring measures engineering activity but does not evaluate repository quality.
- The reasoning engine uses deterministic templates rather than natural language generation.

These trade-offs were intentionally made to maximize determinism and reproducibility.

---

# Future Improvements

Several enhancements could further improve ranking quality.

## Adaptive Weight Learning

Replace manually selected feature weights with learning-to-rank techniques trained on historical hiring outcomes.

---

## Dynamic Job Descriptions

Automatically regenerate semantic anchors for arbitrary job descriptions rather than targeting a single retrieval-focused role.

---

## Richer Behavioral Signals

Incorporate additional recruiter interaction metrics such as response latency trends or historical hiring conversion.

---

## Company Knowledge Graph

Introduce company relationships, industry similarity and organization-level embeddings to improve contextual understanding.

---

## Explainable Feature Attribution

Provide recruiters with feature-level contribution scores alongside each recommendation.

---

# Conclusion

This project presents a deterministic candidate ranking pipeline designed specifically for Retrieval, Search and Applied Machine Learning hiring.

Rather than relying on semantic similarity alone, the proposed system combines:

- semantic career understanding
- structured hiring signals
- production engineering experience
- behavioral hiring indicators
- profile quality assessment
- deterministic recruiter reasoning

into a unified ranking framework.

The resulting system is explainable, reproducible, computationally efficient and aligned with practical recruiter workflows while remaining fully deterministic.

---

# Authors

**Team Outcast**
-Ainesh Behera    -   www.linkedin.com/in/ainesh-behera
-Riya Mehta       -   www.linkedin.com/in/riya-mehta-rz19

RedRob Hackathon 2026