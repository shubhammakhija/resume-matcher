#!/usr/bin/env python3
"""
Resume Matching Engine

Implements the workflow from Redrob_Hackathon_Problem_Sheet.pdf without external libraries.
"""
from math import log, sqrt
from collections import defaultdict
import argparse
import os
from pathlib import Path

# SKILL_ALIASES mapping (exactly as provided in the PDF)
SKILL_ALIASES = {
    "python": "python",
    "pyhton": "python",
    "java": "java",
    "javascript": "javascript",
    "javascrpit": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "typescrpit": "typescript",
    "c++": "cpp",
    "cpp": "cpp",
    "r": "r",
    "kotlin": "kotlin",
    "machinelearning": "machine_learning",
    "machine learning": "machine_learning",
    "ml": "machine_learning",
    "sklearn": "machine_learning",
    "deeplearning": "deep_learning",
    "deep learning": "deep_learning",
    "deep-learning": "deep_learning",
    "tensorflow": "tensorflow",
    "pytorch": "pytorch",
    "keras": "keras",
    "nlp": "nlp",
    "bert": "bert",
    "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "feature_engineering": "feature_engineering",
    "statistics": "statistics",
    "stats": "statistics",
    "regression": "regression",
    "clustering": "clustering",
    "data-viz": "data_visualization",
    "data visualization": "data_visualization",
    "data viz": "data_visualization",
    "matplotlib": "data_visualization",
    "tableau": "data_visualization",
    "power-bi": "data_visualization",
    "power bi": "data_visualization",
    "powerbi": "data_visualization",
    "pandas": "pandas",
    "numpy": "numpy",
    "react": "react",
    "reacts": "react",
    "reactjs": "react",
    "vue": "vue",
    "vue.js": "vue",
    "vuejs": "vue",
    "redux": "redux",
    "tailwind": "tailwind",
    "html/css": "html_css",
    "html css": "html_css",
    "html": "html_css",
    "css": "html_css",
    "jest": "jest",
    "graphql": "graphql",
    "node.js": "nodejs",
    "nodejs": "nodejs",
    "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot",
    "springboot": "spring_boot",
    "rest api": "rest_api",
    "rest": "rest_api",
    "restapi": "rest_api",
    "microservices": "microservices",
    "sql": "sql",
    "mysql": "mysql",
    "mysq": "mysql",
    "postgresql": "postgresql",
    "postgres": "postgresql",
    "mongodb": "mongodb",
    "redis": "redis",
    "docker": "docker",
    "kubernetes": "kubernetes",
    "kubernates": "kubernetes",
    "k8s": "kubernetes",
    "ci/cd": "ci_cd",
    "cicd": "ci_cd",
    "ci cd": "ci_cd",
    "aws": "aws",
    "android": "android",
    "firebase": "firebase",
    "algorithms": "algorithms",
    "algoritms": "algorithms",
    "data structure": "data_structures",
    "data structures": "data_structures",
    "competitive programming": "competitive_programming",
    "ui/ux": "ui_ux",
    "ui ux": "ui_ux",
    "figma": "figma",
}


# Resume dataset (from the PDF)
RESUMES = [
    ("Arjun Sharma", "Pyhton, MachineLearning, SQL, pandas, nump y, Deep-learning"),
    ("Priya Nair", "JavaScrpit, Reacts, Node. JS, MongoDb, REST api, HTML/CSS"),
    ("Rahul Gupta", "Java, Spring Boot, MySql, Microser vices, Dock er, kubernat es"),
    ("Sneha Patel", "Python, TensorFlo w, Keras, NLP, BERT, data-viz, matplotlib"),
    ("Vikram Singh", "C++, Algoritms, Data Structure, compe titive programming, python"),
    ("Ananya Krishnan", "javascrip t, vue.js, python, flask, PostgreSQL , AWS, CI/CD"),
    ("Karan Mehta", "Python, Sklearn, XGboos t, feature engineering, SQL, tableau"),
    ("Deepika Rao", "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"),
    ("Aditya Kumar", "Reactjs, TypeScrpit, GraphQL , redux, tailwind, nodejs, jest"),
    ("Meera Iyer", "python, R, statistics, ML, regression, clustering, Power-BI"),
]

# Job descriptions (required skills only)
JDS = {
    "JD-1": ("Kakao (ML Engineer)", "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, Data Visualization"),
    "JD-2": ("Naver (Backend Engineer)", "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, Kubernetes"),
    "JD-3": ("Line (Frontend Engineer)", "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS"),
}


def normalize_skills(raw_skills):
    """Normalize a raw skills string into a deduplicated list of canonical skills."""
    parts = [p.strip() for p in raw_skills.split(",") if p.strip()]
    canonical = []
    seen = set()
    for part in parts:
        token = part.lower().strip()
        # collapse multiple spaces
        token = " ".join(token.split())

        # generate candidate variants to handle noisy tokens like 'nump y', 'JavaScrpit', 'Node. JS'
        variants = []
        variants.append(token)
        variants.append(token.replace(' ', ''))
        variants.append(token.replace(' ', '_'))
        variants.append(token.replace('-', ''))
        variants.append(token.replace('.', ''))
        variants.append(token.replace(".", '').replace(' ', ''))
        # keep slashes and plus where meaningful, but also try removing punctuation except + and /
        sanitized = ''.join(ch for ch in token if ch.isalnum() or ch in '+/_ ')
        variants.append(sanitized)
        variants.append(sanitized.replace(' ', ''))
        variants.append(sanitized.replace(' ', '_'))

        # also try removing stray spaces inside words (e.g., 'nump y' -> 'numpy')
        variants.append(''.join(token.split()))

        found = None
        for c in variants:
            if c in SKILL_ALIASES:
                found = SKILL_ALIASES[c]
                break
        # finally, try the original token as-is if present in mapping
        if not found and token in SKILL_ALIASES:
            found = SKILL_ALIASES[token]

        if found and found not in seen:
            canonical.append(found)
            seen.add(found)

    return canonical


def build_vocabulary(resumes_normalized):
    vocab = set()
    for skills in resumes_normalized:
        for s in skills:
            vocab.add(s)
    vocab_list = sorted(vocab)
    return vocab_list


def compute_tf_idf(resumes_normalized, vocab):
    # compute df for each vocab term
    df = defaultdict(int)
    for skills in resumes_normalized:
        sset = set(skills)
        for v in vocab:
            if v in sset:
                df[v] += 1

    idf = {}
    N = len(resumes_normalized)
    for v in vocab:
        if df[v] == 0:
            idf[v] = 0.0
        else:
            idf[v] = log(N / df[v])

    vectors = []
    for skills in resumes_normalized:
        N_unique = len(skills)
        vec = []
        for v in vocab:
            tf = 0.0
            if v in skills:
                if N_unique > 0:
                    tf = 1.0 / N_unique
            vec.append(tf * idf[v])
        vectors.append(vec)

    return vectors, idf


def jd_to_binary_vector(jd_skills_raw, vocab):
    norm = normalize_skills(jd_skills_raw)
    sset = set(norm)
    vec = [1 if v in sset else 0 for v in vocab]
    return vec


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def write_redrob_prompts(out_dir: Path):
    """Write staged prompts for Redrob into out_dir."""
    # Stage 1: list raw tokens
    raw_skills_list = [raw for _, raw in RESUMES]
    with open(out_dir / 'stage1_list_tokens.txt', 'w', encoding='utf-8') as f:
        f.write('Stage 1 - List raw tokens (split on commas) from these resumes in order (one resume per line):\n\n')
        for raw in raw_skills_list:
            f.write(raw + '\n')

    # Stage 2: propose normalization logic
    with open(out_dir / 'stage2_propose_normalization.txt', 'w', encoding='utf-8') as f:
        f.write('Stage 2 - Propose a robust normalization procedure for the raw skill tokens from Stage 1. Consider: handling internal spaces (e.g., "nump y" -> "numpy"), removing punctuation, lowercasing, multi-word phrase matching, and deduplication. List edge cases and candidate variants to try. Do NOT apply SKILL_ALIASES yet — just propose the logic.\n')

    # Stage 3: apply SKILL_ALIASES mapping and return normalized skills per resume
    with open(out_dir / 'stage3_normalize.txt', 'w', encoding='utf-8') as f:
        f.write('Stage 3 - Normalize the following raw skills using the exact SKILL_ALIASES mapping. For each resume (in the same order), return a comma-separated list of canonical skills and discard unknown tokens. Match multi-word phrases before token-level matches.\n\n')
        for raw in raw_skills_list:
            f.write(raw + '\n')

    # Stage 4: vocabulary & idf
    with open(out_dir / 'stage4_vocab_idf.txt', 'w', encoding='utf-8') as f:
        f.write('Stage 4 - From normalized, deduplicated resume skills (which you should produce), return an alphabetical vocabulary, df for each term, and IDF = ln(10/df) with at least 6 decimals.\n')

    # Stage 5/6 prompts
    with open(out_dir / 'stage5_tfidf_and_similarity.txt', 'w', encoding='utf-8') as f:
        f.write('Stage 5/6 - Using the vocabulary from Stage 4 and normalized resume skills, compute TF (1/unique_count), TF-IDF vectors, JD binary vectors (use the provided JD required skills), and cosine similarities. Return top-3 per JD with scores rounded to 2 decimals.\n')


def read_redrob_normalized(path: Path):
    """Read normalized skills file produced by Redrob.

    Expected format: one line per resume in the same order as RESUMES. Each line: comma-separated canonical skills (already deduplicated).
    """
    lines = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(lines) != len(RESUMES):
        raise ValueError(f'Expected {len(RESUMES)} lines in {path}, found {len(lines)}')
    resumes_norm = []
    for line in lines:
        tokens = [t.strip() for t in line.split(',') if t.strip()]
        resumes_norm.append(tokens)
    return resumes_norm


def main():
    parser = argparse.ArgumentParser(description="Resume matcher. Use --redrob to enable Redrob-assisted workflow")
    parser.add_argument('--redrob', action='store_true', help='Generate Redrob prompts and use Redrob responses if present')
    parser.add_argument('--responses-dir', default='redrob_responses', help='Directory containing Redrob response files')
    args = parser.parse_args()

    # normalize resumes (local by default)
    if args.redrob:
        prompts_dir = Path('redrob_prompts')
        prompts_dir.mkdir(exist_ok=True)
        write_redrob_prompts(prompts_dir)
        responses_dir = Path(args.responses_dir)
        normalized_file = responses_dir / 'normalized.txt'
        if normalized_file.exists():
            resumes_norm = read_redrob_normalized(normalized_file)
        else:
            print(f"Redrob prompts written to {prompts_dir}.\nPlease run Redrob with these prompts and save normalized results to {normalized_file} (one line per resume, comma-separated canonical skills).")
            return
    else:
        resumes_norm = [normalize_skills(raw) for _, raw in RESUMES]

    vocab = build_vocabulary(resumes_norm)

    tfidf_vectors, idf = compute_tf_idf(resumes_norm, vocab)

    results = {}

    for jd_key, (jd_title, jd_raw) in JDS.items():
        jd_vec = jd_to_binary_vector(jd_raw, vocab)
        scores = []
        for (name, _), vec in zip(RESUMES, tfidf_vectors):
            score = cosine_similarity(vec, jd_vec)
            scores.append((name, score))

        # sort by rounded score desc, then name asc for ties
        scores_sorted = sorted(scores, key=lambda x: (-round(x[1], 2), x[0]))
        top3 = scores_sorted[:3]
        results[jd_key] = (jd_title, top3)

    # print outputs in expected format
    for jd_key in ["JD-1", "JD-2", "JD-3"]:
        jd_title, top3 = results[jd_key]
        line = ", ".join(f"{name}({format(round(score,2), '.2f')})" for name, score in top3)
        print(f"{jd_key} — {jd_title}")
        print(line)
        print()


if __name__ == "__main__":
    main()


def read_redrob_normalized(path: Path):
    """Read normalized skills file produced by Redrob.

    Expected format: one line per resume in the same order as RESUMES. Each line: comma-separated canonical skills (already deduplicated).
    """
    lines = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(lines) != len(RESUMES):
        raise ValueError(f'Expected {len(RESUMES)} lines in {path}, found {len(lines)}')
    resumes_norm = []
    for line in lines:
        tokens = [t.strip() for t in line.split(',') if t.strip()]
        resumes_norm.append(tokens)
    return resumes_norm
