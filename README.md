Resume Matching Engine — Redrob Hackathon

What this repo contains
- `resume_matcher.py`: Implementation of the resume matching engine following the problem sheet exactly (normalization, deduplication, vocabulary, TF-IDF, JD binary vectors, cosine similarity). Uses only Python standard library (`math`, `collections`).
- `Redrob_AI_Prompts.md`: Template of staged Redrob AI prompts to use while developing (for contest submission).
- `tests/test_matcher.py`: Unit tests validating normalization and TF-IDF/IDF math.
- `Redrob_Hackathon_Problem_Sheet.pdf`: Original problem sheet (copied from Downloads).

Usage

Run the matcher to print Top-3 candidates per JD:

```bash
/opt/homebrew/bin/python3 resume_matcher.py
```

Run tests:

```bash
/opt/homebrew/bin/python3 -m unittest tests.test_matcher -v
```

Notes on requirements
- The implementation follows formulas in the PDF: TF = 1 / N_unique (after dedup), IDF = ln(10/df), TF-IDF = TF*IDF, cosine similarity computed with Euclidean norms.
- Vocabulary is built from the 10 resumes only and sorted alphabetically.
- External ML libraries (numpy, scikit-learn, pandas) were not used.
- The repository includes a Redrob AI prompt template you can paste into Redrob for the required AI-usage summary.

Redrob-assisted workflow
- To involve Redrob AI in normalization (required by contest rules), run the script with `--redrob`. This writes staged prompt files to `redrob_prompts/`.
- Use Redrob AI with the prompts, then save normalized outputs to `redrob_responses/normalized.txt` — one line per resume (same order as in the PDF), comma-separated canonical skills (deduplicated).
- Re-run the script with `--redrob` (and `--responses-dir redrob_responses` if you used a different folder). The script will use those normalized skills and continue computing vocab, IDF, TF-IDF, and final rankings.

Example:

```bash
# generate prompts
/opt/homebrew/bin/python3 resume_matcher.py --redrob

# after saving responses to redrob_responses/normalized.txt, run:
/opt/homebrew/bin/python3 resume_matcher.py --redrob
```
