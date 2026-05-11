import math
import unittest

from resume_matcher import (
    normalize_skills,
    build_vocabulary,
    compute_tf_idf,
    RESUMES,
)


class TestMatcher(unittest.TestCase):
    def test_normalization_example(self):
        raw = "Pyhton, Deep-learning, pandas, matplotlib"
        norm = normalize_skills(raw)
        # order preserved from processing and deduplication
        self.assertIn("python", norm)
        self.assertIn("deep_learning", norm)
        self.assertIn("pandas", norm)
        # matplotlib maps to data_visualization in SKILL_ALIASES
        self.assertIn("data_visualization", norm)

    def test_idf_formula_and_tf_idf_values(self):
        # normalize all resumes
        resumes_norm = [normalize_skills(raw) for _, raw in RESUMES]

        vocab = build_vocabulary(resumes_norm)
        vectors, idf = compute_tf_idf(resumes_norm, vocab)

        N = len(resumes_norm)

        # verify idf follows ln(N/df) exactly
        for term in vocab:
            df = sum(1 for skills in resumes_norm if term in set(skills))
            if df == 0:
                expected_idf = 0.0
            else:
                expected_idf = math.log(N / df)
            self.assertAlmostEqual(idf[term], expected_idf, places=12)

        # verify TF-IDF vector entries follow TF * IDF, where TF = 1/unique_count after dedup
        for i, skills in enumerate(resumes_norm):
            unique_count = len(skills)
            for j, term in enumerate(vocab):
                tfidf_value = vectors[i][j]
                if term in skills:
                    expected_tf = 1.0 / unique_count if unique_count > 0 else 0.0
                    expected = expected_tf * idf[term]
                else:
                    expected = 0.0
                self.assertAlmostEqual(tfidf_value, expected, places=12)


if __name__ == "__main__":
    unittest.main()
