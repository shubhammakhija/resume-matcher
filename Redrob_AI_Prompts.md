Redrob AI Prompts — staged template

Use the following staged prompts with Redrob AI when preparing your submission. Record the Redrob AI responses and include them in the submission as the "Redrob AI Usage Summary".

Stage 1 — Explore noisy tokens
- Prompt: "I have these raw skill strings from 10 resumes: <paste raw skills>. List all unique raw tokens (split on commas) and show frequency counts. Don't normalize — only split and lowercase."

Stage 2 — Propose normalization logic
- Prompt: "Using the provided SKILL_ALIASES mapping (I will paste it), propose a robust normalization procedure to map noisy tokens to canonical skills. List edge cases and suggested candidate variants (remove internal spaces, remove punctuation, try underscores)." 

Stage 3 — Apply SKILL_ALIASES mapping
- Prompt: "Normalize the following raw skills using this exact SKILL_ALIASES mapping (paste mapping). For each resume, show normalized tokens and discard tokens not in the mapping. Match multi-word phrases before token-level matches." 

Stage 4 — Vocabulary & IDF checks
- Prompt: "From the normalized, deduplicated resume skills, create an alphabetical vocabulary. For each term compute df (number of resumes containing the term) and IDF = ln(10/df). Show values with at least 6 decimal places."

Stage 5 — TF-IDF & verification
- Prompt: "Compute TF for each resume term as 1 / number_of_unique_skills_in_resume (after dedup). Multiply TF by IDF to get TF-IDF. Show TF-IDF vectors for each resume (vocabulary order must match the vocabulary from Stage 4)." 

Stage 6 — JD binary vectors & similarity
- Prompt: "Given the 3 JDs (paste the JD required skills), produce binary vectors over the vocabulary. Compute cosine similarity between each resume TF-IDF vector and each JD binary vector. Rank top 3 candidates per JD and provide rounded scores to 2 decimals. Break ties alphabetically." 

Stage 7 — Final output for submission
- Prompt: "Provide the final submission output in the exact format required: JD-1 — <title> followed by three candidate entries 'Name(score), Name(score), Name(score)'. Also produce a concise bullet list of the Redrob prompts used (copy of the inputs) and any intermediate verification outputs (vocabulary, idf table)."

Remember: in the contest rules, you must not modify the `SKILL_ALIASES` mapping. Use the mapping exactly as provided.
