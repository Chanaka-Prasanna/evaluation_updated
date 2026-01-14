# Why These Benchmark Scores and Not Accuracy?

## Why NOT "Accuracy"?

**Accuracy is for classification tasks, not sequence generation tasks.**

### 1. Wrong Task Type

- **Accuracy measures**: "Did you pick the correct class from N options?"
- **Your task is**: "Generate a variable-length sequence of characters"
- This is a **sequence-to-sequence generation** task, not classification

### 2. No Single Prediction Point

- Accuracy works when there's ONE prediction per sample
- Your model generates 100-250 characters, each position has its own prediction
- Which position's "accuracy" would you report? Average across all positions is meaningless

### 3. Doesn't Capture Partial Correctness

- If the model generates `0,1\n1,*\n2,*` instead of `0,1\n1,*\n2,*\n`, it's 95% correct but accuracy would say 0%
- Accuracy is binary (right/wrong), but sequence generation has degrees of correctness

### 4. Padding Tokens Inflate Numbers

- If you calculated "accuracy" per token, the `<PAD>` tokens (which are ~60% of each sequence) would always be "correct"
- This would give you 70-80% "accuracy" even for terrible models

---

## Why These Benchmarks?

### 1. Exact Match Rate (0.XX)

- The closest thing to "accuracy" for sequences
- "What percentage of outputs are 100% correct?"
- Strict but interpretable for non-technical interviewers

### 2. Average Edit Distance (lower is better)

- "How many character insertions/deletions/substitutions to fix the output?"
- Captures partial correctness
- Standard metric for string comparison tasks

### 3. BLEU Score (0-1)

- Standard in machine translation
- Measures n-gram overlap between prediction and reference
- Shows if model captures local patterns correctly

### 4. ROUGE-L Score (0-1)

- Measures longest common subsequence
- Shows if model preserves structure/ordering

---

## Important

_"We cannot use accuracy because this is a sequence generation task, not classification. Accuracy measures 'did you pick the right class?' but we're generating 100-250 character sequences. Instead, we use:_

- _Exact Match Rate (similar to accuracy but for complete sequences)_
- _Edit Distance (measures how close we are to correct)_
- _BLEU and ROUGE-L (industry-standard metrics from machine translation)_

_These metrics are appropriate for evaluating sequence-to-sequence models like transformers in text generation tasks."_
