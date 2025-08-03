from collections import Counter
import math
from itertools import chain

# 1) Exact‑match rate ----------------------------------------------------------
def exact_match_rate(predictions, references):
    """
    Fraction of samples whose prediction string is identical to the reference.
    """
    assert len(predictions) == len(references)
    correct = sum(p == r for p, r in zip(predictions, references))
    return correct / len(predictions)



# 2) Character‑level edit (Levenshtein) distance -------------------------------
def edit_distance(a, b):
    """
    Classic O(len(a) * len(b)) dynamic‑programming algorithm.
    Returns the minimum number of single‑character edits
    (insertions, deletions, substitutions) to convert a into b.
    """
    m, n = len(a), len(b)
    dp = list(range(n + 1))               # previous row
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            insert = dp[j - 1] + 1
            delete = dp[j] + 1
            substitute = prev + (a[i - 1] != b[j - 1])
            prev, dp[j] = dp[j], min(insert, delete, substitute)
    return dp[-1]

def average_edit_distance(predictions, references):
    """
    Mean character‑level edit distance across a corpus.
    """
    assert len(predictions) == len(references)
    total = sum(edit_distance(p, r) for p, r in zip(predictions, references))
    return total / len(predictions)


# 3) BLEU (up to 4‑gram, single reference, corpus‑level) -----------------------
def bleu(predictions, references, max_n=4, smooth=1):
    """
    Simplified BLEU implementation with add‑k smoothing (k = smooth).
    predictions and references are lists of token lists.
    """
    assert len(predictions) == len(references)
    
    # Precision for each n‑gram length
    p_ns = []
    for n in range(1, max_n + 1):
        match, total = 0, 0
        for pred, ref in zip(predictions, references):
            pred_ngrams = Counter(tuple(pred[i:i+n]) for i in range(len(pred)-n+1))
            ref_ngrams  = Counter(tuple(ref[i:i+n])  for i in range(len(ref)-n+1))
            match += sum((pred_ngrams & ref_ngrams).values())
            total += sum(pred_ngrams.values())
        p_ns.append((match + smooth) / (total + smooth))

    # Geometric mean of precisions
    log_prec = sum(math.log(p) for p in p_ns) / max_n
    geo_mean = math.exp(log_prec)

    # Brevity penalty
    pred_len = sum(len(p) for p in predictions)
    ref_len  = sum(len(r) for r in references)
    bp = 1 if pred_len > ref_len else math.exp(1 - ref_len / pred_len)

    return bp * geo_mean


# 4) ROUGE‑L (longest common subsequence‑based F‑score) ------------------------
def lcs_length(x, y):
    """
    Length of the longest common subsequence between sequences x and y.
    DP in O(len(x) * len(y)).
    """
    m, n = len(x), len(y)
    prev = [0]*(n+1)
    for i in range(1, m+1):
        curr = [0]
        for j in range(1, n+1):
            if x[i-1] == y[j-1]:
                curr.append(prev[j-1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]

def rouge_l(predictions, references, beta=1.2):
    """
    Corpus‑level ROUGE‑L F‑score (default beta = 1.2 as in Lin 2004).
    predictions and references are lists of token lists.
    """
    assert len(predictions) == len(references)
    lcs_tot, pred_tot, ref_tot = 0, 0, 0
    for pred, ref in zip(predictions, references):
        lcs = lcs_length(pred, ref)
        lcs_tot += lcs
        pred_tot += len(pred)
        ref_tot  += len(ref)
    prec = lcs_tot / pred_tot if pred_tot else 0.0
    rec  = lcs_tot / ref_tot  if ref_tot  else 0.0
    if prec == 0 or rec == 0:
        return 0.0
    beta_sq = beta**2
    return (1 + beta_sq) * prec * rec / (rec + beta_sq * prec)
