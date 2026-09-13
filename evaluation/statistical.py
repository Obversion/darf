import numpy as np
from scipy.stats import wilcoxon, ttest_rel
from statsmodels.stats.multitest import multipletests

def wilcoxon_signed_rank_test(x, y):
    """Return statistic, p-value, and effect size (r)."""
    # Remove ties? Not needed.
    stat, p = wilcoxon(x, y)
    # Effect size r = Z / sqrt(n)
    # Z = (W - n*(n+1)/4) / sqrt(n*(n+1)*(2n+1)/24)
    n = len(x)
    if n > 0:
        # Compute Z
        W = stat
        # For paired, Wilcoxon uses the smaller of sum of ranks of positive and negative differences.
        # But we have stat from scipy which returns the smaller of the two rank sums.
        # We can compute Z from W:
        # Z = (W - n*(n+1)/4) / sqrt(n*(n+1)*(2n+1)/24)
        # But this is for the sum of positive ranks? We'll use a simplified approach:
        # Use the rank-biserial correlation from metrics module.
        from .metrics import compute_rank_biserial_correlation
        r = compute_rank_biserial_correlation(x, y)
        # Alternatively, compute Z and then r = Z / sqrt(n)
        # We'll just return p and r.
        return stat, p, r
    return stat, p, 0.0

def multiple_testing_correction(p_values, method='bonferroni'):
    """Apply multiple testing correction."""
    reject, pvals_corrected, alphac_sidak, alphac_bonf = multipletests(p_values, alpha=0.05, method=method)
    return pvals_corrected, reject