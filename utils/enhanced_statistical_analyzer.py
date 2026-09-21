"""
Enhanced statistical analysis with effect size reporting and multiplicity correction.

This module implements comprehensive statistical analysis for experiment results
including eta-squared effect sizes and Benjamini-Hochberg FDR correction.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.stats import kruskal, rankdata
from statsmodels.stats.multitest import multipletests
import itertools


@dataclass
class EffectSizeResult:
    """Effect size calculation result."""
    eta_squared: float
    interpretation: str  # "small", "medium", "large"
    confidence_interval: Tuple[float, float]


@dataclass
class PostHocResult:
    """Post-hoc comparison result."""
    group1: str
    group2: str
    statistic: float
    p_value: float
    adjusted_p_value: float
    effect_size: float
    significant: bool


@dataclass
class KruskalWallisResult:
    """Complete Kruskal-Wallis analysis result."""
    h_statistic: float
    p_value: float
    eta_squared: EffectSizeResult
    post_hoc_results: List[PostHocResult]
    significant: bool
    interpretation: str


class EnhancedStatisticalAnalyzer:
    """
    Enhanced statistical analyzer with effect sizes and multiplicity correction.
    
    Implements:
    - Kruskal-Wallis with eta-squared effect sizes
    - Post-hoc Dunn's tests with Benjamini-Hochberg correction
    - Comprehensive effect size interpretation
    - Robust confidence intervals
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical analyzer.
        
        Args:
            alpha: Significance level (default 0.05)
        """
        self.alpha = alpha
        
        # Effect size interpretation thresholds (Cohen's conventions)
        self.eta_squared_thresholds = {
            0.01: "small",
            0.06: "medium", 
            0.14: "large"
        }
    
    def calculate_eta_squared(
        self,
        groups: List[np.ndarray],
        h_statistic: float,
        total_n: int
    ) -> EffectSizeResult:
        """
        Calculate eta-squared effect size for Kruskal-Wallis test.
        
        Args:
            groups: List of group data arrays
            h_statistic: Kruskal-Wallis H statistic
            total_n: Total sample size
            
        Returns:
            EffectSizeResult with eta-squared and interpretation
        """
        # Eta-squared for Kruskal-Wallis: η² = (H - k + 1) / (n - k)
        k = len(groups)  # Number of groups
        
        if total_n <= k:
            # Insufficient data for meaningful effect size
            return EffectSizeResult(
                eta_squared=0.0,
                interpretation="insufficient_data",
                confidence_interval=(0.0, 0.0)
            )
        
        eta_squared = (h_statistic - k + 1) / (total_n - k)
        eta_squared = max(0, eta_squared)  # Ensure non-negative
        
        # Interpret effect size
        interpretation = "negligible"
        for threshold in sorted(self.eta_squared_thresholds.keys()):
            if eta_squared >= threshold:
                interpretation = self.eta_squared_thresholds[threshold]
        
        # Calculate confidence interval (approximation)
        # Using bootstrap approximation for CI
        ci_lower, ci_upper = self._bootstrap_eta_squared_ci(groups, eta_squared)
        
        return EffectSizeResult(
            eta_squared=eta_squared,
            interpretation=interpretation,
            confidence_interval=(ci_lower, ci_upper)
        )
    
    def _bootstrap_eta_squared_ci(
        self,
        groups: List[np.ndarray],
        observed_eta_squared: float,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for eta-squared."""
        bootstrap_etas = []
        
        for _ in range(n_bootstrap):
            # Bootstrap resample each group
            bootstrap_groups = [
                np.random.choice(group, size=len(group), replace=True)
                for group in groups
            ]
            
            try:
                # Calculate Kruskal-Wallis for bootstrap sample
                h_boot, _ = kruskal(*bootstrap_groups)
                total_n_boot = sum(len(g) for g in bootstrap_groups)
                k_boot = len(bootstrap_groups)
                
                eta_boot = (h_boot - k_boot + 1) / (total_n_boot - k_boot)
                eta_boot = max(0, eta_boot)
                bootstrap_etas.append(eta_boot)
                
            except (ValueError, ZeroDivisionError):
                # Skip failed bootstrap samples
                continue
        
        if bootstrap_etas:
            ci_lower = np.percentile(bootstrap_etas, 2.5)
            ci_upper = np.percentile(bootstrap_etas, 97.5)
        else:
            # Fallback to observed value
            ci_lower = ci_upper = observed_eta_squared
        
        return ci_lower, ci_upper
    
    def dunn_test_with_correction(
        self,
        groups: List[np.ndarray],
        group_names: List[str],
        correction_method: str = "benjamini-hochberg"
    ) -> List[PostHocResult]:
        """
        Perform Dunn's post-hoc test with multiplicity correction.
        
        Args:
            groups: List of group data arrays
            group_names: Names of the groups
            correction_method: "benjamini-hochberg", "bonferroni", or "holm"
            
        Returns:
            List of PostHocResult objects
        """
        if len(groups) != len(group_names):
            raise ValueError("Number of groups must match number of group names")
        
        # Combine all data and create group labels
        all_data = np.concatenate(groups)
        group_labels = np.concatenate([
            np.full(len(group), i) for i, group in enumerate(groups)
        ])
        
        # Rank all data
        ranks = rankdata(all_data)
        
        # Calculate mean ranks for each group
        mean_ranks = []
        for i, group in enumerate(groups):
            group_ranks = ranks[group_labels == i]
            mean_ranks.append(np.mean(group_ranks))
        
        # Total sample size
        N = len(all_data)
        
        results = []
        p_values = []
        
        # Perform pairwise comparisons
        for i, j in itertools.combinations(range(len(groups)), 2):
            n_i, n_j = len(groups[i]), len(groups[j])
            
            # Dunn's test statistic
            z_statistic = (mean_ranks[i] - mean_ranks[j]) / np.sqrt(
                (N * (N + 1) / 12) * (1/n_i + 1/n_j)
            )
            
            # Two-tailed p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
            p_values.append(p_value)
            
            # Effect size (rank-biserial correlation approximation)
            effect_size = abs(mean_ranks[i] - mean_ranks[j]) / (N + 1)
            
            results.append(PostHocResult(
                group1=group_names[i],
                group2=group_names[j],
                statistic=z_statistic,
                p_value=p_value,
                adjusted_p_value=p_value,  # Will be updated after correction
                effect_size=effect_size,
                significant=False  # Will be updated after correction
            ))
        
        # Apply multiplicity correction
        correction_map = {
            "benjamini-hochberg": "fdr_bh",
            "bonferroni": "bonferroni",
            "holm": "holm"
        }
        
        if correction_method in correction_map:
            rejected, adjusted_p_values, _, _ = multipletests(
                p_values, alpha=self.alpha, method=correction_map[correction_method]
            )
            
            # Update results with corrected p-values
            for i, result in enumerate(results):
                result.adjusted_p_value = adjusted_p_values[i]
                result.significant = rejected[i]
        
        return results
    
    def comprehensive_kruskal_wallis_analysis(
        self,
        data_dict: Dict[str, List[float]],
        correction_method: str = "benjamini-hochberg"
    ) -> KruskalWallisResult:
        """
        Perform comprehensive Kruskal-Wallis analysis with effect sizes and post-hoc tests.
        
        Args:
            data_dict: Dictionary mapping group names to data lists
            correction_method: Method for multiple comparison correction
            
        Returns:
            KruskalWallisResult with complete analysis
        """
        group_names = list(data_dict.keys())
        groups = [np.array(data_dict[name]) for name in group_names]
        
        # Validate input
        if len(groups) < 2:
            raise ValueError("Need at least 2 groups for analysis")
        
        for i, group in enumerate(groups):
            if len(group) == 0:
                raise ValueError(f"Group '{group_names[i]}' is empty")
        
        # Perform Kruskal-Wallis test
        h_statistic, p_value = kruskal(*groups)
        total_n = sum(len(group) for group in groups)
        
        # Calculate effect size
        eta_squared_result = self.calculate_eta_squared(groups, h_statistic, total_n)
        
        # Determine significance
        significant = p_value < self.alpha
        
        # Post-hoc tests (only if main test is significant and >2 groups)
        post_hoc_results = []
        if significant and len(groups) > 2:
            post_hoc_results = self.dunn_test_with_correction(
                groups, group_names, correction_method
            )
        
        # Generate interpretation
        interpretation = self._generate_interpretation(
            h_statistic, p_value, eta_squared_result, post_hoc_results
        )
        
        return KruskalWallisResult(
            h_statistic=h_statistic,
            p_value=p_value,
            eta_squared=eta_squared_result,
            post_hoc_results=post_hoc_results,
            significant=significant,
            interpretation=interpretation
        )
    
    def _generate_interpretation(
        self,
        h_statistic: float,
        p_value: float,
        eta_squared_result: EffectSizeResult,
        post_hoc_results: List[PostHocResult]
    ) -> str:
        """Generate human-readable interpretation of results."""
        
        interpretation_parts = []
        
        # Main test interpretation
        if p_value < self.alpha:
            interpretation_parts.append(
                f"Kruskal-Wallis test shows significant differences between groups "
                f"(H = {h_statistic:.3f}, p = {p_value:.4f})"
            )
        else:
            interpretation_parts.append(
                f"Kruskal-Wallis test shows no significant differences between groups "
                f"(H = {h_statistic:.3f}, p = {p_value:.4f})"
            )
        
        # Effect size interpretation
        interpretation_parts.append(
            f"Effect size is {eta_squared_result.interpretation} "
            f"(η² = {eta_squared_result.eta_squared:.3f}, "
            f"95% CI: [{eta_squared_result.confidence_interval[0]:.3f}, "
            f"{eta_squared_result.confidence_interval[1]:.3f}])"
        )
        
        # Post-hoc interpretation
        if post_hoc_results:
            significant_pairs = [r for r in post_hoc_results if r.significant]
            if significant_pairs:
                interpretation_parts.append(
                    f"Post-hoc analysis reveals {len(significant_pairs)} significant pairwise difference(s):"
                )
                for result in significant_pairs:
                    interpretation_parts.append(
                        f"  - {result.group1} vs {result.group2}: "
                        f"p_adj = {result.adjusted_p_value:.4f}, "
                        f"effect size = {result.effect_size:.3f}"
                    )
            else:
                interpretation_parts.append(
                    "Post-hoc analysis reveals no significant pairwise differences after correction"
                )
        
        return " ".join(interpretation_parts)
    
    def analyze_experiment_results(
        self,
        results_df: pd.DataFrame,
        group_column: str,
        value_columns: List[str],
        correction_method: str = "benjamini-hochberg"
    ) -> Dict[str, KruskalWallisResult]:
        """
        Analyze multiple outcome variables from experiment results.
        
        Args:
            results_df: DataFrame with experiment results
            group_column: Column name containing group labels
            value_columns: List of outcome variable column names
            correction_method: Multiple comparison correction method
            
        Returns:
            Dictionary mapping outcome variables to analysis results
        """
        analysis_results = {}
        
        for outcome in value_columns:
            # Prepare data for this outcome
            data_dict = {}
            for group in results_df[group_column].unique():
                group_data = results_df[results_df[group_column] == group][outcome].dropna()
                if len(group_data) > 0:
                    data_dict[str(group)] = group_data.tolist()
            
            if len(data_dict) >= 2:
                try:
                    result = self.comprehensive_kruskal_wallis_analysis(
                        data_dict, correction_method
                    )
                    analysis_results[outcome] = result
                except Exception as e:
                    print(f"Warning: Analysis failed for {outcome}: {e}")
            else:
                print(f"Warning: Insufficient groups for analysis of {outcome}")
        
        return analysis_results
    
    def generate_statistical_report(
        self,
        analysis_results: Dict[str, KruskalWallisResult]
    ) -> str:
        """Generate comprehensive statistical report."""
        
        report_lines = [
            "# Enhanced Statistical Analysis Report",
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"Analyzed {len(analysis_results)} outcome variables using Kruskal-Wallis tests",
            "with eta-squared effect sizes and Benjamini-Hochberg FDR correction.",
            ""
        ]
        
        # Summary statistics
        significant_outcomes = sum(1 for result in analysis_results.values() if result.significant)
        
        report_lines.extend([
            f"- Significant outcomes: {significant_outcomes}/{len(analysis_results)}",
            f"- Effect sizes range from negligible to large",
            ""
        ])
        
        # Detailed results for each outcome
        for outcome, result in analysis_results.items():
            report_lines.extend([
                f"## {outcome}",
                f"- H-statistic: {result.h_statistic:.3f}",
                f"- p-value: {result.p_value:.4f}",
                f"- Significant: {'Yes' if result.significant else 'No'}",
                f"- Effect size (η²): {result.eta_squared.eta_squared:.3f} ({result.eta_squared.interpretation})",
                f"- 95% CI for η²: [{result.eta_squared.confidence_interval[0]:.3f}, {result.eta_squared.confidence_interval[1]:.3f}]",
                ""
            ])
            
            if result.post_hoc_results:
                report_lines.append("### Post-hoc Comparisons:")
                for ph_result in result.post_hoc_results:
                    significance_marker = "*" if ph_result.significant else ""
                    report_lines.append(
                        f"- {ph_result.group1} vs {ph_result.group2}: "
                        f"p_adj = {ph_result.adjusted_p_value:.4f}{significance_marker}, "
                        f"effect = {ph_result.effect_size:.3f}"
                    )
                report_lines.append("")
            
            report_lines.extend([
                f"**Interpretation:** {result.interpretation}",
                ""
            ])
        
        return "\n".join(report_lines)
