from typing import List, Tuple, Dict
from dataclasses import dataclass
import os, json
import numpy as np


@dataclass(frozen=True)
class ResultStatistics:
    """Statistics for a single parameter combination."""
    mean: float
    median: float
    std: float
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple (mean, median, std) for backward compatibility."""
        return (self.mean, self.median, self.std)
    
    @classmethod
    def from_tuple(cls, tup: Tuple[float, float, float]) -> 'ResultStatistics':
        """Create from tuple (mean, median, std)."""
        return cls(mean=tup[0], median=tup[1], std=tup[2])
    
    @classmethod
    def from_list(cls, values: List[float]) -> 'ResultStatistics':
        """Compute statistics from a list of values."""
        return cls(
            mean=float(np.mean(values)),
            median=float(np.median(values)),
            std=float(np.std(values))
        )


def save_json(obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_single_results(
    *,
    run_dir: str,
    p_values: List[float],
    sign_agreements: List[float] | List[Tuple[float, float, float]] | List[Tuple[float, float]],
    partition_agreement_M: List[float] | None = None,
    partition_agreement_S: List[float] | None = None,
    dot_products: List[float] | None = None,
    reference_partition_quality: float | None = None,
    sigma2_avg_M: List[float] | None = None,
    sigma2_avg_S: List[float] | None = None,
    partition_split_M: List[Tuple[int, int] | None] | None = None,
    partition_split_S: List[Tuple[int, int] | None] | None = None,
    result_source: List[str] | None = None,
    metrics_dict: Dict[str, List[Tuple[float, float, float]]] | None = None
):
    """
    Save single-parameter results with new partition metrics.

    Args:
        run_dir: Directory to save results
        p_values: List of p-values
        sign_agreements: Legacy sign agreement metric (backward compatibility)
        partition_agreement_M: NEW - Agreement using M for both partitions
        partition_agreement_S: NEW - Agreement using M vs S_avg
        dot_products: NEW - Vector alignment metric
        partition_split_M: NEW - Partition split sizes for M (n_small, n_large) or None
        partition_split_S: NEW - Partition split sizes for S_avg (n_small, n_large) or None
        result_source: NEW - Source of result ('computed', 'guardrail_high', etc.)
        metrics_dict: Matrix metrics (optional)

    Supports three formats for sign_agreements:
    1. New format: List[float] - single agreement value per p
    2. Legacy format: List[Tuple[float, float, float]] - (mean, median, std)
    3. Old format: List[Tuple[float, float]] - (median, std)
    """
    # Handle backward compatibility: check if tuples are 2-tuples or 3-tuples or single values
    has_metrics = metrics_dict is not None and len(metrics_dict) > 0

    if sign_agreements and isinstance(sign_agreements[0], (int, float)):
        # New format: single values
        columns = ["p", "sign_agreement"]
        rows = [{"p": float(p), "sign_agreement": float(agreement)} for p, agreement in zip(p_values, sign_agreements)]

        # Add new partition metrics if provided
        if partition_agreement_M is not None:
            columns.append("partition_agreement_M")
            for i, row in enumerate(rows):
                row["partition_agreement_M"] = float(partition_agreement_M[i])

        if partition_agreement_S is not None:
            columns.append("partition_agreement_S")
            for i, row in enumerate(rows):
                row["partition_agreement_S"] = float(partition_agreement_S[i])

        if dot_products is not None:
            columns.append("dot_product")
            for i, row in enumerate(rows):
                row["dot_product"] = float(dot_products[i])

        if sigma2_avg_M is not None:
            columns.append("sigma2_avg_M")
            for i, row in enumerate(rows):
                row["sigma2_avg_M"] = float(sigma2_avg_M[i])

        if sigma2_avg_S is not None:
            columns.append("sigma2_avg_S")
            for i, row in enumerate(rows):
                row["sigma2_avg_S"] = float(sigma2_avg_S[i])

        if partition_split_M is not None:
            columns.append("partition_split_M")
            for i, row in enumerate(rows):
                split = partition_split_M[i]
                row["partition_split_M"] = f"{split[0]}-{split[1]}" if split else None

        if partition_split_S is not None:
            columns.append("partition_split_S")
            for i, row in enumerate(rows):
                split = partition_split_S[i]
                row["partition_split_S"] = f"{split[0]}-{split[1]}" if split else None

        if result_source is not None:
            columns.append("result_source")
            for i, row in enumerate(rows):
                row["result_source"] = result_source[i]

    elif sign_agreements and len(sign_agreements[0]) == 2:
        # Old format: (median, std)
        columns = ["p", "median", "std"]
        rows = [{"p": float(p), "median": float(m), "std": float(s)} for p, (m, s) in zip(p_values, sign_agreements)]
    else:
        # Legacy format: (mean, median, std)
        columns = ["p", "mean", "median", "std"]
        rows = [{"p": float(p), "mean": float(mu), "median": float(m), "std": float(s)} for p, (mu, m, s) in zip(p_values, sign_agreements)]
    
    # Add metrics if provided
    if has_metrics:
        # All possible metric names (ordered for consistency)
        metric_names = [
            'operator_norm_error',
            'empirical_rank_M', 'empirical_rank_S', 'empirical_rank_L_M', 'empirical_rank_L_S',
            'spectral_gap_M', 'spectral_gap_S', 'spectral_gap_L_M', 'spectral_gap_L_S',
            'coherence_M', 'coherence_S', 'coherence_L_M', 'coherence_L_S',
            'min_separation_M', 'min_separation_S', 'min_separation_L_M', 'min_separation_L_S',
            # New spectral metrics
            'lambda2_L_M', 'lambda3_L_M', 'lambda2_L_S', 'lambda3_L_S',
            'ipr_S', 'dk_ratio_S',
            # Phase A: Leveraged sampling diagnostics
            'phase1_s1', 'phase1_s2', 'phase1_s3',
            'leverage_max', 'leverage_std', 'leverage_sum', 'leverage_symmetry_error',
            'ialm_iterations',
            'phase1_sufficiency'  # HLDT Phase 1 quality indicator
        ]
        for metric_name in metric_names:
            if metric_name in metrics_dict and len(metrics_dict[metric_name]) == len(p_values):
                columns.extend([f"mean_{metric_name}", f"median_{metric_name}", f"std_{metric_name}"])
                for i, row in enumerate(rows):
                    mu, med, std = metrics_dict[metric_name][i]
                    row[f"mean_{metric_name}"] = float(mu)
                    row[f"median_{metric_name}"] = float(med)
                    row[f"std_{metric_name}"] = float(std)
    
    result = {
        "columns": columns,
        "rows": rows,
        "reference_partition_quality": float(reference_partition_quality) if reference_partition_quality is not None else None
    }
    save_json(result, os.path.join(run_dir, "results.json"))
    np.save(os.path.join(run_dir, "sign_agreements.npy"), np.array(sign_agreements, dtype=float))

def save_multi_results(*, run_dir: str, p_values: List[float], all_results: Dict[float, List[Tuple[float, float, float]]] | Dict[float, List[Tuple[float, float]]]):
    """Save multi-parameter results (mutation rate sweep). Supports both old and new formats."""
    # Handle backward compatibility
    if all_results and len(list(all_results.values())[0][0]) == 2:
        rows = []
        for mu, sig in all_results.items():
            for p, (m, s) in zip(p_values, sig):
                rows.append({"mutation_rate": float(mu), "p": float(p), "median": float(m), "std": float(s)})
            np.save(os.path.join(run_dir, f"sign_agreements_mu={mu}.npy"), np.array(sig, dtype=float))
        save_json({"columns": ["mutation_rate", "p", "median", "std"], "rows": rows}, os.path.join(run_dir, "results_multi.json"))
    else:
        rows = []
        for mu, sig in all_results.items():
            for p, (mu_val, m, s) in zip(p_values, sig):
                rows.append({"mutation_rate": float(mu), "p": float(p), "mean": float(mu_val), "median": float(m), "std": float(s)})
            np.save(os.path.join(run_dir, f"sign_agreements_mu={mu}.npy"), np.array(sig, dtype=float))
        save_json({"columns": ["mutation_rate", "p", "mean", "median", "std"], "rows": rows}, os.path.join(run_dir, "results_multi.json"))

def save_taxa_results(
    *, 
    run_dir: str, 
    p_values: List[float], 
    all_results: Dict[int, List[float]] | Dict[int, List[Tuple[float, float, float]]] | Dict[int, List[Tuple[float, float]]], 
    all_metrics: Dict[int, Dict[str, List[Tuple[float, float, float]]]] | None = None,
    all_partition_agreement_M: Dict[int, List[float]] | None = None,
    all_partition_agreement_S: Dict[int, List[float]] | None = None,
    all_dot_products: Dict[int, List[float]] | None = None,
    all_reference_partition_quality: Dict[int, float] | None = None,
    all_sigma2_avg_M: Dict[int, List[float]] | None = None,
    all_sigma2_avg_S: Dict[int, List[float]] | None = None,
    all_partition_split_M: Dict[int, List[Tuple[int, int] | None]] | None = None,
    all_partition_split_S: Dict[int, List[Tuple[int, int] | None]] | None = None,
    all_result_source: Dict[int, List[str]] | None = None
):
    """
    Save taxa sweep results.
    
    Supports three formats for sign_agreements:
    1. New format: List[float] - single agreement value per p
    2. Legacy format: List[Tuple[float, float, float]] - (mean, median, std)
    3. Old format: List[Tuple[float, float]] - (median, std)
    """
    # Handle backward compatibility
    has_metrics = all_metrics is not None and len(all_metrics) > 0
    
    # Check format of first result
    first_result = list(all_results.values())[0] if all_results else []
    is_single_values = first_result and isinstance(first_result[0], (int, float))
    is_two_tuple = first_result and not is_single_values and len(first_result[0]) == 2
    
    if is_single_values:
        # New format: single values
        columns = ["num_taxa", "p", "sign_agreement"]
        rows = []
        for n_taxa, sig in all_results.items():
            for i, (p, agreement) in enumerate(zip(p_values, sig)):
                row = {"num_taxa": int(n_taxa), "p": float(p), "sign_agreement": float(agreement)}
                
                # Add partition agreements if provided
                if all_partition_agreement_M is not None and n_taxa in all_partition_agreement_M:
                    if "partition_agreement_M" not in columns:
                        columns.append("partition_agreement_M")
                    row["partition_agreement_M"] = float(all_partition_agreement_M[n_taxa][i])
                
                if all_partition_agreement_S is not None and n_taxa in all_partition_agreement_S:
                    if "partition_agreement_S" not in columns:
                        columns.append("partition_agreement_S")
                    row["partition_agreement_S"] = float(all_partition_agreement_S[n_taxa][i])
                
                if all_dot_products is not None and n_taxa in all_dot_products:
                    if "dot_product" not in columns:
                        columns.append("dot_product")
                    row["dot_product"] = float(all_dot_products[n_taxa][i])
                
                if all_sigma2_avg_M is not None and n_taxa in all_sigma2_avg_M:
                    if "sigma2_avg_M" not in columns:
                        columns.append("sigma2_avg_M")
                    row["sigma2_avg_M"] = float(all_sigma2_avg_M[n_taxa][i])
                
                if all_sigma2_avg_S is not None and n_taxa in all_sigma2_avg_S:
                    if "sigma2_avg_S" not in columns:
                        columns.append("sigma2_avg_S")
                    row["sigma2_avg_S"] = float(all_sigma2_avg_S[n_taxa][i])
                
                if all_partition_split_M is not None and n_taxa in all_partition_split_M:
                    if "partition_split_M" not in columns:
                        columns.append("partition_split_M")
                    split = all_partition_split_M[n_taxa][i]
                    row["partition_split_M"] = f"{split[0]}-{split[1]}" if split else None
                
                if all_partition_split_S is not None and n_taxa in all_partition_split_S:
                    if "partition_split_S" not in columns:
                        columns.append("partition_split_S")
                    split = all_partition_split_S[n_taxa][i]
                    row["partition_split_S"] = f"{split[0]}-{split[1]}" if split else None
                
                if all_result_source is not None and n_taxa in all_result_source:
                    if "result_source" not in columns:
                        columns.append("result_source")
                    row["result_source"] = all_result_source[n_taxa][i]
                
                rows.append(row)
            np.save(os.path.join(run_dir, f"sign_agreements_n={n_taxa}.npy"), np.array(sig, dtype=float))
    elif is_two_tuple:
        # Old format: (median, std)
        columns = ["num_taxa", "p", "median", "std"]
        rows = []
        for n_taxa, sig in all_results.items():
            for p, (m, s) in zip(p_values, sig):
                rows.append({"num_taxa": int(n_taxa), "p": float(p), "median": float(m), "std": float(s)})
            np.save(os.path.join(run_dir, f"sign_agreements_n={n_taxa}.npy"), np.array(sig, dtype=float))
    else:
        # Legacy format: (mean, median, std)
        columns = ["num_taxa", "p", "mean", "median", "std"]
        rows = []
        for n_taxa, sig in all_results.items():
            for p, (mu, m, s) in zip(p_values, sig):
                rows.append({"num_taxa": int(n_taxa), "p": float(p), "mean": float(mu), "median": float(m), "std": float(s)})
            np.save(os.path.join(run_dir, f"sign_agreements_n={n_taxa}.npy"), np.array(sig, dtype=float))
    
    # Add metrics if provided
    if has_metrics:
        metric_names = [
            'operator_norm_error',
            'empirical_rank_M', 'empirical_rank_S', 'empirical_rank_L_M', 'empirical_rank_L_S',
            'spectral_gap_M', 'spectral_gap_S', 'spectral_gap_L_M', 'spectral_gap_L_S',
            'coherence_M', 'coherence_S', 'coherence_L_M', 'coherence_L_S',
            'min_separation_M', 'min_separation_S', 'min_separation_L_M', 'min_separation_L_S',
            # New spectral metrics
            'lambda2_L_M', 'lambda3_L_M', 'lambda2_L_S', 'lambda3_L_S',
            'ipr_S', 'dk_ratio_S'
        ]
        for metric_name in metric_names:
            columns.extend([f"mean_{metric_name}", f"median_{metric_name}", f"std_{metric_name}"])
            row_idx = 0
            for n_taxa, sig in all_results.items():
                if n_taxa in all_metrics and metric_name in all_metrics[n_taxa]:
                    metrics_list = all_metrics[n_taxa][metric_name]
                    if len(metrics_list) == len(p_values):
                        for i in range(len(p_values)):
                            mu, med, std = metrics_list[i]
                            rows[row_idx][f"mean_{metric_name}"] = float(mu)
                            rows[row_idx][f"median_{metric_name}"] = float(med)
                            rows[row_idx][f"std_{metric_name}"] = float(std)
                            row_idx += 1
                    else:
                        row_idx += len(p_values)
                else:
                    row_idx += len(p_values)
    
    result = {
        "columns": columns,
        "rows": rows,
        "reference_partition_quality": {int(k): float(v) for k, v in all_reference_partition_quality.items()} if all_reference_partition_quality else None
    }
    save_json(result, os.path.join(run_dir, "results_taxa.json"))


def save_grid_results(
    *, 
    run_dir: str, 
    p_values: List[float], 
    all_results: Dict[Tuple[int, int], List[float]] | Dict[Tuple[int, int], List[Tuple[float, float, float]]], 
    all_metrics: Dict[Tuple[int, int], Dict[str, List[Tuple[float, float, float]]]] | None = None,
    all_partition_agreement_M: Dict[Tuple[int, int], List[float]] | None = None,
    all_partition_agreement_S: Dict[Tuple[int, int], List[float]] | None = None,
    all_dot_products: Dict[Tuple[int, int], List[float]] | None = None,
    all_reference_partition_quality: Dict[Tuple[int, int], float] | None = None,
    all_sigma2_avg_M: Dict[Tuple[int, int], List[float]] | None = None,
    all_sigma2_avg_S: Dict[Tuple[int, int], List[float]] | None = None,
    all_partition_split_M: Dict[Tuple[int, int], List[Tuple[int, int] | None]] | None = None,
    all_partition_split_S: Dict[Tuple[int, int], List[Tuple[int, int] | None]] | None = None,
    all_result_source: Dict[Tuple[int, int], List[str]] | None = None
):
    """
    Save grid search results for 2D parameter sweep (taxa × sequence_length).
    
    Supports two formats for sign_agreements:
    1. New format: List[float] - single agreement value per p
    2. Legacy format: List[Tuple[float, float, float]] - (mean, median, std)
    
    Args:
        run_dir: Directory to save results
        p_values: List of p values
        all_results: Dict mapping (n_taxa, seq_len) -> list of agreements
        all_metrics: Optional dict mapping (n_taxa, seq_len) -> metrics_dict
        all_partition_agreement_M: Optional partition agreement M values
        all_partition_agreement_S: Optional partition agreement S values
        all_dot_products: Optional dot product values
        all_partition_split_M: Optional partition split sizes for M
        all_partition_split_S: Optional partition split sizes for S_avg
        all_result_source: Optional result source flags
    """
    # Check format of first result
    first_result = list(all_results.values())[0] if all_results else []
    is_single_values = first_result and isinstance(first_result[0], (int, float))
    
    if is_single_values:
        # New format: single values
        columns = ["num_taxa", "sequence_length", "p", "sign_agreement"]
        rows = []
        for (n_taxa, seq_len), sig in all_results.items():
            for i, (p, agreement) in enumerate(zip(p_values, sig)):
                row = {
                    "num_taxa": int(n_taxa),
                    "sequence_length": int(seq_len),
                    "p": float(p),
                    "sign_agreement": float(agreement)
                }
                
                # Add partition agreements if provided
                key = (n_taxa, seq_len)
                if all_partition_agreement_M is not None and key in all_partition_agreement_M:
                    if "partition_agreement_M" not in columns:
                        columns.append("partition_agreement_M")
                    row["partition_agreement_M"] = float(all_partition_agreement_M[key][i])
                
                if all_partition_agreement_S is not None and key in all_partition_agreement_S:
                    if "partition_agreement_S" not in columns:
                        columns.append("partition_agreement_S")
                    row["partition_agreement_S"] = float(all_partition_agreement_S[key][i])
                
                if all_dot_products is not None and key in all_dot_products:
                    if "dot_product" not in columns:
                        columns.append("dot_product")
                    row["dot_product"] = float(all_dot_products[key][i])
                
                if all_sigma2_avg_M is not None and key in all_sigma2_avg_M:
                    if "sigma2_avg_M" not in columns:
                        columns.append("sigma2_avg_M")
                    row["sigma2_avg_M"] = float(all_sigma2_avg_M[key][i])
                
                if all_sigma2_avg_S is not None and key in all_sigma2_avg_S:
                    if "sigma2_avg_S" not in columns:
                        columns.append("sigma2_avg_S")
                    row["sigma2_avg_S"] = float(all_sigma2_avg_S[key][i])
                
                if all_partition_split_M is not None and key in all_partition_split_M:
                    if "partition_split_M" not in columns:
                        columns.append("partition_split_M")
                    split = all_partition_split_M[key][i]
                    row["partition_split_M"] = f"{split[0]}-{split[1]}" if split else None
                
                if all_partition_split_S is not None and key in all_partition_split_S:
                    if "partition_split_S" not in columns:
                        columns.append("partition_split_S")
                    split = all_partition_split_S[key][i]
                    row["partition_split_S"] = f"{split[0]}-{split[1]}" if split else None
                
                if all_result_source is not None and key in all_result_source:
                    if "result_source" not in columns:
                        columns.append("result_source")
                    row["result_source"] = all_result_source[key][i]
                
                rows.append(row)
            np.save(os.path.join(run_dir, f"sign_agreements_n={n_taxa}_L={seq_len}.npy"), np.array(sig, dtype=float))
    else:
        # Legacy format: (mean, median, std)
        columns = ["num_taxa", "sequence_length", "p", "mean", "median", "std"]
        rows = []
        for (n_taxa, seq_len), sig in all_results.items():
            for p, (mu, m, s) in zip(p_values, sig):
                rows.append({
                    "num_taxa": int(n_taxa),
                    "sequence_length": int(seq_len),
                    "p": float(p),
                    "mean": float(mu),
                    "median": float(m),
                    "std": float(s)
                })
            np.save(os.path.join(run_dir, f"sign_agreements_n={n_taxa}_L={seq_len}.npy"), np.array(sig, dtype=float))
    
    # Add metrics if provided
    if all_metrics is not None and len(all_metrics) > 0:
        metric_names = [
            'operator_norm_error',
            'empirical_rank_M', 'empirical_rank_S', 'empirical_rank_L_M', 'empirical_rank_L_S',
            'spectral_gap_M', 'spectral_gap_S', 'spectral_gap_L_M', 'spectral_gap_L_S',
            'coherence_M', 'coherence_S', 'coherence_L_M', 'coherence_L_S',
            'min_separation_M', 'min_separation_S', 'min_separation_L_M', 'min_separation_L_S',
            # New spectral metrics
            'lambda2_L_M', 'lambda3_L_M', 'lambda2_L_S', 'lambda3_L_S',
            'ipr_S', 'dk_ratio_S'
        ]
        for metric_name in metric_names:
            columns.extend([f"mean_{metric_name}", f"median_{metric_name}", f"std_{metric_name}"])
            row_idx = 0
            for (n_taxa, seq_len), sig in all_results.items():
                key = (n_taxa, seq_len)
                if key in all_metrics and metric_name in all_metrics[key]:
                    metrics_list = all_metrics[key][metric_name]
                    if len(metrics_list) == len(p_values):
                        for i in range(len(p_values)):
                            mu, med, std = metrics_list[i]
                            rows[row_idx][f"mean_{metric_name}"] = float(mu)
                            rows[row_idx][f"median_{metric_name}"] = float(med)
                            rows[row_idx][f"std_{metric_name}"] = float(std)
                            row_idx += 1
                    else:
                        row_idx += len(p_values)
                else:
                    row_idx += len(p_values)
    
    result = {
        "columns": columns,
        "rows": rows,
        "reference_partition_quality": {f"{k[0]}_{k[1]}": float(v) for k, v in all_reference_partition_quality.items()} if all_reference_partition_quality else None
    }
    save_json(
        result,
        os.path.join(run_dir, "results_grid.json")
    )