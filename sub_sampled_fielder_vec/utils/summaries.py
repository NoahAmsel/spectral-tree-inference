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

def save_single_results(*, run_dir: str, p_values: List[float], sign_agreements: List[float] | List[Tuple[float, float, float]] | List[Tuple[float, float]], metrics_dict: Dict[str, List[Tuple[float, float, float]]] | None = None):
    """
    Save single-parameter results.
    
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
            'min_separation_M', 'min_separation_S', 'min_separation_L_M', 'min_separation_L_S'
        ]
        for metric_name in metric_names:
            if metric_name in metrics_dict and len(metrics_dict[metric_name]) == len(p_values):
                columns.extend([f"mean_{metric_name}", f"median_{metric_name}", f"std_{metric_name}"])
                for i, row in enumerate(rows):
                    mu, med, std = metrics_dict[metric_name][i]
                    row[f"mean_{metric_name}"] = float(mu)
                    row[f"median_{metric_name}"] = float(med)
                    row[f"std_{metric_name}"] = float(std)
    
    save_json({"columns": columns, "rows": rows}, os.path.join(run_dir, "results.json"))
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

def save_taxa_results(*, run_dir: str, p_values: List[float], all_results: Dict[int, List[float]] | Dict[int, List[Tuple[float, float, float]]] | Dict[int, List[Tuple[float, float]]], all_metrics: Dict[int, Dict[str, List[Tuple[float, float, float]]]] | None = None):
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
            for p, agreement in zip(p_values, sig):
                rows.append({"num_taxa": int(n_taxa), "p": float(p), "sign_agreement": float(agreement)})
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
            'min_separation_M', 'min_separation_S', 'min_separation_L_M', 'min_separation_L_S'
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
    
    save_json({"columns": columns, "rows": rows}, os.path.join(run_dir, "results_taxa.json"))


def save_grid_results(*, run_dir: str, p_values: List[float], all_results: Dict[Tuple[int, int], List[float]] | Dict[Tuple[int, int], List[Tuple[float, float, float]]], all_metrics: Dict[Tuple[int, int], Dict[str, List[Tuple[float, float, float]]]] | None = None):
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
    """
    # Check format of first result
    first_result = list(all_results.values())[0] if all_results else []
    is_single_values = first_result and isinstance(first_result[0], (int, float))
    
    if is_single_values:
        # New format: single values
        columns = ["num_taxa", "sequence_length", "p", "sign_agreement"]
        rows = []
        for (n_taxa, seq_len), sig in all_results.items():
            for p, agreement in zip(p_values, sig):
                rows.append({
                    "num_taxa": int(n_taxa),
                    "sequence_length": int(seq_len),
                    "p": float(p),
                    "sign_agreement": float(agreement)
                })
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
            'min_separation_M', 'min_separation_S', 'min_separation_L_M', 'min_separation_L_S'
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
    
    save_json(
        {"columns": columns, "rows": rows},
        os.path.join(run_dir, "results_grid.json")
    )