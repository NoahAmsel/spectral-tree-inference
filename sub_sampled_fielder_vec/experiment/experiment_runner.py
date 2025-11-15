"""Experiment runner class for different experiment modes."""
import os
from typing import Tuple, Dict, List
from itertools import product

import numpy as np

from utils.experiment_config import Config, set_seed, make_run_dir, save_config
from utils.summaries import save_single_results, save_taxa_results, save_grid_results
from utils.plotting import (
    plot_from_json_simple,
    plot_fiedler_vectors,
    plot_faceted_by_sequence_length
)
from utils.random_entries import clear_similarity_cache
from utils.logging import log_info
from .bootstrap_sweep import sweep_for_params


class ExperimentRunner:
    """Orchestrates and runs bootstrap sweep experiments."""
    
    def __init__(self, cfg: Config):
        """
        Initialize experiment runner and perform setup.
        
        Args:
            cfg: Experiment configuration
        """
        self.cfg = cfg
        
        # Setup: seed, create run directory, save config
        set_seed(cfg.seed)
        self.run_dir = make_run_dir(cfg.run_name)
        save_config(cfg, self.run_dir)
    
    def run(self) -> Tuple[str, Dict[Tuple[int, int], List[float]] | Dict[int, List[float]] | List[float]]:
        """
        Run the appropriate experiment based on configuration.
        
        Returns:
            Tuple of (run_dir, results):
            - list[float] for single parameter combination - single agreement per p-value
            - dict[n_taxa -> list[float]] for taxa sweep only
            - dict[(n_taxa, seq_len) -> list[float]] for grid search
        """
        # Dispatch to appropriate experiment runner
        if self.cfg.taxa_values is None and self.cfg.sequence_length_values is None:
            return self._run_single_experiment()
        
        elif self.cfg.sequence_length_values is None:
            return self._run_taxa_sweep()
        
        else:
            return self._run_grid_search()
    
    def _run_single_experiment(self) -> Tuple[str, List[float]]:
        """
        Run a single parameter combination experiment.
        
        Returns:
            Tuple of (run_dir, sign_agreements) - list of single agreement values per p-value
        """
        fiedler_ref, sign_agreements, metrics_dict = sweep_for_params(
            cfg=self.cfg,
            n_taxa=self.cfg.num_taxa,
            seq_len=self.cfg.sequence_length,
            run_dir=self.run_dir,
            incremental_save=False
        )
        
        # Final save + plot via helper modules
        save_single_results(
            run_dir=self.run_dir,
            p_values=self.cfg.p_values,
            sign_agreements=sign_agreements,
            metrics_dict=metrics_dict
        )
        np.save(os.path.join(self.run_dir, "fiedler_ref.npy"), fiedler_ref)
        
        plot_from_json_simple(
            json_path=os.path.join(self.run_dir, "results.json"),
            output_path=os.path.join(self.run_dir, "plot_single.png")
        )
        plot_fiedler_vectors(
            run_dir=self.run_dir,
            output_path=os.path.join(self.run_dir, "fiedler_vectors.png")
        )
        
        log_info('experiment', f"Completed! Results saved to: {self.run_dir}")
        # Clear cache after single run
        clear_similarity_cache()
        
        return self.run_dir, sign_agreements
    
    def _run_taxa_sweep(self) -> Tuple[str, Dict[int, List[float]]]:
        """
        Run taxa sweep experiment (varying taxa, fixed sequence length).
        
        Returns:
            Tuple of (run_dir, all_results) - dict mapping taxa counts to list of single agreement values
        """
        all_results: Dict[int, List[float]] = {}
        all_metrics: Dict[int, Dict[str, List[Tuple[float, float, float]]]] = {}
        
        for i, n_taxa in enumerate(self.cfg.taxa_values):
            log_info('experiment', f"\n{'='*80}")
            log_info('experiment', f"Processing taxa count {i+1}/{len(self.cfg.taxa_values)}: n={n_taxa}")
            log_info('experiment', f"{'='*80}")
            
            # Clear cache before each new taxa count to avoid memory buildup
            if i > 0:  # Don't clear on first iteration
                clear_similarity_cache()
            
            fiedler_ref, sig, metrics_dict = sweep_for_params(
                cfg=self.cfg,
                n_taxa=n_taxa,
                seq_len=self.cfg.sequence_length,
                run_dir=self.run_dir,
                incremental_save=False
            )
            all_results[n_taxa] = sig
            all_metrics[n_taxa] = metrics_dict
            np.save(os.path.join(self.run_dir, f"fiedler_ref_n={n_taxa}.npy"), fiedler_ref)
            
            # Save incremental taxa-results after each taxa count
            save_taxa_results(
                run_dir=self.run_dir,
                p_values=self.cfg.p_values,
                all_results=all_results,
                all_metrics=all_metrics
            )
            log_info('experiment', f"Saved incremental taxa-results for n={n_taxa}")
        
        # Clear cache after all runs
        clear_similarity_cache()
        
        # Final save + plot
        save_taxa_results(
            run_dir=self.run_dir,
            p_values=self.cfg.p_values,
            all_results=all_results,
            all_metrics=all_metrics
        )
        plot_from_json_simple(
            json_path=os.path.join(self.run_dir, "results_taxa.json"),
            output_path=os.path.join(self.run_dir, "plot_multi_taxa.png")
        )
        plot_fiedler_vectors(
            run_dir=self.run_dir,
            output_path=os.path.join(self.run_dir, f"fiedler_vectors_mu={self.cfg.mutation_rate}.png")
        )
        
        log_info('experiment', f"[done] artifacts written to: {self.run_dir} for taxa counts: {list(self.cfg.taxa_values)}")
        
        return self.run_dir, all_results
    
    def _run_grid_search(self) -> Tuple[str, Dict[Tuple[int, int], List[float]]]:
        """
        Run grid search experiment (varying both taxa and sequence length).
        
        Returns:
            Tuple of (run_dir, all_results) - dict mapping (taxa, seq_len) to list of single agreement values
        """
        all_results: Dict[Tuple[int, int], List[float]] = {}
        all_metrics: Dict[Tuple[int, int], Dict[str, List[Tuple[float, float, float]]]] = {}
        total_combinations = len(self.cfg.taxa_values) * len(self.cfg.sequence_length_values)
        combo_idx = 0
        
        for n_taxa, seq_len in product(self.cfg.taxa_values, self.cfg.sequence_length_values):
            combo_idx += 1
            log_info('experiment', f"\n{'='*80}")
            log_info('experiment', f"Processing combination {combo_idx}/{total_combinations}: n={n_taxa}, L={seq_len}")
            log_info('experiment', f"{'='*80}")
            
            # Clear cache before each new combination (except first)
            if combo_idx > 1:
                clear_similarity_cache()
            
            fiedler_ref, sig, metrics_dict = sweep_for_params(
                cfg=self.cfg,
                n_taxa=n_taxa,
                seq_len=seq_len,
                run_dir=self.run_dir,
                incremental_save=False
            )
            all_results[(n_taxa, seq_len)] = sig
            all_metrics[(n_taxa, seq_len)] = metrics_dict
            np.save(os.path.join(self.run_dir, f"fiedler_ref_n={n_taxa}_L={seq_len}.npy"), fiedler_ref)
            
            # Save incremental grid results after each combination
            save_grid_results(
                run_dir=self.run_dir,
                p_values=self.cfg.p_values,
                all_results=all_results,
                all_metrics=all_metrics
            )
            log_info('experiment', f"Saved incremental grid results for n={n_taxa}, L={seq_len}")
        
        # Clear cache after all runs
        clear_similarity_cache()
        
        # Final save + plot
        save_grid_results(
            run_dir=self.run_dir,
            p_values=self.cfg.p_values,
            all_results=all_results,
            all_metrics=all_metrics
        )
        
        plot_faceted_by_sequence_length(
            json_path=os.path.join(self.run_dir, "results_grid.json"),
            output_path=os.path.join(self.run_dir, "plot_grid_faceted.png")
        )
        
        plot_fiedler_vectors(
            run_dir=self.run_dir,
            output_path=os.path.join(self.run_dir, f"fiedler_vectors_mu={self.cfg.mutation_rate}.png")
        )
        
        log_info('experiment',
            f"[done] artifacts written to: {self.run_dir} for grid search: "
            f"taxa={list(self.cfg.taxa_values)}, seq_len={list(self.cfg.sequence_length_values)}"
        )
        
        return self.run_dir, all_results
