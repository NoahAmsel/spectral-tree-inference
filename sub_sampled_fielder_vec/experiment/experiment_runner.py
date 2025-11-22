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
from utils.logging import log_info, create_progress_bar, create_config_progress_bar, set_display_mode, is_progress_mode
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

        # Set display mode globally
        set_display_mode(cfg.display_mode)

        # Print opening title with configuration
        self._print_opening_title()

        # Setup: seed, create run directory, save config
        set_seed(cfg.seed)
        self.run_dir = make_run_dir(cfg.run_name)
        save_config(cfg, self.run_dir)

    def _print_opening_title(self):
        """Print opening title with experiment configuration."""
        cfg = self.cfg

        print("\n" + "="*80)
        print("SPECTRAL TREE INFERENCE - Sub-sampled STDR Experiment")
        print("="*80)

        # Determine experiment type
        if cfg.taxa_values is None and cfg.sequence_length_values is None:
            exp_type = "Single Configuration"
            configs = f"n={cfg.num_taxa}, L={cfg.sequence_length}"
        elif cfg.sequence_length_values is None:
            exp_type = "Taxa Sweep"
            configs = f"n={list(cfg.taxa_values)}, L={cfg.sequence_length}"
        else:
            exp_type = "Grid Search"
            configs = f"n={list(cfg.taxa_values)}, L={list(cfg.sequence_length_values)}"

        print(f"Experiment Type:    {exp_type}")
        print(f"Configurations:     {configs}")
        print(f"Mutation Rate:      μ = {cfg.mutation_rate}")
        print(f"P-values:           {len(cfg.p_values)} values from {min(cfg.p_values):.2e} to {max(cfg.p_values):.2e}")
        print(f"Bootstrap Reps:     {cfg.bootstrap_reps}")
        print(f"Display Mode:       {cfg.display_mode}")
        print(f"Run Name:           {cfg.run_name}")
        print("="*80 + "\n")

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
        fiedler_ref, sign_agreements, partition_agreement_M, partition_agreement_S, dot_products, metrics_dict = sweep_for_params(
            cfg=self.cfg,
            n_taxa=self.cfg.num_taxa,
            seq_len=self.cfg.sequence_length,
            run_dir=self.run_dir,
            incremental_save=False,
            show_progress=True  # Show all progress bars for single experiment
        )

        # Final save + plot via helper modules
        save_single_results(
            run_dir=self.run_dir,
            p_values=self.cfg.p_values,
            sign_agreements=sign_agreements,
            partition_agreement_M=partition_agreement_M,
            partition_agreement_S=partition_agreement_S,
            dot_products=dot_products,
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
        
        log_info('experiment', f"Completed! Results saved to: {self.run_dir}", force=True)
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
        all_partition_agreement_M: Dict[int, List[float]] = {}
        all_partition_agreement_S: Dict[int, List[float]] = {}
        all_dot_products: Dict[int, List[float]] = {}

        # Create all configuration progress bars upfront
        config_pbars = []
        for i, n_taxa in enumerate(self.cfg.taxa_values):
            pbar = create_config_progress_bar(
                config_idx=i + 1,
                total_configs=len(self.cfg.taxa_values),
                n_taxa=n_taxa,
                seq_len=self.cfg.sequence_length,
                num_p_values=len(self.cfg.p_values),
                position=i  # 0-indexed positions
            )
            config_pbars.append(pbar)

        # Run experiments for each taxa value
        for i, n_taxa in enumerate(self.cfg.taxa_values):
            pbar = config_pbars[i]

            # Clear cache before each new taxa count to avoid memory buildup
            if i > 0:  # Don't clear on first iteration
                clear_similarity_cache()

            # Define callback to update this config's progress bar
            def update_progress(p_idx):
                pbar.update(1)

            fiedler_ref, sig, partition_agr_M, partition_agr_S, dot_prod, metrics_dict = sweep_for_params(
                cfg=self.cfg,
                n_taxa=n_taxa,
                seq_len=self.cfg.sequence_length,
                run_dir=self.run_dir,
                incremental_save=False,
                show_progress=not is_progress_mode(),  # Only show bootstrap bars in debug mode
                progress_callback=update_progress
            )
            all_results[n_taxa] = sig
            all_metrics[n_taxa] = metrics_dict
            all_partition_agreement_M[n_taxa] = partition_agr_M
            all_partition_agreement_S[n_taxa] = partition_agr_S
            all_dot_products[n_taxa] = dot_prod
            np.save(os.path.join(self.run_dir, f"fiedler_ref_n={n_taxa}.npy"), fiedler_ref)

            # Save incremental taxa-results after each taxa count
            save_taxa_results(
                run_dir=self.run_dir,
                p_values=self.cfg.p_values,
                all_results=all_results,
                all_metrics=all_metrics,
                all_partition_agreement_M=all_partition_agreement_M,
                all_partition_agreement_S=all_partition_agreement_S,
                all_dot_products=all_dot_products
            )

        # Clear cache after all runs
        clear_similarity_cache()

        # Close all progress bars
        for pbar in config_pbars:
            pbar.close()
        
        # Final save + plot
        save_taxa_results(
            run_dir=self.run_dir,
            p_values=self.cfg.p_values,
            all_results=all_results,
            all_metrics=all_metrics,
            all_partition_agreement_M=all_partition_agreement_M,
            all_partition_agreement_S=all_partition_agreement_S,
            all_dot_products=all_dot_products
        )
        plot_from_json_simple(
            json_path=os.path.join(self.run_dir, "results_taxa.json"),
            output_path=os.path.join(self.run_dir, "plot_multi_taxa.png")
        )
        plot_fiedler_vectors(
            run_dir=self.run_dir,
            output_path=os.path.join(self.run_dir, f"fiedler_vectors_mu={self.cfg.mutation_rate}.png")
        )
        
        log_info('experiment', f"[done] artifacts written to: {self.run_dir} for taxa counts: {list(self.cfg.taxa_values)}", force=True)
        
        return self.run_dir, all_results
    
    def _run_grid_search(self) -> Tuple[str, Dict[Tuple[int, int], List[float]]]:
        """
        Run grid search experiment (varying both taxa and sequence length).

        Returns:
            Tuple of (run_dir, all_results) - dict mapping (taxa, seq_len) to list of single agreement values
        """
        all_results: Dict[Tuple[int, int], List[float]] = {}
        all_metrics: Dict[Tuple[int, int], Dict[str, List[Tuple[float, float, float]]]] = {}
        all_partition_agreement_M: Dict[Tuple[int, int], List[float]] = {}
        all_partition_agreement_S: Dict[Tuple[int, int], List[float]] = {}
        all_dot_products: Dict[Tuple[int, int], List[float]] = {}
        total_combinations = len(self.cfg.taxa_values) * len(self.cfg.sequence_length_values)

        # Create all configuration progress bars upfront
        config_pbars = []
        combo_idx = 0
        for n_taxa, seq_len in product(self.cfg.taxa_values, self.cfg.sequence_length_values):
            combo_idx += 1
            pbar = create_config_progress_bar(
                config_idx=combo_idx,
                total_configs=total_combinations,
                n_taxa=n_taxa,
                seq_len=seq_len,
                num_p_values=len(self.cfg.p_values),
                position=combo_idx - 1  # 0-indexed positions
            )
            config_pbars.append(pbar)

        # Run experiments for each configuration
        combo_idx = 0
        for n_taxa, seq_len in product(self.cfg.taxa_values, self.cfg.sequence_length_values):
            pbar = config_pbars[combo_idx]

            # Clear cache before each new combination (except first)
            if combo_idx > 0:
                clear_similarity_cache()

            # Define callback to update this config's progress bar
            def update_progress(p_idx):
                pbar.update(1)

            fiedler_ref, sig, partition_agr_M, partition_agr_S, dot_prod, metrics_dict = sweep_for_params(
                cfg=self.cfg,
                n_taxa=n_taxa,
                seq_len=seq_len,
                run_dir=self.run_dir,
                incremental_save=False,
                show_progress=not is_progress_mode(),  # Only show bootstrap bars in debug mode
                progress_callback=update_progress
            )
            all_results[(n_taxa, seq_len)] = sig
            all_metrics[(n_taxa, seq_len)] = metrics_dict
            all_partition_agreement_M[(n_taxa, seq_len)] = partition_agr_M
            all_partition_agreement_S[(n_taxa, seq_len)] = partition_agr_S
            all_dot_products[(n_taxa, seq_len)] = dot_prod
            np.save(os.path.join(self.run_dir, f"fiedler_ref_n={n_taxa}_L={seq_len}.npy"), fiedler_ref)

            # Save incremental grid results after each combination
            save_grid_results(
                run_dir=self.run_dir,
                p_values=self.cfg.p_values,
                all_results=all_results,
                all_metrics=all_metrics,
                all_partition_agreement_M=all_partition_agreement_M,
                all_partition_agreement_S=all_partition_agreement_S,
                all_dot_products=all_dot_products
            )

            combo_idx += 1

        # Clear cache after all runs
        clear_similarity_cache()

        # Close all progress bars
        for pbar in config_pbars:
            pbar.close()
        
        # Final save + plot
        save_grid_results(
            run_dir=self.run_dir,
            p_values=self.cfg.p_values,
            all_results=all_results,
            all_metrics=all_metrics,
            all_partition_agreement_M=all_partition_agreement_M,
            all_partition_agreement_S=all_partition_agreement_S,
            all_dot_products=all_dot_products
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
            f"taxa={list(self.cfg.taxa_values)}, seq_len={list(self.cfg.sequence_length_values)}",
            force=True
        )
        
        return self.run_dir, all_results
