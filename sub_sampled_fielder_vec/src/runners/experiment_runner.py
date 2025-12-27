"""Experiment runner class for different experiment modes."""
import os
from typing import Tuple, Dict, List
from itertools import product
from pathlib import Path
import sys

import numpy as np

from ..config import StructuredConfig
from ..models import get_tree_factory, get_sequence_factory
import os
import time
import random
from ..utils.summaries import save_single_results, save_taxa_results, save_grid_results
from ..utils.plotting import (
    plot_from_json_simple,
    plot_fiedler_vectors,
    plot_faceted_by_sequence_length
)
from ..utils.random_entries import clear_similarity_cache
from ..utils.logging import (
    log_info, create_progress_bar, create_config_progress_bar,
    set_display_mode, is_progress_mode, setup_log_file, close_log_file, get_log_file_path
)
from .bootstrap_sweep import sweep_for_params

# Add spectral_analysis to path for tree plotting
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SPECTRAL_ANALYSIS_PATH = PACKAGE_ROOT / "spectral_analysis" / "target_quality_anlysis" / "visualization"
if str(SPECTRAL_ANALYSIS_PATH) not in sys.path:
    sys.path.insert(0, str(SPECTRAL_ANALYSIS_PATH))
from tree_plots import plot_tree_with_partition, plot_combined_tree_and_fiedler


class ExperimentRunner:
    """Orchestrates and runs bootstrap sweep experiments."""
    
    def __init__(self, cfg: StructuredConfig, base_dir: str = None, subdir_name: str = None):
        """
        Initialize experiment runner and perform setup.

        Args:
            cfg: Experiment configuration
            base_dir: Optional base directory for results (for grid sweeps)
            subdir_name: Optional subdirectory name within base_dir (for grid sweeps)
        """
        self.cfg = cfg

        # Set display mode globally
        set_display_mode(cfg.experiment.display_mode)

        # Print opening title with configuration
        self._print_opening_title()

        # Setup: seed, create run directory, save config
        self._set_seed(cfg.experiment.seed)
        self.run_dir = self._make_run_dir(cfg.experiment.run_name, base_dir, subdir_name)
        self._save_config(cfg, self.run_dir)

        # Setup file-based logging
        setup_log_file(self.run_dir)
        log_info('experiment', f"Log file created: {get_log_file_path()}", force=True)
        log_info('experiment', f"All progress will be logged to this file", force=True)

    def _print_opening_title(self):
        """Print opening title with experiment configuration."""
        cfg = self.cfg

        print("\n" + "="*80)
        print("SPECTRAL TREE INFERENCE - Sub-sampled STDR Experiment")
        print("="*80)

        # Simple experiment (single n_taxa, single seq_len)
        exp_type = "Single Configuration"
        configs = f"n={cfg.get_num_taxa()}, L={cfg.get_sequence_length()}"

        print(f"Experiment Type:    {exp_type}")
        print(f"Configurations:     {configs}")
        print(f"Tree Model:         {cfg.tree.model}")
        print(f"Sequence Model:     {cfg.sequence.model}")
        print(f"Mutation Rate:      μ = {cfg.sequence.params['mutation_rate']}")
        print(f"P-values:           {len(cfg.experiment.p_values)} values from {min(cfg.experiment.p_values):.2e} to {max(cfg.experiment.p_values):.2e}")
        print(f"Bootstrap Reps:     {cfg.experiment.bootstrap_reps}")
        print(f"Display Mode:       {cfg.experiment.display_mode}")
        print(f"Run Name:           {cfg.experiment.run_name}")
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
        # For now, only support single experiment mode
        # TODO: Add sweep support in future
        return self._run_single_experiment()
    
    def _run_single_experiment(self) -> Tuple[str, List[float]]:
        """
        Run a single parameter combination experiment.
        
        Returns:
            Tuple of (run_dir, sign_agreements) - list of single agreement values per p-value
        """
        (fiedler_ref, sign_agreements, partition_agreement_M, partition_agreement_S, 
         dot_products, metrics_dict, reference_partition_quality, 
         sigma2_avg_M_list, sigma2_avg_S_list,
         partition_split_M_list, partition_split_S_list, result_source_list,
         tree, partition_ref) = sweep_for_params(
            cfg=self.cfg,
            n_taxa=self.cfg.get_num_taxa(),
            seq_len=self.cfg.get_sequence_length(),
            run_dir=self.run_dir,
            incremental_save=False,
            show_progress=True  # Show all progress bars for single experiment
        )

        # Final save + plot via helper modules
        save_single_results(
            run_dir=self.run_dir,
            p_values=self.cfg.experiment.p_values,
            sign_agreements=sign_agreements,
            partition_agreement_M=partition_agreement_M,
            partition_agreement_S=partition_agreement_S,
            dot_products=dot_products,
            reference_partition_quality=reference_partition_quality,
            sigma2_avg_M=sigma2_avg_M_list,
            sigma2_avg_S=sigma2_avg_S_list,
            partition_split_M=partition_split_M_list,
            partition_split_S=partition_split_S_list,
            result_source=result_source_list,
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
        
        # Generate tree partition visualizations if partition is available
        if partition_ref is not None and tree is not None:
            log_info('experiment', "Generating tree partition visualizations...", force=True)
            n_taxa = self.cfg.get_num_taxa()
            tree_plot_path = Path(self.run_dir) / f"tree_partition_n{n_taxa}"
            
            # Prepare stats dict for visualization
            stats_dict = {
                'sigma2': reference_partition_quality,
                'partition_split': partition_split_M_list[0] if partition_split_M_list and partition_split_M_list[0] else None
            }
            # Add spectral gap if available
            if metrics_dict and 'spectral_gap_L_M' in metrics_dict and metrics_dict['spectral_gap_L_M']:
                stats_dict['spectral_gap'] = metrics_dict['spectral_gap_L_M'][0][0]  # Mean value
            # Add coherence if available
            if metrics_dict and 'coherence_L_M' in metrics_dict and metrics_dict['coherence_L_M']:
                stats_dict['coherence'] = metrics_dict['coherence_L_M'][0][0]  # Mean value
            
            try:
                plot_tree_with_partition(
                    tree=tree,
                    partition_mask=partition_ref,
                    fiedler_vector=fiedler_ref,
                    output_path=tree_plot_path,
                    title=f"Tree Partition (n={n_taxa})",
                    stats_dict=stats_dict
                )
                plot_combined_tree_and_fiedler(
                    tree=tree,
                    partition_mask=partition_ref,
                    fiedler_vector=fiedler_ref,
                    output_path=tree_plot_path,
                    title=f"Tree Partition (n={n_taxa})",
                    stats_dict=stats_dict
                )
                log_info('experiment', f"Tree partition visualizations saved to: {tree_plot_path}", force=True)
            except Exception as e:
                log_info('experiment', f"Failed to generate tree partition visualizations: {e}", force=True)
        
        log_info('experiment', f"Completed! Results saved to: {self.run_dir}", force=True)
        # Clear cache after single run
        clear_similarity_cache()

        # Close log file
        close_log_file()

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
        all_reference_partition_quality: Dict[int, float] = {}
        all_sigma2_avg_M: Dict[int, List[float]] = {}
        all_sigma2_avg_S: Dict[int, List[float]] = {}
        all_partition_split_M: Dict[int, List[Tuple[int, int] | None]] = {}
        all_partition_split_S: Dict[int, List[Tuple[int, int] | None]] = {}
        all_result_source: Dict[int, List[str]] = {}

        # Create all configuration progress bars upfront
        config_pbars = []
        for i, n_taxa in enumerate(self.cfg.taxa_values):
            pbar = create_config_progress_bar(
                config_idx=i + 1,
                total_configs=len(self.cfg.taxa_values),
                n_taxa=n_taxa,
                seq_len=self.cfg.get_sequence_length(),
                num_p_values=len(self.cfg.experiment.p_values),
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

            (fiedler_ref, sig, partition_agr_M, partition_agr_S, dot_prod, metrics_dict,
             ref_quality, sigma2_M, sigma2_S,
             split_M, split_S, result_src, tree, partition_ref) = sweep_for_params(
                cfg=self.cfg,
                n_taxa=n_taxa,
                seq_len=self.cfg.get_sequence_length(),
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
            all_reference_partition_quality[n_taxa] = ref_quality
            all_sigma2_avg_M[n_taxa] = sigma2_M
            all_sigma2_avg_S[n_taxa] = sigma2_S
            all_partition_split_M[n_taxa] = split_M
            all_partition_split_S[n_taxa] = split_S
            all_result_source[n_taxa] = result_src
            np.save(os.path.join(self.run_dir, f"fiedler_ref_n={n_taxa}.npy"), fiedler_ref)
            
            # Generate tree partition visualizations for this n_taxa if partition is available
            if partition_ref is not None and tree is not None:
                log_info('experiment', f"Generating tree partition visualizations for n={n_taxa}...", force=True)
                tree_plot_path = Path(self.run_dir) / f"tree_partition_n{n_taxa}"
                
                # Prepare stats dict for visualization
                stats_dict = {
                    'sigma2': ref_quality,
                    'partition_split': split_M[0] if split_M and split_M[0] else None
                }
                # Add spectral gap if available
                if metrics_dict and 'spectral_gap_L_M' in metrics_dict and metrics_dict['spectral_gap_L_M']:
                    stats_dict['spectral_gap'] = metrics_dict['spectral_gap_L_M'][0][0]  # Mean value
                # Add coherence if available
                if metrics_dict and 'coherence_L_M' in metrics_dict and metrics_dict['coherence_L_M']:
                    stats_dict['coherence'] = metrics_dict['coherence_L_M'][0][0]  # Mean value
                
                try:
                    plot_tree_with_partition(
                        tree=tree,
                        partition_mask=partition_ref,
                        fiedler_vector=fiedler_ref,
                        output_path=tree_plot_path,
                        title=f"Tree Partition (n={n_taxa})",
                        stats_dict=stats_dict
                    )
                    plot_combined_tree_and_fiedler(
                        tree=tree,
                        partition_mask=partition_ref,
                        fiedler_vector=fiedler_ref,
                        output_path=tree_plot_path,
                        title=f"Tree Partition (n={n_taxa})",
                        stats_dict=stats_dict
                    )
                    log_info('experiment', f"Tree partition visualizations saved to: {tree_plot_path}", force=True)
                except Exception as e:
                    log_info('experiment', f"Failed to generate tree partition visualizations for n={n_taxa}: {e}", force=True)

            # Save incremental taxa-results after each taxa count
            save_taxa_results(
                run_dir=self.run_dir,
                p_values=self.cfg.experiment.p_values,
                all_results=all_results,
                all_metrics=all_metrics,
                all_partition_agreement_M=all_partition_agreement_M,
                all_partition_agreement_S=all_partition_agreement_S,
                all_dot_products=all_dot_products,
                all_reference_partition_quality=all_reference_partition_quality,
                all_sigma2_avg_M=all_sigma2_avg_M,
                all_sigma2_avg_S=all_sigma2_avg_S,
                all_partition_split_M=all_partition_split_M,
                all_partition_split_S=all_partition_split_S,
                all_result_source=all_result_source
            )

        # Clear cache after all runs
        clear_similarity_cache()

        # Close all progress bars
        for pbar in config_pbars:
            pbar.close()
        
        # Final save + plot
        save_taxa_results(
            run_dir=self.run_dir,
            p_values=self.cfg.experiment.p_values,
            all_results=all_results,
            all_metrics=all_metrics,
            all_partition_agreement_M=all_partition_agreement_M,
            all_partition_agreement_S=all_partition_agreement_S,
            all_dot_products=all_dot_products,
            all_reference_partition_quality=all_reference_partition_quality,
            all_sigma2_avg_M=all_sigma2_avg_M,
            all_sigma2_avg_S=all_sigma2_avg_S,
            all_partition_split_M=all_partition_split_M,
            all_partition_split_S=all_partition_split_S,
            all_result_source=all_result_source
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

        # Close log file
        close_log_file()

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
        all_reference_partition_quality: Dict[Tuple[int, int], float] = {}
        all_sigma2_avg_M: Dict[Tuple[int, int], List[float]] = {}
        all_sigma2_avg_S: Dict[Tuple[int, int], List[float]] = {}
        # Grid search not yet supported with new config
        # TODO: Re-add grid search support
        raise NotImplementedError("Grid search not yet supported with new StructuredConfig")

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
                num_p_values=len(self.cfg.experiment.p_values),
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

            (fiedler_ref, sig, partition_agr_M, partition_agr_S, dot_prod, metrics_dict,
             ref_quality, sigma2_M, sigma2_S,
             split_M, split_S, result_src, tree, partition_ref) = sweep_for_params(
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
            all_reference_partition_quality[(n_taxa, seq_len)] = ref_quality
            all_sigma2_avg_M[(n_taxa, seq_len)] = sigma2_M
            all_sigma2_avg_S[(n_taxa, seq_len)] = sigma2_S
            np.save(os.path.join(self.run_dir, f"fiedler_ref_n={n_taxa}_L={seq_len}.npy"), fiedler_ref)

            # Save incremental grid results after each combination
            save_grid_results(
                run_dir=self.run_dir,
                p_values=self.cfg.experiment.p_values,
                all_results=all_results,
                all_metrics=all_metrics,
                all_partition_agreement_M=all_partition_agreement_M,
                all_partition_agreement_S=all_partition_agreement_S,
                all_dot_products=all_dot_products,
                all_reference_partition_quality=all_reference_partition_quality,
                all_sigma2_avg_M=all_sigma2_avg_M,
                all_sigma2_avg_S=all_sigma2_avg_S
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
            p_values=self.cfg.experiment.p_values,
            all_results=all_results,
            all_metrics=all_metrics,
            all_partition_agreement_M=all_partition_agreement_M,
            all_partition_agreement_S=all_partition_agreement_S,
            all_dot_products=all_dot_products,
            all_reference_partition_quality=all_reference_partition_quality,
            all_sigma2_avg_M=all_sigma2_avg_M,
            all_sigma2_avg_S=all_sigma2_avg_S
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

        # Close log file
        close_log_file()

        return self.run_dir, all_results

    # Helper methods for config handling
    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def _make_run_dir(self, run_name: str, base_dir: str = None, subdir_name: str = None) -> str:
        """Create a directory for experiment results.

        Args:
            run_name: Name of the experiment run
            base_dir: Optional base directory (for grid sweeps)
            subdir_name: Optional subdirectory within base_dir (for grid sweeps)

        Returns:
            Absolute path to the run directory
        """
        if base_dir is not None and subdir_name is not None:
            # Grid sweep mode: use provided base_dir/subdir_name
            path = os.path.join(base_dir, subdir_name)
        else:
            # Single experiment mode: create timestamped directory
            ts = time.strftime("%Y%m%d-%H%M%S")
            # Get the directory of the parent of src (sub_sampled_fielder_vec)
            repo_base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            path = os.path.join(repo_base, "results", f"{ts}-{run_name}")

        os.makedirs(path, exist_ok=True)
        return os.path.abspath(path)

    def _save_config(self, cfg: StructuredConfig, run_dir: str) -> None:
        """Save configuration to JSON file in run directory."""
        config_path = os.path.join(run_dir, "config.json")
        cfg.to_json_file(config_path)
