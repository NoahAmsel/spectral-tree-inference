% Latent Tree Toolbox (LTT)
% Version 1.1, 7-Sep-2009
%
% Demo scripts:
%   ltt_demo1                   - Demonstrates our algorithm on dataset 1.
%   ltt_demo2                   - Demonstrates our algorithm on dataset 2.
%   ltt_demo3                   - Demonstrates our algorithm on dataset 3.
%   ltt_demo4                   - Demonstrates our algorithm on dataset 4.
%   ltt_script                  - Demonstrates how to call the algorithms.
%   ltt_cv_script               - Demonstrates how to do cross-validation.
%                               
% Algorithms to generate latent trees.
%   bin_forrest                 - Generates binary latent forest.
%   ind_forrest                 - Generates model with single nodes.
%   lcm_forrest                 - Generates latent class model.
%   zhang_forrest               - Interfaces to Zhang's Java code (not included).
%   chowliu                     - Generates Chow-Liu tree.
%   hc_forrest                  - Generates tree using hierarchical clustering.
%                               
% Utility functions.            
%   create_data                 - Creates example datasets.
%   create_tree                 - Creates example trees.
%   aic                         - Calculates Akaike information criterion.
%   bic                         - Calculates Bayesian information criterion.
%   forrest_ll                  - Calculates the log-likelihood for a tree.
%   forrest_ll_fast             - Calculates the log-likelihood for a tree faster.
%   tree2dot                    - Generates dot-file for GraphViz (not included).
%   marginal                    - Calculates marginals for each node.
%   orient                      - Orients binary values of latent variables.
%                               
% Private functions.            
%   private/check_forrest       - Checks the tree data structure.
%   private/data2zhang          - Transforms data to Zhang's data format.
%   private/zhang_read_model    - Transforms Zhang's result to our data format.
%   private/em_automatic        - Runs EM for different number of latent states.
%   private/em_with_restarts    - Runs EM several times with random restarts.
%   private/lcm_lambda          - Calculates the Latent Class Model.
%   private/degreesoffreedom    - Calculates the degrees of freedom of our model.
%   private/mst                 - Calculates minimum spanning tree for chowliu.m.
%   private/mi                  - Calculates mutual information.
%   private/mi_mat              - Calculates mutual information matrix.
%   private/normalizzze         - Normalizes a distribution.
%   private/sample2freq         - Transform a sample into a frequency table.
%   private/sample2dist         - Transform a sample into a distribution.
%   private/data2dist           - Transforms data into a distribution.
%   private/forrest_refine      - Tries to improve all CPTs for a given structure.
%   private/saturated_ll        - Calculates the saturated log-likelihood.
%   private/sample_from_forrest - Samples data from a tree structure.
%   private/sample              - Samples a distribution.
%
% Copyright (C) 2006-2009 by Stefan Harmeling.
