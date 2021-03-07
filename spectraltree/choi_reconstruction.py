import os.path
import scipy.sparse

import numpy as np
import oct2py

from .utils import adjacency_matrix_to_tree
from .reconstruct_tree import ReconstructionMethod
from .raxml_reconstruction import SPECTRALTREE_LIB_PATH

SPECTRALTREE_CHOI_PATH = os.path.join(SPECTRALTREE_LIB_PATH, "ChoilatentTree")

class RG(ReconstructionMethod):
    def __init__(self, distance_metric):
        self.distance_metric = distance_metric

    def __call__(self, observations, taxa_metadata=None):
        oct2py.octave.addpath(SPECTRALTREE_CHOI_PATH)
        oc = oct2py.Oct2Py()
        num_taxa = observations.shape[0]

        D = self.distance_metric(observations)
        #adj_mat = oc.feval("./spectraltree/ChoilatentTree/toolbox/RGb.m",observations+1,0)
        adj_mat = oc.feval(os.path.join(SPECTRALTREE_CHOI_PATH, "toolbox", "RGb.m"),D,1,observations.shape[1], verbose=False)
        adj_mat = np.array(scipy.sparse.csr_matrix.todense(adj_mat))
        tree_RG = adjacency_matrix_to_tree(adj_mat,num_taxa,taxa_metadata)
        return tree_RG

    def __repr__(self):
        return "RG"

class CLRG(ReconstructionMethod):
    def __call__(self, observations, taxa_metadata=None):        
        octave.addpath('./spectraltree/ChoilatentTree/')
        oc = oct2py.Oct2Py()
        num_taxa = observations.shape[0]
        adj_mat = oc.feval("./spectraltree/ChoilatentTree/toolbox/CLRGb.m",observations+1,0, verbose=False)
        adj_mat = np.array(scipy.sparse.csr_matrix.todense(adj_mat))
        tree_CLRG = adjacency_matrix_to_tree(adj_mat,num_taxa,taxa_metadata)
        return tree_CLRG

    def __repr__(self):
        return "CLRG"