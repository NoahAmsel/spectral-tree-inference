import os.path

import oct2py

from . import utils
from .reconstruct_tree import ReconstructionMethod
from .raxml_reconstruction import SPECTRALTREE_LIB_PATH

SPECTRALTREE_LLT_PATH = os.path.join(SPECTRALTREE_LIB_PATH, "ltt-1.4")

class Forrest(ReconstructionMethod):
    def __call__(self, observations, taxa_metadata=None):
        oct2py.octave.addpath(SPECTRALTREE_LLT_PATH)
        oc = oct2py.Oct2Py()
        num_taxa = observations.shape[0]
        adj_mat = oc.feval(os.path.join(SPECTRALTREE_LLT_PATH, "bin_forrest_wrapper.m"),observations+1,4,verbose=False)
        tree_Forrest = utils.adjacency_matrix_to_tree(adj_mat,num_taxa,taxa_metadata)
        return tree_Forrest

    def __repr__(self):
        return "Forrest"