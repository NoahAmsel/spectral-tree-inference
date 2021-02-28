import oct2py

from . import utils
from .reconstruct_tree import ReconstructionMethod

class Forrest(ReconstructionMethod):
    def __call__(self, observations, taxa_metadata=None):        
        return self.estimate_tree_topology(observations, taxa_metadata)
    def estimate_tree_topology(self, observations, taxa_metadata=None,bifurcating=False):
        oct2py.octave.addpath('./spectraltree/ltt-1.4/')
        oc = oct2py.Oct2Py()
        num_taxa = observations.shape[0]
        adj_mat = oc.feval("./spectraltree/ltt-1.4/bin_forrest_wrapper.m",observations+1,4)
        tree_Forrest = utils.adjacency_matrix_to_tree(adj_mat,num_taxa,taxa_metadata)
        return tree_Forrest
    def __repr__(self):
        return "Forrest"