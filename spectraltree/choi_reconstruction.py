import os.path
import scipy.sparse

import numpy as np
import oct2py

from . import utils
from .reconstruct_tree import ReconstructionMethod
from .raxml_reconstruction import SPECTRALTREE_LIB_PATH
from .similarities import JC_distance_matrix

SPECTRALTREE_CHOI_PATH = os.path.join(SPECTRALTREE_LIB_PATH, "ChoilatentTree")

class RG(ReconstructionMethod):

    def __call__(self, observations, taxa_metadata=None):        
        return self.estimate_tree_topology(observations, taxa_metadata)
    def estimate_tree_topology(self, observations, taxa_metadata=None,bifurcating=False):
        import oct2py
        from oct2py import octave
        octave.addpath(SPECTRALTREE_CHOI_PATH)
        oc = oct2py.Oct2Py()
        num_taxa = observations.shape[0]
        
        ## shouldn't this be a DistanceReconstructionMethod?

        D = JC_distance_matrix(observations)
        #adj_mat = oc.feval("./spectraltree/ChoilatentTree/toolbox/RGb.m",observations+1,0)
        adj_mat = oc.feval(os.path.join(SPECTRALTREE_CHOI_PATH, "toolbox", "RGb.m"),D,1,observations.shape[1], verbose=False)
        adj_mat = np.array(scipy.sparse.csr_matrix.todense(adj_mat))
        tree_RG = utils.adjacency_matrix_to_tree(adj_mat,num_taxa,taxa_metadata)
        return tree_RG
    def __repr__(self):
        return "RG"

class CLRG(ReconstructionMethod):

    def __call__(self, observations, taxa_metadata=None):        
        return self.estimate_tree_topology(observations, taxa_metadata)
    def estimate_tree_topology(self, observations, taxa_metadata=None,bifurcating=False):
        import oct2py
        from oct2py import octave

        octave.addpath('./spectraltree/ChoilatentTree/')
        oc = oct2py.Oct2Py()
        num_taxa = observations.shape[0]
        adj_mat = oc.feval("./spectraltree/ChoilatentTree/toolbox/CLRGb.m",observations+1,0, verbose=False)
        adj_mat = np.array(scipy.sparse.csr_matrix.todense(adj_mat))
        tree_CLRG = utils.adjacency_matrix_to_tree(adj_mat,num_taxa,taxa_metadata)
        return tree_CLRG
    def __repr__(self):
        return "CLRG"