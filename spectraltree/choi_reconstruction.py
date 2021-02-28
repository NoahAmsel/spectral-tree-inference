import numpy as np
import oct2py
import scipy.sparse

from . import utils
from .reconstruct_tree import ReconstructionMethod, JC_distance_matrix

class RG(ReconstructionMethod):

    def __call__(self, observations, taxa_metadata=None):        
        return self.estimate_tree_topology(observations, taxa_metadata)
    def estimate_tree_topology(self, observations, taxa_metadata=None,bifurcating=False):
        import oct2py
        from oct2py import octave
        octave.addpath('./spectraltree/ChoilatentTree/')
        oc = oct2py.Oct2Py()
        num_taxa = observations.shape[0]
        
        ## shouldn't this be a DistanceReconstructionMethod?

        D = JC_distance_matrix(observations)
        #adj_mat = oc.feval("./spectraltree/ChoilatentTree/toolbox/RGb.m",observations+1,0)
        adj_mat = oc.feval("./spectraltree/ChoilatentTree/toolbox/RGb.m",D,1,observations.shape[1])
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
        adj_mat = oc.feval("./spectraltree/ChoilatentTree/toolbox/CLRGb.m",observations+1,0)
        adj_mat = np.array(scipy.sparse.csr_matrix.todense(adj_mat))
        tree_CLRG = utils.adjacency_matrix_to_tree(adj_mat,num_taxa,taxa_metadata)
        return tree_CLRG
    def __repr__(self):
        return "CLRG"