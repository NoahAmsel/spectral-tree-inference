import os.path
import platform
import subprocess

import dendropy
from dendropy.interop import raxml
import numpy as np

from . import utils
from .reconstruct_tree import ReconstructionMethod
from .similarities import JC_distance_matrix

SPECTRALTREE_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
SPECTRALTREE_LIB_PATH = os.path.join(SPECTRALTREE_DIR_PATH, "libs")
SPECTRALTREE_RAXML_PATH = os.path.join(SPECTRALTREE_LIB_PATH, "raxmlHPC_bin")

class RAxML(ReconstructionMethod):
    """Reconstructs a binary tree using the RAxML program.

    See here https://cme.h-its.org/exelixis/web/software/raxml/.

    Args:
        raxml_args: string of command line arguments to pass to the RAxML executable.
    """

    def __init__(self, raxml_args="-T 2 --JC69 -c 1"):
        self.raxml_args = [raxml_args]

        if platform.system() == 'Windows':
            # Windows version:
            raxml_path = os.path.join(SPECTRALTREE_RAXML_PATH, 'raxmlHPC-SSE3.exe')
        elif platform.system() == 'Darwin':
            #MacOS version:
            raxml_path = os.path.join(SPECTRALTREE_RAXML_PATH,'raxmlHPC-macOS')
        elif platform.system() == 'Linux':
            #Linux version
            raxml_path = os.path.join(SPECTRALTREE_RAXML_PATH,'raxmlHPC-SSE3-linux')
        else:
            raise OSError(f"Cannot identify operating system {platform.system()}.")

        self._rx = raxml.RaxmlRunner(raxml_path=raxml_path)

    def __call__(self, sequences, taxa_metadata=None):
        if not isinstance(sequences, dendropy.DnaCharacterMatrix):
            # data = FastCharacterMatrix(sequences, taxon_namespace=taxon_namespace).to_dendropy()
            data = utils.array2charmatrix(sequences, taxa_metadata=taxa_metadata) 
            data.taxon_namespace = dendropy.TaxonNamespace(taxa_metadata)
        else:
            data = sequences

        tree = self._rx.estimate_tree(data, raxml_args=self.raxml_args)
        tree.is_rooted = False
        if taxa_metadata != None:
            tree.taxon_namespace = taxa_metadata.taxon_namespace
        return tree

    def __repr__(self):
        return "RAxML"

def raxml_gamma_corrected_distance_matrix(observations, taxa_metadata):
    charmatrix = utils.array2charmatrix(observations, taxa_metadata)
    tempfile_path = "tempfile54321.tree"
    outfile_path = "temp.phylib"
    charmatrix.write(path=tempfile_path, schema="phylip")

    if platform.system() == 'Windows':
        # Windows version:
        raxml_path = os.path.join(SPECTRALTREE_RAXML_PATH, 'raxmlHPC-SSE3.exe')
    elif platform.system() == 'Darwin':
        #MacOS version:
        raxml_path = os.path.join(SPECTRALTREE_RAXML_PATH,'raxmlHPC-macOS')
    elif platform.system() == 'Linux':
        #Linux version
        raxml_path = os.path.join(SPECTRALTREE_RAXML_PATH,'raxmlHPC-SSE3-linux')

    subprocess.call([raxml_path, '-f', 'x', '-T', '4', '-p', '12345', '-s', tempfile_path, '-m', 'GTRGAMMA', '-n', outfile_path], stdout=subprocess.DEVNULL)
    distances_path = f"RAxML_distances.{outfile_path}"
    info_path = f"RAxML_info.{outfile_path}"
    parsimony_path = f"RAxML_parsimonyTree.{outfile_path}"

    m, n = observations.shape
    distance_matrix = np.zeros((m, m))
    with open(distances_path) as f:
        for line in f:
            T1, T2, distance = line.split()
            distance_matrix[taxa_metadata[T1], taxa_metadata[T2]] = float(distance) 
            distance_matrix[taxa_metadata[T2], taxa_metadata[T1]] = float(distance) 

    os.remove(tempfile_path)
    os.remove(distances_path)
    os.remove(info_path)
    os.remove(parsimony_path)

    return distance_matrix