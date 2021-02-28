import dendropy
from dendropy.interop import raxml
import os.path
import platform

from . import utils
from .reconstruct_tree import ReconstructionMethod, JC_distance_matrix

RECONSTRUCT_TREE_PATH = os.path.abspath(__file__)
RECONSTRUCT_TREE_DIR_PATH = os.path.dirname(RECONSTRUCT_TREE_PATH)

class RAxML(ReconstructionMethod):
    def __call__(self, sequences, taxa_metadata=None, raxml_args = "-T 2 --JC69 -c 1"):
        if not isinstance(sequences, dendropy.DnaCharacterMatrix):
            # data = FastCharacterMatrix(sequences, taxon_namespace=taxon_namespace).to_dendropy()
            data = utils.array2charmatrix(sequences, taxa_metadata=taxa_metadata) 
            data.taxon_namespace = dendropy.TaxonNamespace(taxa_metadata)
        else:
            data = sequences
            
        if platform.system() == 'Windows':
            # Windows version:
            rx = raxml.RaxmlRunner(raxml_path = os.path.join(RECONSTRUCT_TREE_DIR_PATH, 'raxmlHPC-SSE3.exe'))
        elif platform.system() == 'Darwin':
            #MacOS version:
            rx = raxml.RaxmlRunner(raxml_path = os.path.join(RECONSTRUCT_TREE_DIR_PATH,'raxmlHPC-macOS'))
        elif platform.system() == 'Linux':
            #Linux version
            rx = raxml.RaxmlRunner(raxml_path = os.path.join(RECONSTRUCT_TREE_DIR_PATH,'raxmlHPC-SSE3-linux'))

        tree = rx.estimate_tree(char_matrix=data, raxml_args=[raxml_args])
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
        raxml_path = os.path.join(RECONSTRUCT_TREE_DIR_PATH, 'raxmlHPC-SSE3.exe')
    elif platform.system() == 'Darwin':
        #MacOS version:
        raxml_path = os.path.join(RECONSTRUCT_TREE_DIR_PATH,'raxmlHPC-macOS')
    elif platform.system() == 'Linux':
        #Linux version
        raxml_path = os.path.join(RECONSTRUCT_TREE_DIR_PATH,'raxmlHPC-SSE3-linux')

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