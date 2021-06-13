import spectraltree
import spectraltree.compare_methods as compare_methods

tree_list = [spectraltree.balanced_binary(32)]
jc = spectraltree.Jukes_Cantor(num_classes=2)
Ns = [100,200,300]
mutation_rates = [jc.p2t(0.9)]
snj = spectraltree.SpectralNeighborJoining(spectraltree.JC_similarity_matrix) 
nj = spectraltree.NeighborJoining(spectraltree.JC_similarity_matrix) 
RG = spectraltree.RG(spectraltree.JC_distance_matrix)
CLRG = spectraltree.CLRG()
methods = [nj,snj,RG,CLRG]
num_reps = 1
results = compare_methods.experiment(tree_list = tree_list, 
sequence_model = jc, Ns = Ns, methods=methods, mutation_rates = mutation_rates, 
reps_per_tree=num_reps,savepath='balanced_binary_m_512.pkl',folder = './data/',overwrite=True)
df = compare_methods.results2frame(results)
print(df)
a =1