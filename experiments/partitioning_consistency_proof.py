###########################################################
# This code generates a tree T such that the Schur complimant 
# of the Laplacian of T is the similarity matrix.
###############################################################

import numpy as np
import sys, os, platform
import dendropy
import igraph
import copy
import itertools

# Generate an unrooted tree to work with
G_orig = igraph.Graph.Tree(127,2)
G_orig.delete_edges([(0,1),(0,2)])
G_orig.add_edge(1,2)
G_orig.delete_vertices([0])
p = np.random.rand(5)
G_orig.es["weight"] = -np.log(p)


leafs = [i for i in range(len(G_orig.vs)) if len(G_orig.neighbors(i)) == 1] # list of all the leafs
# current_G - A tree with the same structure as G_orig. The weights are updated so "resulting_G" will correspond to the similarity of "current_G"
current_G = copy.deepcopy(G_orig)
current_G.vs["name"] = current_G.vs.indices
# resuting_G - A graph. at each iteration we add a node, and keep the relrvant schurs compement equal to the similarity
resultingG = igraph.Graph.Weighted_Adjacency( current_G.shortest_paths_dijkstra(source=leafs, target = leafs, weights = "weight"), mode = igraph.ADJ_UNDIRECTED , attr = "weight")
resultingG.vs["name"] = leafs
# targetG - A full graph of the leafs of G_orig. It's adjacency matrix is the similarity.
targetG = copy.deepcopy(resultingG)

active_set = copy.deepcopy(leafs) # The set of all vertices 
added_nodes = copy.deepcopy(leafs) # hold all the vertices names of "resulting_G"

## Supporting functions
def myPlot(G):
    visual_style = {}
    visual_style["vertex_label"] = G.vs["name"]
    visual_style["edge_label"] = [ int(i*1000)/1000 for i in G.es["weight"]]
    igraph.plot(G, **visual_style)

def find_parant_in_active_set(current_G, active_set, added_nodes, avoid_nodes = []):
    for node in current_G.vs:
        if not(node.index in added_nodes):
            children = []
            for i in current_G.neighbors(node.index):
                if (i in active_set) and (not(i in avoid_nodes)):
                    children.append(i)
            if len(children) >= 2:
                return children, node

def schur_compliment(mat, inds):
    # Copute the schur's complement of a matrix
    # mat - matrix of shape nXn 
    # inds - list of indices. The indices by which to compute the compliment
    n = mat.shape[0]
    not_inds = list(set(range(n))-set(inds))
    A = mat[not_inds,:]
    A = A[:,not_inds]
    B = mat[not_inds,:]
    B = B[:,inds]
    C = mat[inds,:]
    C = C[:,not_inds]
    D = mat[inds,:]
    D = D[:, inds]
    return A-B@np.linalg.inv(D)@C
def computeExpLaplacianG(G):
    A = np.array(G.get_adjacency(attribute = "weight").data)
    A[A!=0] = np.exp(-A[A!=0])
    return np.diag(np.sum(A,1)) - A

while len(added_nodes) < len(G_orig.vs):
    ############
    # Step 1 : updating resulting_G
    ############
    # finding an appropriate node to add: not in added_nodes, but his "children" are in active_set
    children, new_node = find_parant_in_active_set(current_G, active_set, added_nodes)
    # we now add node
    resultingG.add_vertex(name = new_node.index)
    new_node_idx = resultingG.vs.select(lambda vertex: vertex["name"] == new_node.index )[0].index
    active_set_idx = resultingG.vs.select(lambda vertex: vertex["name"] in active_set)
    d = 0
    #correct weight for children of added node
    for i in children:
        active_set.remove(i)
        child = resultingG.vs.select(lambda vertex: vertex["name"] == i )[0].index #child's index in resultingG
        resultingG.delete_edges(resultingG.es.select( _between =([child],active_set_idx)))
        cur_weight = current_G.es.select( _between = ([i],[new_node.index]) )[0]["weight"]
        resultingG.add_edge(child,new_node_idx)["weight"] = cur_weight
        d+=np.exp(-cur_weight)

    # correct weights for all active nodes
    for i in active_set:
        i_idx = resultingG.vs.select(lambda vertex: vertex["name"] == i )[0].index
        cur_weight = current_G.shortest_paths_dijkstra(i,new_node.index, weights = "weight")[0][0]
        resultingG.add_edge(i_idx,new_node_idx)["weight"]  = cur_weight
        d+=np.exp(-cur_weight)
    
    # compute "d" and update all weights connectd to the new node
    d = -np.log(d)
    for i in resultingG.es.select(lambda edge: edge.target == new_node_idx):
        i["weight"] = i["weight"] + d
    
    #correct weights for all pairs of old active nodes
    for i,j in itertools.combinations(active_set,2):
        resultingG_i_idx = resultingG.vs.select(lambda vertex: vertex["name"] == i )[0].index
        resultingG_j_idx = resultingG.vs.select(lambda vertex: vertex["name"] == j )[0].index
        s_i_new = resultingG.es.select( _between = ([resultingG_i_idx],[new_node_idx]) )[0]["weight"]
        s_i_new = s_i_new-d
        s_j_new = resultingG.es.select( _between = ([resultingG_j_idx],[new_node_idx]) )[0]["weight"]
        s_j_new = s_j_new-d
        curr_s_i_j = resultingG.es.select( _between = ([resultingG_i_idx],[resultingG_j_idx]) )[0]["weight"]
        resultingG.es.select( _between = ([resultingG_i_idx],[resultingG_j_idx]) )["weight"] = -np.log( np.exp(-curr_s_i_j) - np.exp(-s_i_new - s_j_new))
    # add new node to active nodes
    active_set.append(new_node.index)
    added_nodes.append(new_node.index)
    #if we finished the reconstraction we do not need step 2
    if len(active_set) == 1:
        break
    ############
    # Step 2 : updating current_G
    ############
    current_G_active_set = copy.deepcopy(active_set)
    current_G_added_nodes = copy.deepcopy(added_nodes)
    while not(len(current_G_active_set) <= 2):
        children, node = find_parant_in_active_set(current_G, current_G_active_set, current_G_added_nodes, avoid_nodes=[new_node.index])
        #computing the "correction coeff" for the edges between node and it's childrenn
        weight_node_new_node = current_G.shortest_paths_dijkstra(node.index, new_node.index, weights = "weight")[0][0]
        alpha = -np.log(np.sqrt(1-np.exp(-weight_node_new_node)**2))
        #updating the weights to children (alpha)
        current_G.es.select( _between = ([node.index], [children[0]]) )[0]["weight"] = alpha + current_G.es.select( _between = ([node.index], [children[0]]) )[0]["weight"]
        current_G.es.select( _between = ([node.index], [children[1]]) )[0]["weight"] = alpha + current_G.es.select( _between = ([node.index], [children[1]]) )[0]["weight"]
        #updating weight to parent (1/alpha)
        s = set(current_G.neighbors(node)) - set(children)
        parent = list(s)[0]
        current_G.es.select( _between = ([node.index], [parent]) )[0]["weight"] = current_G.es.select( _between = ([node.index], [parent]) )[0]["weight"] - alpha
        #updating the current_G_active_set
        current_G_active_set.remove(children[0])
        current_G_active_set.remove(children[1])
        current_G_active_set.append(node.index)
        current_G_added_nodes.append(node.index)

    current_G.es.select( _between = ([node.index], [new_node.index]))[0]["weight"] = current_G.es.select( _between = ([node.index], [new_node.index]))[0]["weight"] + d

targetG_GL = computeExpLaplacianG(targetG)
retulting_GL = schur_compliment(computeExpLaplacianG(resultingG),range(len(leafs),len(G_orig.vs)))
print("TargetG:")
print(targetG_GL)
print("Resulting_G:")
print(retulting_GL)
print ("norm of diff:", np.linalg.norm(targetG_GL-retulting_GL))
print("Hurray!")
