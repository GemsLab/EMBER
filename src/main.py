import argparse
import numpy as np, scipy as sp, networkx as nx
import math, time, os, sys, random
from collections import deque
from sklearn.decomposition import NMF, DictionaryLearning
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


from numpy import genfromtxt
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn import svm
from sklearn import linear_model
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from scipy.sparse.linalg import svds, eigs

from itertools import cycle

from rep_method import *



def init_node_neighbors(node, graph, rep_method, verbose = False):
	neighbors = np.nonzero(graph.G_adj[node])[-1].tolist() ###
	if len(neighbors) == 0: #disconnected node
		kneighbors = {0: {node: create_node(node, graph, rep_method)}, 1: {}}
	else:
		if type(neighbors[0]) is list:
			neighbors = neighbors[0]

		orig_node = create_node(node, graph, rep_method)
		distinct_neighbors = set(neighbors) - set([node])
		distinct_neighbor_nodes = {}
		for neighbor_id in distinct_neighbors:
			distinct_neighbor_nodes[neighbor_id] = create_node(neighbor_id, graph, rep_method, parent = node)
		kneighbors = {0: {node: orig_node}, 1: distinct_neighbor_nodes} #in case of self loops--node itself is not a 1 hop neighbor
	return kneighbors


#Input: adjacency matrix of graph
#Output: dictionary of dictionaries: for each node, dictionary containing {node : {layer_num : [list of neighbors]}}
#        dictionary {node ID: degree}
#def get_khop_neighbors(G_adj, until_layer = None):
def get_khop_neighbors(graph, rep_method=None, verbose = False, nodes_to_embed = None):
	#If we want to learn embeddings for only a smaller number of nodes, only collect their full neighbor info
	#Otherwise, learn embeddings for all nodes and get all their neighborhood information
	if nodes_to_embed is None:
		nodes_to_embed = range(graph.N)

	if rep_method.max_layer is None:
		rep_method.max_layer = graph.N #num_nodes #TODO Don't need this line, just sanity prevent infinite loop

	kneighbors_dict = {}

	#only 0-hop neighbor of a node is itself
	#neighbors of a node have nonzero connections to it in adj matrix
	#for node in range(num_nodes):
	for node in nodes_to_embed:
		kneighbors_dict[node] = init_node_neighbors(node, graph, rep_method, verbose)

	#For each node, keep track of neighbors we've already seen
	all_neighbors = {}

	#for node in range(graph.N):
	for node in nodes_to_embed:
		all_neighbors[node] = set([node])
		all_neighbors[node] = all_neighbors[node].union(kneighbors_dict[node][1])

	#Recursively compute neighbors in k
	#Neighbors of k-1 hop neighbors, unless we've already seen them before
	current_layer = 2 #need to at least consider neighbors
	while True:
		if rep_method.max_layer is not None and current_layer > rep_method.max_layer: break
		reached_max_layer = True #whether we've reached the graph diameter

		#nodes we need to get further information on
		#because they're nodes we want to embed or nodes in the neighborhood of ones we want to embed
		#nodes_to_expand = kneighbors_dict.keys() 

		for i in nodes_to_embed:
			#All neighbors k-1 hops away
			neighbors_prevhop = kneighbors_dict[i][current_layer - 1]
			
			#khop_neighbors = set()
			khop_neighbors = dict()
			#Add neighbors of each k-1 hop neighbors
			for n in neighbors_prevhop:
				#Make sure we've collected neighbor information on this node
				if n not in kneighbors_dict:
					kneighbors_dict[n] = init_node_neighbors(n, graph, rep_method, verbose)

				#Add this neighbor's neighbors to nodes we must consider
				neighbors_of_n = kneighbors_dict[n][1]
				for neighbor2nd in neighbors_of_n: 
					#neighbor2nd_node = Node(node_id = neighbor, parent = n)

					#don't add node if already seen (k-hop neighbor, but reachable at shorter hop distance)
					#TODO what about if lower path weight?
					#if neighbor2nd not in all_neighbors[i]:
					if neighbor2nd not in all_neighbors[i]:
						#khop_neighbors.add(neighbor2nd)
						#khop_neighbors.add(create_node(neighbor2nd, graph, rep_method, parent = n))
						edge_label = None
						if "edge_label" in rep_method.binning_features:
							edge_label = graph.edge_labels[n, neighbor2nd]
						khop_neighbors[neighbor2nd] = create_node(neighbor2nd, graph, rep_method, parent = n, edge_label = edge_label)
						all_neighbors[i].add(neighbor2nd)

			#See if we've added any more neighbors
			#If so, we may not have reached the max layer: we have to see if these nodes have neighbors
			if len(khop_neighbors) > 0:
				reached_max_layer = False 

			#add neighbors
			kneighbors_dict[i][current_layer] = khop_neighbors #k-hop neighbors must be at least k hops away

		if reached_max_layer:
			break
		else:
			current_layer += 1

	return kneighbors_dict

#Implementation of k hop neighbors with BFS
def get_khop_neighbors_bfs(graph, rep_method):
	kneighbors_dict = {}
	degrees = {} #also store degree of each node so we don't have to recompute later
	for i in range(graph.N): #perform BFS from each node to get its number of neighbors
		kneighbors_dict[i] = {}
		current_node = Node(node_id = i)
		visited = {i : 0}
		queue = [current_node]
		while len(queue) > 0:
			#pop from queue
			current_node = queue[0]
			del queue[0]

			#we're at whatever layer the current node is at
			current_layer = visited[current_node.node_id]

			if rep_method.max_layer is not None and current_layer > rep_method.max_layer:
				break #move on to next node

			#prepare to store current neighbors at current layer
			if current_layer not in kneighbors_dict[i]:
				kneighbors_dict[i][current_layer] = {} #[]

			#store in dict of neighbors
			centrality = dict()
			for feature in rep_method.binning_features:
				if feature == "edge_label":
					if current_node.edge_label is not None: #TODO hacky to treat this case specially
						centrality["edge_label"] = current_node.edge_label #this will have been set below, except for root node
				else:
					centrality[feature] = graph.node_features[feature][current_node.node_id]
			current_node.set_centrality(centrality)
			kneighbors_dict[i][current_layer][current_node.node_id] = current_node

			#visit all its unvisited neighbors
			for neighbor in np.nonzero(graph.G_adj[current_node.node_id])[-1].tolist():
				if neighbor not in visited:
					visited[neighbor] = current_layer + 1 #put them at one hop farther away
					edge_label = None
					if "edge_label" in rep_method.binning_features:
						edge_label = graph.edge_labels[current_node.node_id, neighbor]
					neighbor_node = Node(node_id = neighbor, parent = current_node.node_id, edge_label = edge_label)
					queue.append(neighbor_node) #have to visit them later

	return kneighbors_dict

def create_node(node_id, graph, rep_method, parent = None, edge_label = None):
	node = Node(node_id = node_id, parent = parent, edge_label = edge_label)
	centrality = dict()
	for feature in rep_method.binning_features:
		if feature == "edge_label": #have to treat this separately TODO hacky
			centrality["edge_label"] = edge_label
		else:
			centrality[feature] = graph.node_features[feature][node.node_id]
	node.set_centrality(centrality)
	return node


#Get the combined degree/other feature sequence for a given node
def get_combined_feature_sequence(graph, rep_method, current_node, gadj = None):
	features = []
	for feature in rep_method.binning_features:
		if rep_method.num_buckets is not None:
			n_buckets = int(math.log(graph.max_features[feature], rep_method.num_buckets) + 1)
		else:
			n_buckets = int(graph.max_features[feature]) + 1

		#print("%s will be split into %d buckets" % (feature, n_buckets))
		features.append([0] * n_buckets)

	if graph.signed:
		negative_features = features[:]

	if graph.weighted:
		if gadj is None:
			print "converting sparse array to dict for faster indexing..."
			rows, cols = graph.G_adj.nonzero()
			data = np.ravel(graph.G_adj[rows, cols])
			A = zip(rows, cols)
			gadj = dict(zip(A, data)) #graph.G_adj.todense() sparse indexing is slow
	#weight_time = 0
	for layer in graph.khop_neighbors[current_node].keys():
		khop = graph.khop_neighbors[current_node][layer]
		#kn_count = 0
		for khop_neighbor in khop:
			#kn_count += 1
			kn_node = khop[khop_neighbor]
			weight = 1
			
			if graph.weighted:
				current_layer = layer
				node = kn_node
				while current_layer > 0:
					#weight *= 2#graph.G_adj[node.parent, node.node_id]
					weight *= gadj[(node.parent, node.node_id)]
					node = graph.khop_neighbors[current_node][current_layer - 1][node.parent]
					current_layer -= 1


			try:
				for i in range(len(rep_method.binning_features)):
					feature = rep_method.binning_features[i]
					if feature == "edge_label":
						if kn_node.edge_label is not None:
							node_feature = kn_node.edge_label
						else:
							pass #print "no edge label for this node object"
					else:
						node_feature = kn_node.centrality[feature] #TODO can only handle centralities rn

					if rep_method.num_buckets is not None:
						bucket_index = int(math.log(node_feature, rep_method.num_buckets))
					else:
						bucket_index = int(node_feature)

					#if we observe larger values than seen in the buckets, put them in last bucket
					if graph.signed and weight < 0:
						negative_features[i][min(bucket_index, len(negative_features[i]) - 1)] += (rep_method.alpha ** layer) * abs(weight)
					else:
						features[i][min(bucket_index, len(features[i]) - 1)] += (rep_method.alpha ** layer) * weight
			except Exception as e:
				print "Exception:", e
				print("Node %d has %s value %d and will not contribute to feature distribution" % (khop_neighbor, feature, node_feature))
		#print("# kneighbors at layer %d: %d" % (layer, kn_count))
	combined_feature_vector = features[0]
	#print("spent %f time computing path weight" % weight_time)
	
	#TODO got slightly higher signed accuracy before changed to allow binning multiple features?
	for feature_vector in features[1:]:
		combined_feature_vector += feature_vector
	
	if graph.signed:
		for neg_feature_vector in negative_features:
			combined_feature_vector += neg_feature_vector
	
	return combined_feature_vector

#Get structural features for nodes in a graph based on degree sequences of neighbors
#Input: adjacency matrix of graph
#Output: nxD feature matrix
def get_features(graph, rep_method, verbose = True, nodes_to_embed = None):
	graph.khop_neighbors = get_khop_neighbors(graph, rep_method, nodes_to_embed = nodes_to_embed)

	if nodes_to_embed is None:
		nodes_to_embed = range(graph.N)
	


	deg_seqs = {}
	feature_matrix = None

	gadj = None
	if graph.weighted:
		rows, cols = graph.G_adj.nonzero()
		data = np.ravel(graph.G_adj[rows, cols])
		A = zip(rows, cols)
		gadj = dict(zip(A, data)) #graph.G_adj.todense() sparse indexing is slow

	counter = 0
	for n in nodes_to_embed:
		counter += 1

		combined_feature_sequence = get_combined_feature_sequence(graph, rep_method, n, gadj = gadj)

		if feature_matrix is None:
			feature_matrix = combined_feature_sequence
		else:
			feature_matrix = np.vstack((feature_matrix, combined_feature_sequence))

	return feature_matrix

#Input: two vectors of the same length
#Optional: tuple of (same length) vectors of node attributes for corresponding nodes
#Output: number between 0 and 1 representing their similarity
#Computes ||sum_k(deg_k(v1)) - sum_k(deg_k(v2))||_2^2 (possibly he terms in the sum weighted by alpha)
def compute_similarity(graph, rep_method, vec1, vec2, node_attributes = None, node_indices = None, attribute_class_sizes = None):
	_sigma_struc = 1 #RBF Kernel hyperparameter
	_sigma_attr = 100

	dist = rep_method.gamma * np.linalg.norm(vec1 - vec2)
	#print "structural distance: ", dist
	#NOTE: may want to only compare if attributes are same or different
	if graph.node_attributes is not None and rep_method.use_attr_dist: 
		attr_dist = 0
		if graph.attribute_class_sizes is not None: #weight by size of class (higher similarity for smaller class)
			for attr in range(graph.node_attributes.shape[1]):
				if graph.node_attributes[node_indices[0]] == graph.node_attributes[node_indices[1]]:
					#the smaller the class the more significant the node similarity, so the higher the dist
					attr_dist += (1 - attribute_class_sizes[attr][graph.node_attributes[node_indices[0],attr]]) 
		else:
			if node_indices is not None: #TODO figure out how to work around if hard coded landmarks
				attr_dist = np.sum(graph.node_attributes[node_indices[0]] != graph.node_attributes[node_indices[1]])
		dist += rep_method.gammaattr * attr_dist
	#print "combined distance: ", dist
	return np.exp(-dist)

#Sample landmark nodes (to compute all pairwise similarities to in Nystrom approx)
#Input: graph (in case want to do leverage score sampling, etc.), number of samples
#Output: np array of node IDs
#def get_sample_nodes(G_adj, p, sampling_method = "pagerank", sampling_prob = "proportional", verbose = True):
def get_sample_nodes(graph, rep_method, verbose = True, nodes_to_embed = None):
	#Sample uniformly at random
	#OPTION sample according to levesrage scores or some more sophisticated measure
	if verbose:
		print "embed: ", nodes_to_embed
	if nodes_to_embed is not None:
		num_nodes = len(nodes_to_embed)

	else:
		nodes_to_embed = np.arange(graph.N)
		num_nodes = graph.N
	
	if rep_method.sampling_method == "random":
		sample = np.random.permutation(np.asarray(nodes_to_embed))[:rep_method.p]
	else:
		if verbose:
			print("sampling nodes with probability proportional to their %s..." % rep_method.sampling_method)
		if rep_method.sampling_method == "degree":
			sampling_dist = np.asarray(graph.node_features["degree"])
		elif rep_method.sampling_method == "leverage":
			#Sample nodes with probability proportional to their leverage score (in adj matrix)
			#TODO fast approximation to leverage scores, use local cumulative coherence instead?
			
			leverage_scores = np.diag(np.dot(graph.G_adj.T, graph.G_adj)) #https://arxiv.org/pdf/1507.05870.pdf

			leverage_scores = np.sqrt(leverage_scores) #https://arxiv.org/pdf/1507.05870.pdf intro??
			sampling_dist = leverage_scores
		else:
			temp = []

			if rep_method.sampling_method == "pagerank":
#                                sampling_dist = np.asarray(graph.node_features["pagerank"])
				pageranks = nx.pagerank(nx_graph)
                                for fk in range(max(pageranks.keys())+1):
                                        if fk in pageranks.keys():
                                                temp.append(pageranks[fk])
                                        else:
                                                temp.append(0.0)
                                sampling_dist = np.asarray(temp)
			elif rep_method.sampling_method == "betweenness":
				bc = nx.betweenness_centrality(nx_graph)
				sampling_dist = np.asarray(bc.values())
			elif rep_method.sampling_method == "closeness":
				closeness = nx.closeness_centrality(nx_graph)
				sampling_dist = np.asarray(closeness.values())
			else:
				raise ValueError(("sampling method %s not implemented yet" % rep_method.sampling_method))
	
# 		print sampling_dist, len(sampling_dist)
# 		print np.asarray(nodes_to_embed)
		sampling_dist = sampling_dist[np.asarray([e for e in nodes_to_embed])] #only consider sampling distribution over nodes we can choose
# 		print sampling_dist, len(sampling_dist)
		prob_dist = sampling_dist / float(np.sum(sampling_dist))
		prob_dist = np.ravel(prob_dist)# prob_dist.reshape((-1,))

		#select nodes with probability proportional to their sampling distribution
		if rep_method.sampling_prob == "proportional":

			#ASSUMES NODES ARE GIVEN IN ASCENDING ORDER OF ID
			sample = np.random.choice(np.asarray(nodes_to_embed), size = rep_method.p, p = prob_dist)
		
		#select p top nodes according to sampling distribution
		else:
			sample = np.argsort(prob_dist)[prob_dist.size - rep_method.p:]

	return sample

#Get dimensionality of learned representations
#Related to rank of similarity matrix approximations
#Input: graph
#Output: dimensionality of representations to learn (tied into rank of similarity matrix approximation)
def get_feature_dimensionality(graph, rep_method, verbose = True, nodes_to_embed = None):
	if nodes_to_embed is None:
		num_nodes = graph.N
	else:
		num_nodes = len(nodes_to_embed)
	p = int(rep_method.k*math.log(num_nodes, 2)) #k*log(n), where k = 10
	if verbose:
		print "feature dimensionality is ", min(p, num_nodes)
	rep_method.p = min(p,num_nodes)
	return rep_method.p #don't return larger dimensionality than # of nodes


#Input: nxD feature matrix, p (dimension of representations to learn)
#align_info: dictionary of alignments
#Output: nxp learned representations
#def get_representations(G_adj, align_info = None, until_layer = None, p = None, k = 10, method = "xnetmf", sampling_method = "random", sampling_prob = "proportional"):
def get_representations(graph, rep_method, verbose = True, return_rep_method = False, nodes_to_embed = None):
	if rep_method.method == "degseqs" and rep_method.p is not None:
		graph.max_degree = rep_method.p #so that we can get degseqs of an arbitrary length (basically by padding with 0s)

	if verbose:
		print "Until layer: ", rep_method.max_layer
		print "sampling method: ", rep_method.sampling_method
		print "attribute class sizes: ", graph.attribute_class_sizes
	
	#Explicitly featurize the nodes
	feature_matrix = get_features(graph, rep_method, verbose, nodes_to_embed)
	if graph.directed == False:
		feature_matrix = np.hstack((feature_matrix, feature_matrix))
	if graph.directed:
		# print "got outdegree degseqs, now getting indegree..."
		indegree_graph = Graph(graph.G_adj.T, weighted = graph.weighted, signed = graph.signed, directed = graph.directed, node_features = graph.node_features)
		indegree_feature_matrix = get_features(indegree_graph, rep_method, verbose, nodes_to_embed)
# 		print feature_matrix.shape
# 		print indegree_feature_matrix.shape
		feature_matrix = np.hstack((np.dot(feature_matrix, 1), np.dot(indegree_feature_matrix, 17)))

	if verbose:
		print "Dimensionality in explicit feature space: ", feature_matrix.shape
	
	if rep_method.method == "degseqs": #want explicit featurization of nodes by deg sequences of neighbors
		if verbose:
			print "returning explicit features"
		return feature_matrix

	
	#Get landmark nodes
	if rep_method.p is None:
		rep_method.p = get_feature_dimensionality(graph, rep_method, verbose = verbose) #k*log(n), where k = 10
	elif rep_method.p > graph.N or (nodes_to_embed is not None and rep_method.p > len(nodes_to_embed)): 
# 		print "Warning: dimensionality greater than number of nodes to embed. Reducing to n"
		if (nodes_to_embed is not None and rep_method.p > len(nodes_to_embed)):
			rep_method.p = len(nodes_to_embed)
		else:
			rep_method.p = graph.N

	
	landmarks = get_sample_nodes(graph, rep_method, verbose = verbose, nodes_to_embed = nodes_to_embed)
	if verbose:
		print "landmark IDs: ", landmarks
		print feature_matrix.shape


	rep_method.landmarks = feature_matrix[landmarks]
	rep_method.landmark_indices = landmarks


	
	if nodes_to_embed is None:
		nodes_to_embed = range(graph.N)
		num_nodes = graph.N
	else:
		num_nodes = len(nodes_to_embed)

	C = np.zeros((num_nodes,rep_method.p))
	for graph_node in range(num_nodes):#nodes_to_embed:
		for landmark_node in range(rep_method.p):
			if rep_method.method == "struc2vec":
				#use struc2vec similarities
				C[graph_node,landmark_node] = struc2vec_sim_matrix[graph_node, landmarks[landmark_node]]
			else:
				if rep_method.landmarks is not None: #landmarks are hard coded as actual features of landmark nodes
					landmark_node_features = rep_method.landmarks[landmark_node]
				else:
					landmark_node_features = feature_matrix[landmarks[landmark_node]] #landmarks are indices of landmark nodes
				C[graph_node,landmark_node] = compute_similarity(graph, rep_method, feature_matrix[graph_node], 
					landmark_node_features, graph.node_attributes, (graph_node, landmark_node), attribute_class_sizes = graph.attribute_class_sizes)


	if rep_method.landmarks is None or not rep_method.use_landmarks:
		W_pinv = np.linalg.pinv(C[landmarks]) #make W have same entries for aligning nodes?
	else: #we have hard coded landmarks, compute their pairwise similarities
		W = np.zeros((rep_method.landmarks.shape[0], rep_method.landmarks.shape[0]))
		for i in range(W.shape[0]):
			for j in range(W.shape[1]): #NOTE based on structure only for now
				W[i,j] = compute_similarity(graph, rep_method, rep_method.landmarks[i], rep_method.landmarks[j])
		W_pinv = np.linalg.pinv(W)

    #Nystrom approximation: S'  = CWC^T /approx S, the full nxn similarity matrix
    #xnetmf: rank-p factorization of similarity matrix /approx skipgram:
    #YZ /approx S ==> Y as learned by struc2vec
    
    #Take rank-k factorization of S' to approximate skipgram embeddings
    #But Csqrt(W) * sqrt(W)C^T = S', and Csqrt(W) is rank k
    #So take Csqrt(W) as our representation
	if rep_method.implicit_factorization:
		# if np.linalg.matrix_rank(C[landmarks]) == C[landmarks].shape[0]:
		# 	# print ':)'
		# 	reprsn = np.dot(C, sp.linalg.sqrtm(W_pinv))
		# else:
		if verbose:
			print("W is singular: rank %d vs size %d" % (np.linalg.matrix_rank(C[landmarks]), C[landmarks].shape[0])), C[landmarks]
		U,X,V = np.linalg.svd(W_pinv)
		# dim = U_f.shape[1]
		# K = min(dim, 24)
		# print K
		# print X_f
		# U = U_f[:K]
		sqrtW_substitute = np.dot(U, np.diag(np.sqrt(X)))
		reprsn = np.dot(C, sqrtW_substitute) #make representation have same entries for aligning nodes?

	else:
		Sapprox = np.dot(np.dot(C, W_pinv), C.T)
		print "created similarity matrix"

		#doesn't work since low rank approx can have negative values
		#nmf = NMF(n_components = W_pinv.shape[0], init='random', random_state=0)
		nmf = DictionaryLearning(n_components = W_pinv.shape[0], random_state=0)
		reprsn = nmf.fit_transform(Sapprox)	
		print "...and factorized it"
		reprsn = reprsn.todense()

	if rep_method.normalize:
# 		print("normalizing...")
		norms = np.linalg.norm(reprsn, axis = 1).reshape((reprsn.shape[0],1))
		norms[norms == 0] = 1 #TODO figure out why getting zeros representation
		reprsn = reprsn / norms

	if return_rep_method:
		return reprsn, rep_method
	return reprsn

def find_nodes_to_embed(lookup_file, delimiter = "\t"):
	result = []
	fIn = open(lookup_file, 'r')
	lines = fIn.readlines()
	for line in lines:
		line = line.strip('\r\n')
		parts = line.split(delimiter)
		ID = int(parts[0])
		result.append(ID)

	return result

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
	'''
	Parses the arguments.
	'''
	print '?'
	parser = argparse.ArgumentParser(description="EMBER: Inferring professional roles in email networks.")

	parser.add_argument('--input', nargs='?', default='../graph/karate.tsv', help='Input graph file path')

	parser.add_argument('--lookup', nargs='?', default='../graph/karate_lookup.tsv', help='Input file with nodes to embed')

	parser.add_argument('--output', nargs='?', default='../emb/karate_emb.tsv', help='Embedding file path')

	parser.add_argument('--weighted', type=str2bool, default=True, help='Consider weights in the graph')

	parser.add_argument('--directed', type=str2bool, default=True, help='Consider directionality in the graph')

	parser.add_argument('--p', type=int, default=128, help='Number of landmarks, also the embedding dimension')

	parser.add_argument('--alpha', type=int, default=0.1, help='Decay factor')

	parser.add_argument('--gamma', type=int, default=1.0, help='Similarity constant')

	return parser.parse_args()


if __name__ == "__main__":

	args = parse_args()

	graph_file = args.input
	lookup_file = args.lookup
	embed_file = args.output
	weighted = args.weighted
	directed = args.directed
	p = args.p
	alpha = args.alpha
	gamma = args.gamma

	print '----------------------------------'
	print '[Input graph file] ' + graph_file
	print '[Input lookup file] ' + lookup_file
	print '[Output embedding file] ' + embed_file
	print '[Weight?] ' + str(weighted)
	print '[Direction?] ' + str(directed)
	print '[p] ' + str(p)
	print '[alpha] ' + str(alpha)
	print '[gamma] ' + str(gamma)
	print '----------------------------------'


	delimiter = "\t"
	if directed:
		nx_graph = nx.read_edgelist(graph_file, nodetype = int, create_using=nx.DiGraph(), comments="%", delimiter=delimiter, data=(('weight',int),))
	else:
		nx_graph = nx.read_edgelist(graph_file, nodetype = int, create_using=nx.Graph(), comments="%", delimiter=delimiter, data=(('weight',int),))


	
	max_id = max(nx_graph.nodes())
	print 'max_id: ' + str(max_id)
	adj_matrix = nx.adjacency_matrix(nx_graph, nodelist=range(max_id+1))#.todense()
	if not weighted:
		adj_matrix[np.nonzero(adj_matrix)] = 1

	graph = Graph(adj_matrix, weighted = weighted, directed = directed)
	
		
	nodes_to_embed = find_nodes_to_embed(lookup_file)
	
	rep_method = RepMethod(method = "xnetmf", max_layer = 2, num_buckets = 2, p = p, alpha = alpha, gamma = gamma, sampling_method = "degree", sampling_prob = "top", normalize = True)
	graph.compute_node_features(rep_method.binning_features)
	representations = get_representations(graph, rep_method, verbose = False, nodes_to_embed = nodes_to_embed)

	
	write_embedding(representations, embed_file, nodes_to_embed)

