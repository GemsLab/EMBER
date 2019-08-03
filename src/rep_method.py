import numpy as np, networkx as nx
import scipy.sparse

class RepMethod():
	def __init__(self, align_info = None, p=None, k=10, max_layer=None, method="smf", alpha = 0.1, sampling_method="random", sampling_prob="proportional", num_buckets = None, use_other_features = False, normalize = False, gamma = 1, gammaattr = 1, implicit_factorization = True, landmarks = None, use_landmarks = False, use_attr_dist = True, binning_features = ["degree"], concatenate = False):
		self.method = method #representation learning method
		self.sampling_method = sampling_method #sample according to this distribution
		self.sampling_prob = sampling_prob #sample top or proportional
		self.p = p #sample p points
		self.k = k #sample 2k log N points
		self.max_layer = max_layer #furthest hop distance up to which to compare neighbors
		self.alpha = alpha #discount factor for higher layers
		self.num_buckets = num_buckets #number of buckets to split node feature values into #CURRENTLY BASE OF LOG SCALE
		self.align_info = align_info #alignment information (known alignments, alignment similarities)
		self.use_other_features = use_other_features
		self.normalize = normalize
		self.gamma = gamma
		self.gammaattr = gammaattr
		self.implicit_factorization = implicit_factorization
		self.landmarks = landmarks
		self.landmark_indices = None
		self.use_landmarks = False #use hard coded landmarks
		self.use_attr_dist = use_attr_dist
		self.binning_features = binning_features
		self.concatenate = concatenate


class Graph():
	#Undirected, unweighted
	def __init__(self, adj, weighted = False, directed = False, signed = False, num_buckets=None, node_labels = None, edge_labels = None, graph_label = None, node_attributes = None, true_alignments = None, attribute_class_sizes = None, node_features = None, graph_id = None, neighbor_list = None, max_id = None):
		self.graph_id = graph_id
		self.G_adj = adj #adjacency matrix
		self.N = self.G_adj.shape[0] #number of nodes
		self.weighted = weighted
		self.directed = directed
		self.signed = signed
		self.max_id = max_id
		self.neighbor_list = neighbor_list
		self.num_buckets = num_buckets #how many buckets to break node features into

		if node_features is None:
			#TODO think about how to do this better
			self.node_features = {}#{"degree": self.node_degrees}
		else:
			self.node_features = node_features
		
		self.max_features = {}
		for feature in self.node_features:
			self.max_features[feature] = np.max(self.node_features[feature])

		self.node_labels = node_labels
		self.edge_labels = edge_labels
		self.graph_label = graph_label
		self.node_attributes = node_attributes #N x A matrix, where N is # of nodes, and A is # of attributes
		self.kneighbors = None #dict of k-hop neighbors for each node
		self.true_alignments = true_alignments #dict of true alignments, if this graph is a combination of multiple graphs
		
		self.attribute_class_sizes = attribute_class_sizes
		self.distances = None

		#Count the proportion of attributes that take on each possible value, per attribute
		if self.node_attributes is not None and self.attribute_class_sizes is not None:
			self.attribute_class_sizes = dict()
			for attribute in range(self.node_attributes.shape[1]):
				values, counts = np.unique(self.node_attributes[:,attribute], return_counts = True)
				if attribute not in self.attribute_class_sizes:
					self.attribute_class_sizes[attribute] = dict()
				for val in range(len(values)):
					self.attribute_class_sizes[attribute][values[val]] = float(counts[val]) / self.N

	def set_node_features(self, node_features):
		self.node_features = node_features
		for feature in self.node_features: #TODO hacky to handle this case separately
			self.max_features[feature] = np.max(self.node_features[feature])


	def compute_node_features(self, features_to_compute):
		if self.weighted:
			weight = "weight"
		else:
			weight = None
		if scipy.sparse.issparse(self.G_adj):
			if self.directed:
				nx_graph = nx.from_scipy_sparse_matrix(self.G_adj, create_using=nx.DiGraph())
			else:
				nx_graph = nx.from_scipy_sparse_matrix(self.G_adj)
		else:
			if self.directed:
				nx_graph = nx.from_numpy_matrix(self.G_adj, create_using=nx.DiGraph())
			else:
				nx_graph = nx.from_numpy_matrix(self.G_adj)
		new_node_features = self.node_features

		########## Do not use ##########
		if self.signed:
			pos_graph = nx.from_numpy_matrix(self.G_adj[self.G_adj > 0])
			neg_graph = nx.from_numpy_matrix(np.abs(self.G_adj[self.G_adj < 0]))
			abs_graph = nx.from_numpy_matrix(np.abs(self.G_adj))

		if self.directed and "outdegree" in features_to_compute:
			if self.signed:
				outdegrees = abs_graph.out_degree(abs_graph.nodes(), weight = weight)
			else:
				outdegrees = nx_graph.out_degree(nx_graph.nodes(), weight = weight)
			new_node_features["outdegree"] = outdegrees
		if self.directed and "indegree" in features_to_compute:
			if self.signed:
				indegrees = abs_graph.in_degree(abs_graph.nodes(), weight = weight)
			else:
				indegrees = nx_graph.in_degree(nx_graph.nodes(), weight = weight) 
			new_node_features["indegree"] = indegrees
		if "degree" in features_to_compute:
			if self.weighted:
				weight = "weight"
			else:
				weight = "None"
			if self.signed:
				total_degrees = abs_graph.degree(abs_graph.nodes(), weight = weight)
			else:
				total_degrees = nx_graph.degree(nx_graph.nodes(), weight = weight)
				# print "total degrees: ", nx_graph.degree(nx_graph.nodes())
			new_node_features["degree"] = total_degrees

		if self.signed and "positive_outdegree" in features_to_compute:
			positive_outdegrees = pos_graph.out_degree(pos_graph.nodes(), weight = weight)
			new_node_features["positive_outdegree"] = positive_outdegrees
		if self.signed and "positive_indegree" in features_to_compute:
			positive_indegrees = pos_graph.in_degree(pos_graph.nodes(), weight = weight)
			new_node_features["positive_indegree"] = positive_indegrees
		if self.signed and "positive_degree" in features_to_compute:
			positive_degrees = pos_graph.degree(pos_graph.nodes(), weight = weight)
			new_node_features["positive_degree"] = positive_degrees

		if self.signed and "negative_outdegree" in features_to_compute:
			negative_outdegrees = neg_graph.out_degree(neg_graph.nodes(), weight = weight)
			new_node_features["negative_outdegree"] = negative_outdegrees
		if self.signed and "negative_indegree" in features_to_compute:
			negative_indegrees = neg_graph.in_degree(neg_graph.nodes(), weight = weight)
			new_node_features["negative_indegree"] = negative_indegrees
		if self.signed and "negative_degree" in features_to_compute:
			negative_degrees = neg_graph.degree(neg_graph.nodes(), weight = weight)
			new_node_features["negative_degree"] = negative_degrees

		for feature in new_node_features:
			new_node_features[feature] = [new_node_features[feature][x] for x in nx_graph.nodes()]  


		if not self.signed and "pagerank" in features_to_compute:
			pr = nx.pagerank(nx.from_numpy_matrix(combined_graph.G_adj))
			pr_list = np.asarray([pr[node] for node in pr.keys()])
			new_node_features["pagerank"] = pr_list
			
		if "attributes" in features_to_compute: #TODO bin more than one attribute
			new_node_features["attributes"] = combined_graph.node_attributes

		if "edge_label" in features_to_compute: #can't set this for each node, since it depends on what edge it's connected to...just set max feature
			if self.edge_labels is not None:
				self.max_features["edge_label"] = np.max(self.edge_labels)
			else:
				raise ValueError("no edge labels to bin")
		#print "features to compute:", features_to_compute
		self.set_node_features(new_node_features)

	def normalize_node_features(self):
		normalized_features_dict = dict()
		for feature in self.node_features:
			normalized_features = self.node_features[feature]
			if np.min(normalized_features) < 0: #shift so no negative values
				normalized_features += abs(np.min(normalized_features))
			#scale so no feature values less than 1 (for logarithmic binning)
			if np.min(normalized_features) < 1:
				normalized_features /= np.min(normalized_features[normalized_features != 0])
				if np.max(normalized_features) == 1: #e.g. binary features
					normalized_features += 1
				normalized_features[normalized_features == 0] = 1 #set 0 values to 1--bin them in first bucket (smallest values)
			normalized_features_dict[feature] = normalized_features
		self.set_node_features(normalized_features_dict)


class Node():
	def __init__(self, node_id, centrality = None, attributes = None, parent = None, path_weight = 0, edge_label = None):
		self.node_id = node_id
		self.centrality = centrality
		self.attributes = attributes
		self.parent = parent
		self.path_weight = path_weight
		self.edge_label = edge_label

	def set_centrality(self, centrality):
		self.centrality = centrality



def get_delimiter(input_file_path):
	delimiter = " "
	if ".csv" in input_file_path:
		delimiter = ","
	elif ".tsv" in input_file_path:
		delimiter = "\t"
	else:
		sys.exit('Format not supported.')

	return delimiter

def write_embedding(rep, output_file_path, nodes_to_embed):
	N, K = rep.shape

	fOut = open(output_file_path, 'w')
	fOut.write(str(N) + ' ' + str(K) + '\n')

	for i in range(N):
		cur_line = ' '.join([str(np.round(ii, 6)) for ii in rep[i,:]])
		fOut.write(str(nodes_to_embed[i]) + ' ' + cur_line + '\n')

	fOut.close()

	return
