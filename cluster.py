import pandas as pd 

from scipy.cluster.hierarchy import ClusterNode
from HACluster import Clusterer
from dtw import dtw_distance
from functools import partial

def to_tree(data, Z, averaging_fn):
    n = Z.shape[0] + 1
    d = [None] * (n * 2 - 1)
    for i in range(n):
    	d[i] = DtwClusterNode(i, data[i])
    nd = None 
    for i in range(n-1):
    	fi = int(Z[i, 0])
    	fj = int(Z[i, 1])
    	assert fi <= i + n 
    	assert fj <= i + n
    	left = d[fi]
    	right = d[fj]
    	prototype = averaging_fn(left.prototype, right.prototype, 
    		                     left.count, right.count)
    	nd = DtwClusterNode(i + n, prototype, left, right, Z[i, 2])
        assert Z[i, 3] == nd.count
        d[n + i] = nd
    return nd, d

class DtwClusterNode(ClusterNode):
	def __init__(self, id, prototype, left=None, right=None, dist=0, count=1):
		ClusterNode.__init__(self, id, left, right, dist, count)
		self.prototype = prototype

def cuttree(tree, t):
	queue = set([tree])
	clusters = set()
	while queue:
		current_node = queue.pop()
		if current_node.dist > t:
			queue.add(current_node.get_right())
			queue.add(current_node.get_left())
		else:
			clusters.add(current_node)
	return clusters

def cluster(data, Z, averaging_fn, dtw_fn, cuttree=0):
	averaging_fn = partial(averaging_fn, dtw_function=dtw_fn)
	tree, nodes = to_tree(data, Z, averaging_fn)
	if cuttree > 0:
		clusters = cuttree(tree, cuttree)
	return clusters

