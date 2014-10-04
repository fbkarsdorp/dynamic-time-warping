from _dtw import dtw_distance, dtw_slanted_band
from cluster import *

from joblib import Parallel, delayed
from itertools import combinations, islice
from scipy.spatial.distance import squareform

def pairwise_dtw(data, dtw_fn, n_slices=8, n_jobs=-1):
	
	def slices(iterable, n=8):
		it = iter(iterable)
		item = islice(it, n)
		while item:
			yield item 
			item = list(islice(it, n))

	distances = Parallel(n_jobs=n_jobs)(
		delayed(_pairwise)(s, dtw_fn) for s in slices(combinations(data, 2), n=n_slices))
	return squareform([dist for s in distances for dist in s])

def _pairwise(data, fn):
	return [fn(u, v) for u, v in data]