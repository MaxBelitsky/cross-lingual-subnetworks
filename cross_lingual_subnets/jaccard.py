import numpy as np
import itertools

# Function to compute Jaccard similarity
def jaccard_similarity(mask1, mask2):
    set1 = set(zip(*np.where(mask1 == 1)))
    set2 = set(zip(*np.where(mask2 == 1)))
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def compute_jaccard_similarities(masks):
    jaccard_results = {}
    
    for (key1, mask1), (key2, mask2) in itertools.combinations(masks.items(), 2):
        similarity = jaccard_similarity(mask1, mask2)
        if key1 not in jaccard_results:
            jaccard_results[key1] = {}
        if key2 not in jaccard_results:
            jaccard_results[key2] = {}
        jaccard_results[key1][key2] = similarity
        jaccard_results[key2][key1] = similarity
    
    return jaccard_results