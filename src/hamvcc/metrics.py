import itertools, numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score

def internal_metrics(X, labels):
    u=np.unique(labels)
    if len(u)<2 or len(u)>=len(labels):
        return {"silhouette":np.nan,"calinski_harabasz":np.nan,"davies_bouldin":np.nan}
    return {"silhouette":float(silhouette_score(X,labels)),"calinski_harabasz":float(calinski_harabasz_score(X,labels)),"davies_bouldin":float(davies_bouldin_score(X,labels))}

def external_metrics(y, labels):
    if y is None: return {"ari":np.nan,"nmi":np.nan}
    return {"ari":float(adjusted_rand_score(y,labels)),"nmi":float(normalized_mutual_info_score(y,labels))}

def pairwise_stability(label_list):
    if len(label_list)<2:return np.nan
    vals=[adjusted_rand_score(label_list[i],label_list[j]) for i,j in itertools.combinations(range(len(label_list)),2)]
    return float(np.mean(vals))
