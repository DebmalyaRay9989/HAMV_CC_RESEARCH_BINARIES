import numpy as np
from src.hamvcc.metrics import external_metrics, internal_metrics, pairwise_stability

def test_external_perfect():
    y=np.array([0,0,1,1]); assert external_metrics(y,y)['ari']==1.0; assert external_metrics(y,y)['nmi']==1.0

def test_internal_keys():
    X=np.array([[0,0],[0,1],[5,5],[5,6]],float); y=np.array([0,0,1,1]); m=internal_metrics(X,y); assert set(m)=={'silhouette','calinski_harabasz','davies_bouldin'}

def test_pairwise():
    y=np.array([0,0,1,1]); assert pairwise_stability([y,y])==1.0
