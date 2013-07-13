import cv2
import numpy
import cPickle as pickle
import ast
from sklearn.cluster import MiniBatchKMeans

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6

def main():
    sift = pickle.load( open('rsift.pkl', 'rb') )
    
    print 'Finish reading.'
    
    fea = []
    for feature in sift.keys():
        fea.append(ast.literal_eval(feature))
    
    kmeans = MiniBatchKMeans(
        n_clusters = 70000, init='k-means++', max_iter = 100, batch_size = 10000,
        verbose=1, compute_labels=True, random_state=None, tol=0.0,
        max_no_improvement=10, init_size=None, n_init=3, reassignment_ratio=0.01
    )

    result = kmeans.fit(fea)
    
    pickle.dump(result, open('kmeans.pkl', 'wb'))
    pickle.dump(result.cluster_centers_, open('centers.pkl', 'wb'))

    idx = result.labels_.tolist()
    centroids_tags = [0 for i in range(70000)]
    counter = 0
    for ins in fea:
        if centroids_tags[ idx[counter] ] == 0:
            centroids_tags[ idx[counter] ] = []
        centroids_tags[ idx[counter] ].extend( sift[ str(ins) ] )
        counter += 1
    
    pickle.dump(centroids_tags, open('centroids_tags.pkl', 'wb'))
    

if __name__ == '__main__':
    main()