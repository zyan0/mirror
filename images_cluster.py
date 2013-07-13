import cv2
import numpy
import cPickle as pickle
import ast
from sklearn.cluster import MiniBatchKMeans
import random

def main():
    sift = pickle.load( open('rsift.pkl', 'rb') )
    
    print 'Finish reading.'
    
    fea = []
    for feature in sift.keys():
        fea.append(ast.literal_eval(feature))
    del sift

    kmeans = MiniBatchKMeans(
        n_clusters = 10000, init='random', max_iter = 100, batch_size = 10000,
        verbose=1, compute_labels=True, random_state=None, tol=0.0,
        max_no_improvement=10, init_size=None, n_init=3, reassignment_ratio=0.01
    )

    result = kmeans.fit(fea)

    pickle.dump(result, open('kmeans.pkl', 'wb'))
    pickle.dump(result.cluster_centers_, open('centers.pkl', 'wb'))

def assign_points():
    sift = pickle.load( open('rsift.pkl', 'rb') )
    result = pickle.load( open('kmeans.pkl', 'rb') )
    
    print 'Finish reading.'
    
    fea = []
    for feature in sift.keys():
        fea.append(ast.literal_eval(feature))
    
    idx = result.labels_.tolist()
    centroids_tags = [0 for i in range(8000)]
    counter = 0
    for ins in fea:
        if centroids_tags[ idx[counter] ] == 0:
            centroids_tags[ idx[counter] ] = []
        centroids_tags[ idx[counter] ].extend( sift[ str(ins) ] )
        counter += 1
    
    pickle.dump(centroids_tags, open('centroids_tags.pkl', 'wb'))
    

if __name__ == '__main__':
    main()
    # assign_points()
