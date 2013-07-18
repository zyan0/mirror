import cv2
import numpy
import cPickle as pickle
import ast
from sklearn.cluster import MiniBatchKMeans
import random
from munkres import Munkres
from kmeans import kMeans
from emd import calcEMD
from common import *
import pyflann
import math
import operator
from vq import get_cosine

result = None
centroids_files = None
centroids = None
flann = None
params = None
sift = None
vq = None

def main():
    fea = pickle.load( open(FEATURE_LIST_LOCATION, 'rb') )
    
    print 'loaded {} features.'.format( len(fea) )

    # assign init to 'k-means++' for better result
    kmeans = MiniBatchKMeans(
        n_clusters = 20000, init='k-means++', max_iter = 100, batch_size = 50000,
        verbose=1, compute_labels=False, random_state=None, tol=0.0,
        max_no_improvement=10, init_size=None, n_init=3, reassignment_ratio=0.01
    )
    
    # kmeans = kMeans(fea = fea, k = 10000, max_iter = 100)
    # kmeans.train()

    # result = kmeans.fit(fea)
    # 
    # pickle.dump(result, open('kmeans.pkl', 'wb'))
    # pickle.dump(result.cluster_centers_, open('centers.pkl', 'wb'))
    
    #save_svmlight_data(numpy.array( fea ), numpy.array( [0 for i in range(len(fea))] ), 'fea.train')

# ./sofia-kmeans --k 10000 --init_type random --opt_type mini_batch_kmeans --mini_batch_size 10000 --iterations 100 --objective_after_init --objective_after_training --training_file /Users/yanzheng/Workspace/mirror/fea.train --model_out /Users/yanzheng/Workspace/mirror/clusters.txt

def clustering_with_flann():
    fea = pickle.load( open(FEATURE_LIST_LOCATION, 'rb') )

    print 'loaded {} features.'.format( len(fea) )

    k = kMeans(fea, 220000, 20)
    del fea
    k.train()

    pickle.dump(k, open(KMEANS_LOCATION, 'wb'))

def save_svmlight_data(data, labels, data_filename, data_folder = ''):
    file = open(data_folder+data_filename,'w')

    for i,x in enumerate(data):
        indexes = x.nonzero()[0]
        values = x[indexes]

        label = '%i'%(labels[i])
        pairs = ['%i:%f'%(indexes[i]+1,values[i]) for i in xrange(len(indexes))]

        sep_line = [label]
        sep_line.extend(pairs)
        sep_line.append('\n')

        line = ' '.join(sep_line)

        file.write(line)

def create_feature_list():
    sift = pickle.load( open(RSIFT_LOCATION, 'rb') )
    
    print 'Finish reading.'
    
    fea = []
    for feature in sift.keys():
        fea.append(ast.literal_eval(feature))

    pickle.dump(fea, open(FEATURE_LIST_LOCATION, 'wb'))

def assign_points():
    sift = pickle.load( open(RSIFT_LOCATION, 'rb') )
    fea = pickle.load( open(FEATURE_LIST_LOCATION, 'rb') )
    kmeans = pickle.load( open(KMEANS_LOCATION, 'rb') )

    print 'Finish reading.'

    idx = kmeans.predict(fea)
    centroids_files = [0 for i in range(220000)]
    counter = 0
    for ins in fea:
        if centroids_files[ idx[counter] ] == 0:
            centroids_files[ idx[counter] ] = []
        centroids_files[ idx[counter] ].extend( sift[ str(ins) ] )
        counter += 1

    pickle.dump(centroids_files, open(CENTROIDS_FILES_LOCATION, 'wb'))

def find_cluster(img_location):
    global flann, params
    
    if flann == None:
        init()
    
    detector = cv2.FeatureDetector_create("SIFT")
    descriptor = cv2.DescriptorExtractor_create("SIFT")

    img = cv2.imread(img_location)
    keypoints = detector.detect(img)
    keypoints = sorted(keypoints, key=lambda x: -x.response)
    keypoints, features = descriptor.compute(img, keypoints[0:10])

    idx, dummy = flann.nn_index( numpy.array(features, dtype=numpy.float64), 1, checks=params["checks"])

    idx = idx.tolist()
    ret = set()
    for i in idx:
        ret = set.union(ret, set(centroids_files[i]) )
    ret = list(ret)
    return ret

FLANN_INDEX_KDTREE = 1
flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 4)

def match(img_location, cluster):
    detector = cv2.FeatureDetector_create("SIFT")
    descriptor = cv2.DescriptorExtractor_create("SIFT")

    img1 = cv2.imread(img_location)
    if img1.shape[1] > 500:
        newx,newy = 500, int(img1.shape[0] * (500.0 / float(img1.shape[1])))
        img1 = cv2.resize(img1, (newx, newy))

    kp1 = detector.detect(img1)
    kp1, desc1 = descriptor.compute(img1, kp1)
    idx1 = flann.nn_index( numpy.array(desc1, dtype=numpy.float64), 1, checks=params["checks"])
    vec1 = {}
    idx1 = idx1[0].tolist()
    for i in idx1:
        try:
            vec1[i] += 1
        except:
            vec1[i] = 1

    distances = {}
    for file_name in cluster:
        try:
            vec2 = vq[file_name]
            distances[file_name] = get_cosine(vec1, vec2)
        except:
            distances[file_name] = -1

    results = cluster
    return results, distances

def save_centroids():
    kmeans = pickle.load( open(KMEANS_LOCATION, 'rb') )
    centroids = kmeans.get_centers()
    pickle.dump(centroids, open(CENTROIDS_LOCATION, 'wb') )

def init():
    global centroids_files, centroids, flann, params, sift, vq
    centroids_files = pickle.load( open(CENTROIDS_FILES_LOCATION, 'rb') )
    centroids = pickle.load( open('centroids.pkl', 'rb') )
    params = pickle.load( open('params.pkl', 'rb') )
    flann = pyflann.FLANN()
    flann.load_index('flann.pkl', numpy.array(centroids, dtype=numpy.float64))
    vq = pickle.load( open('vq.pkl', 'rb') )
    print 'init in images_cluster done.'

if __name__ == '__main__':
    clustering_with_flann()
    save_centroids()
    assign_points()
    # create_feature_list()
    # main()
