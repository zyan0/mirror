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

def main():
    fea = pickle.load( open(FEATURE_LIST_LOCATION, 'rb') )
    
    print 'loaded {} features.'.format( len(fea) )

    # assign init to 'k-means++' for better result
    # kmeans = MiniBatchKMeans(
    #     n_clusters = 20000, init='k-means++', max_iter = 100, batch_size = 50000,
    #     verbose=1, compute_labels=False, random_state=None, tol=0.0,
    #     max_no_improvement=10, init_size=None, n_init=3, reassignment_ratio=0.01
    # )
    
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

    k = kMeans(fea, 150000, 10)
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
    centroids_files = [0 for i in range(150000)]
    counter = 0
    for ins in fea:
        if centroids_files[ idx[counter] ] == 0:
            centroids_files[ idx[counter] ] = []
        centroids_files[ idx[counter] ].extend( sift[ str(ins) ] )
        counter += 1

    pickle.dump(centroids_files, open(CENTROIDS_FILES_LOCATION, 'wb'))

result = None
centroids_files = None
centroids = None

def find_cluster(img_location):
    detector = cv2.FeatureDetector_create("SIFT")
    descriptor = cv2.DescriptorExtractor_create("SIFT")

    img = cv2.imread(img_location)
    keypoints = detector.detect(img)
    keypoints = sorted(keypoints, key=lambda x: -x.response)
    keypoints, features = descriptor.compute(img, keypoints[0:10])
    features = features.tolist()

    idx = result.predict(features)
    ret = set()
    for i in idx:
        ret = set.union(ret, set(centroids_files[i]) )

    return list(ret)

FLANN_INDEX_KDTREE = 1
flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 4)

def match(img_location, cluster):
    detector = cv2.FeatureDetector_create("SIFT")
    descriptor = cv2.DescriptorExtractor_create("SIFT")
    
    img = cv2.imread(img_location)
    kp1 = detector.detect(img)
    kp1 = sorted(kp1, key=lambda x: -x.response)
    kp1, desc1 = descriptor.compute(img, kp1[0:50])
    
    distances = {}
    for file_name in cluster:
        img = cv2.imread( 'static/mirflickr/' + file_name)
        kp2 = detector.detect(img)
        kp2 = sorted(kp2, key=lambda x: -x.response)
        kp2, desc2 = descriptor.compute(img, kp2[0:50])
    
        distances[file_name] = _match( desc1, desc2, kp1, kp2 )
    
    # distances = {}
    # for file_name in cluster:
    #     distances[file_name] = calcEMD(img_location, 'static/mirflickr/' + file_name)
        
    
    # results = sorted(cluster, key = lambda x: distances[x])
    results = cluster
    return results, distances

m = Munkres()

# def _match(desc1, desc2):
#     global m
#     distance = 0
#     matrix = [[0 for i in range(10)] for j in range(10)]
#     
#     for i in range(10):
#         for j in range(10):
#             if matrix[j][i] != 0:
#                 matrix[i][j] = matrix[j][i]
#                 continue
#             try:
#                 matrix[i][j] = numpy.linalg.norm( desc1[i] - desc2[j] )
#             except:
#                 matrix[i][j] = 100000
# 
#     indexes = m.compute(matrix)
#     for row, column in indexes:
#         distance += matrix[row][column]
# 
#     return distance

def _match(desc1, desc2, kp1, kp2):
    norm = cv2.NORM_L2
    # matcher = cv2.BFMatcher(norm)
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2)
    p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
    return len(p1)

def filter_matches(kp1, kp2, matches, ratio = 0.8):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = numpy.float32([kp.pt for kp in mkp1])
    p2 = numpy.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs

def save_centroids():
    kmeans = pickle.load( open(KMEANS_LOCATION, 'rb') )
    centroids = kmeans.get_centers()
    pickle.dump(centroids, open(CENTROIDS_LOCATION, 'wb') )

if __name__ == '__main__':
    # create_feature_list()
    # clustering_with_flann()
    # main()
    # assign_points()
    save_centroids()
else:
    result = pickle.load( open(KMEANS_LOCATION, 'rb') )
    centroids_files = pickle.load( open(CENTROIDS_FILES_LOCATION, 'rb') )
    centroids = pickle.load( open('centroids.pkl', 'rb') )
