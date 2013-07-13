from numpy import vstack, array
from numpy.random import rand
import numpy
from scipy.cluster.vq import kmeans, vq, whiten
import ast
import sys
import os
import cPickle as pickle
import cv2
from munkres import Munkres

REVERSED_SIFT_STORE_LOCATION = 'rsift.pkl'

def reversed_extract_features():
    detector = cv2.FeatureDetector_create("SIFT")
    descriptor = cv2.DescriptorExtractor_create("SIFT")

    sift = {}

    os.chdir('static/mirflickr')
    count = 0
    for file_name in os.listdir("."):
        print file_name
        if file_name.endswith(".jpg"):
            img = cv2.imread(file_name)
            keypoints = detector.detect(img)
            keypoints = sorted(keypoints, key=lambda x: -x.response)
            keypoints, features = descriptor.compute(img, keypoints[0:30])
            try:
                for fea in features:
                    try:
                        sift[str(fea.tolist())].append(file_name)
                    except:
                        sift[str(fea.tolist())] = [file_name]
            except:
                pass
            # count += 1
            # if count > 1000:
            #     break

    os.chdir('..')
    os.chdir('..')
    pickle.dump(sift, open(REVERSED_SIFT_STORE_LOCATION, 'wb'))

CENTER_STORE_LOCATION = 'centers.pkl'

def features_clustering():
    sift = pickle.load(open(REVERSED_SIFT_STORE_LOCATION, 'rb'))
    
    fea = []
    for feature in sift.keys():
        fea.append(ast.literal_eval(feature))
        
    fea = numpy.asarray(fea)
    # whitened = whiten(fea)
    
    print '- k-Means begin -'

    codebook, distortion = kmeans(fea, 1000, 20)

    pickle.dump(codebook, open(CENTER_STORE_LOCATION, 'wb'))

def assign_features_to_centers():
    sift = pickle.load(open(REVERSED_SIFT_STORE_LOCATION, 'rb'))
    centroids = pickle.load(open(CENTER_STORE_LOCATION, 'rb'))

    fea = []
    for feature in sift.keys():
        fea.append(ast.literal_eval(feature))
    fea = numpy.asarray(fea)
    
    idx, _ = vq(fea, centroids)
    
    centroids_tags = [0 for i in range(1000)]
    counter = 0
    for ins in fea.tolist():
        if centroids_tags[ idx[counter] ] == 0:
            centroids_tags[ idx[counter] ] = []
        centroids_tags[ idx[counter] ].extend( sift[ str(ins) ] )
        counter += 1
    
    pickle.dump(centroids_tags, open('centroids_tags.pkl', 'wb'))

def match(img_location, sift):
    img = cv2.imread(img_location)
    
    detector = cv2.FeatureDetector_create("SIFT")
    descriptor = cv2.DescriptorExtractor_create("SIFT")
    
    keypoints = detector.detect(img)
    keypoints = sorted(keypoints, key=lambda x: -x.response)
    keypoints, img_features = descriptor.compute(img, keypoints[0:50])
    
    centroids = pickle.load(open(CENTER_STORE_LOCATION, 'rb'))
    idx, _ = vq(img_features, centroids)
    
    centroids_tags = pickle.load(open('centroids_tags.pkl', 'rb'))
    count = [0 for i in range(50)]
    for i in idx:
        count[i] += 1
    idx = [i[0] for i in sorted(enumerate(count), key=lambda x:x[1])]

    results = []
    for i in range(3):
        results.extend( centroids_tags[idx[-i]] )

    results = list( set(results) )

    m = Munkres()
    
    distance = {}
    for filename in results:
        distance[filename] = 0
        fea = sift[filename]
        matrix = []
        for i in range(5):
            matrix.append([])
            for j in range(5):
                try:
                    matrix[i].append( numpy.linalg.norm( array(img_features[i]) - array(fea[j]) )  )
                except:
                    matrix[i].append( 10000 )
        indexes = m.compute(matrix)
        for row, column in indexes:
            distance[filename] += matrix[row][column]

    results = sorted(results, key = lambda x: distance[x])

    return results, distance

if __name__ == '__main__':
    # reversed_extract_features()
    features_clustering()
    # assign_features_to_centers()
    # sift = pickle.load(open(SIFT_STORE_LOCATION, 'rb'))
    # match('static/mirflickr/im2.jpg')
