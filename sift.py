import cv2
import numpy as np
from numpy import array
import itertools
import sys
import os
import cPickle as pickle
from munkres import Munkres

def findKeyPoints(img, template, distance=200):
    detector = cv2.FeatureDetector_create("SIFT")
    descriptor = cv2.DescriptorExtractor_create("SIFT")

    skp = detector.detect(img)
    skp, sd = descriptor.compute(img, skp)

    tkp = detector.detect(template)
    tkp, td = descriptor.compute(template, tkp)

    flann_params = dict(algorithm=1, trees=4)
    flann = cv2.flann_Index(sd, flann_params)
    idx, dist = flann.knnSearch(td, 1, params={})
    del flann

    dist = dist[:,0]/2500.0
    dist = dist.reshape(-1,).tolist()
    idx = idx.reshape(-1).tolist()
    indices = range(len(dist))
    indices.sort(key=lambda i: dist[i])
    dist = [dist[i] for i in indices]
    idx = [idx[i] for i in indices]
    skp_final = []
    for i, dis in itertools.izip(idx, dist):
        if dis < distance:
            skp_final.append(skp[i])

    flann = cv2.flann_Index(td, flann_params)
    idx, dist = flann.knnSearch(sd, 1, params={})
    del flann

    dist = dist[:,0]/2500.0
    dist = dist.reshape(-1,).tolist()
    idx = idx.reshape(-1).tolist()
    indices = range(len(dist))
    indices.sort(key=lambda i: dist[i])
    dist = [dist[i] for i in indices]
    idx = [idx[i] for i in indices]
    tkp_final = []
    for i, dis in itertools.izip(idx, dist):
        if dis < distance:
            tkp_final.append(tkp[i])

    return skp_final, tkp_final

def drawKeyPoints(img, template, skp, tkp, num=-1):
    h1, w1 = img.shape[:2]
    h2, w2 = template.shape[:2]
    nWidth = w1+w2
    nHeight = max(h1, h2)
    hdif = (h1-h2)/2
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
    newimg[hdif:hdif+h2, :w2] = template
    newimg[:h1, w2:w1+w2] = img

    maxlen = min(len(skp), len(tkp))
    if num < 0 or num > maxlen:
        num = maxlen
    for i in range(num):
        pt_a = (int(tkp[i].pt[0]), int(tkp[i].pt[1]+hdif))
        pt_b = (int(skp[i].pt[0]+w2), int(skp[i].pt[1]))
        cv2.line(newimg, pt_a, pt_b, (255, 0, 0))
    return newimg

def match(img_location, features, cluster):
    img = cv2.imread(img_location)

    detector = cv2.FeatureDetector_create("SIFT")
    descriptor = cv2.DescriptorExtractor_create("SIFT")
    
    keypoints = detector.detect(img)
    keypoints = sorted(keypoints, key=lambda x: -x.response)
    keypoints, img_features = descriptor.compute(img, keypoints[0:5])
    img_features = img_features.tolist()

    m = Munkres()

    distance = {}
    for filename in cluster:
        distance[filename] = 0
        fea = features[filename]
        matrix = [[0 for i in range(5)] for j in range(5)]
        for i in range(5):
            for j in range(5):
                if matrix[j][i] != 0:
                    matrix[i][j] = matrix[j][i]
                    continue
                try:
                    matrix[i][j] = np.linalg.norm( array(img_features[i]) - array(fea[j]) )
                except:
                    matrix[i][j] = 10000
        indexes = m.compute(matrix)
        for row, column in indexes:
            distance[filename] += matrix[row][column]

    results = sorted(cluster, key = lambda x: distance[x])
    # results = features.keys()
    return results, distance

SIFT_STORE_LOCATION = 'sift.pkl'

def extract_features():
    detector = cv2.FeatureDetector_create("SIFT")
    descriptor = cv2.DescriptorExtractor_create("SIFT")

    sift = {}

    os.chdir('static/mirflickr')
    for file_name in os.listdir("."):
        print file_name
        if file_name.endswith(".jpg"):
            sift[file_name] = []
            img = cv2.imread(file_name)
            keypoints = detector.detect(img)
            keypoints = sorted(keypoints, key=lambda x: -x.response)
            keypoints, features = descriptor.compute(img, keypoints[0:100])
            try:
                for fea in features:
                    sift[file_name].append(fea.tolist())
            except:
                pass
    
    os.chdir('..')
    os.chdir('..')
    pickle.dump(sift, open('sift_100.pkl', 'wb'))

if __name__ == '__main__':
    extract_features()
