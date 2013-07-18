import cv2
import pyflann
import numpy
import cPickle as pickle
import os
import math

vq = {}

def main():
    global vq

    vq = pickle.load( open('vq.pkl', 'rb') )

    centroids = pickle.load( open('centroids.pkl', 'rb') )
    params = pickle.load( open('params.pkl', 'rb') )
    print 'vars finish loading.'

    flann = pyflann.FLANN()
    flann.load_index('flann.pkl', numpy.array(centroids, dtype=numpy.float64))
    print 'flann finish loading.'

    detector = cv2.FeatureDetector_create("SIFT")
    descriptor = cv2.DescriptorExtractor_create("SIFT")

    counter = 0
    os.chdir('static/mirflickr')
    for file_name in os.listdir("."):
        if file_name in vq or file_name == 'im13064.jpg':
            continue
        if file_name.endswith(".jpg"):
            print file_name
            img = cv2.imread(file_name)
            keypoints = detector.detect(img)
            keypoints, features = descriptor.compute(img, keypoints)
            try:
                idx = flann.nn_index( numpy.array(features, dtype=numpy.float64), 1, checks=params["checks"])
            except:
                continue
            vq[file_name] = {}
            idx = idx[0].tolist()
            for i in idx:
                if i in vq[file_name]:
                    vq[file_name][i] += 1
                else:
                    vq[file_name][i] = 1
        if counter % 1000 == 0:
            pickle.dump(vq, open('vq.pkl', 'wb'))
        counter += 1

    os.chdir('..')
    os.chdir('..')
    pickle.dump(vq, open('vq.pkl', 'wb'))

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
       return 0.0
    else:
       return float(numerator) / denominator

if __name__ == '__main__':
    try:
        main()
    except:
        os.chdir('..')
        os.chdir('..')
        pickle.dump(vq, open('vq.pkl', 'wb'))
