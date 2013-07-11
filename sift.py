# -*- coding: utf-8 -*-
"""
SIFT 

it appears that very old versions of OpenCV had the class SIFT,
the newest version has it too, but the version installed doesn't....

Therefore this file contains a wrapper class for the factory functions 

Everything is done with default parameters.

We also give a matching function (it appears that the interface to the C++ still 
needs some work...

sift = SIFT()
kpts1, descrs1 = sift.computeAndExtract(image1)
kpts2, descrs2 = sift.computeAndExtract(image2)

the keypoints and descriptors are vectors/arrays of python wrappers of the
C++ classes (with very little documentation...)

p1, p2, kp_matches = matchDescriptors(kpts1,descrs1,kpts2,descrs2)

p1 is array with the points in first image, p2 the corresponding points in 
the second image, kp_matches is an array of tuples, each tuple contains 
matching keypoints (see documentation for the attributes of the class)

"""
import cv2
import numpy as np

class SIFT:
    def __init__(self):
        self.fd = cv2.FeatureDetector_create('SIFT')
        self.de = cv2.DescriptorExtractor_create('SIFT')
        
    def detectAndCompute(self,image,mask=None):
        kpts = self.fd.detect(image,mask=mask)
        descrs = self.de.compute(image,kpts)
        return kpts, descrs
        
        
def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs        
        
def matchDescriptors(kp1, d1, kp2, d2):
    #matcher = cv2.BFMatcher(cv2.NORM_L2) # oops not in the pythonxy opencv
    matcher = cv2.DescriptorMatcher_create('BruteForce')
    raw_matches = matcher.knnMatch(d1[1], trainDescriptors = d2[1], k = 2) 
    p1, p2, kp_matches = filter_matches(kp1,kp2,raw_matches)
    return p1, p2, kp_matches
