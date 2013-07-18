import cv2
import cPickle as pickle
import os

def image_cache():
    detector = cv2.FeatureDetector_create("SIFT")
    descriptor = cv2.DescriptorExtractor_create("SIFT")

    os.chdir('static/mirflickr')
    for file_name in os.listdir("."):
        print file_name
        if file_name.endswith(".jpg"):
            img = cv2.imread(file_name)
            keypoints = detector.detect(img)
            keypoints, desc = descriptor.compute(img, keypoints)
            temp = [(point.pt, point.size, point.angle, point.response, point.octave, point.class_id) for point in keypoints]
            temp.append(desc)
            pickle.dump(temp, open(file_name + '.pkl', 'wb'))

if __name__ == '__main__':
    image_cache()
