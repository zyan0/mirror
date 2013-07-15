import cv2
import numpy
import pyflann
import random

class kMeans():

    def __init__(self, dataset, k, max_iter):
        self.dataset = numpy.array( dataset )
        self.k = k
        self.max_iter = max_iter
        self.centers = None
        self.labels = None

    def train(self):
        flann = pyflann.FLANN()

        c_idx = random.sample( xrange( len(self.dataset) ), self.k )
        centers = self.dataset[c_idx, :]

        for iter in range(self.max_iter):
            print iter

            params = flann.build_index(centers, algorithm="autotuned", target_precision=0.8, log_level = "warning")
            random_idx = numpy.random.choice(xrange(len(self.dataset)), len(self.dataset) / 5)
            random_dataset = self.dataset[random_idx, :]
            nn, distance = flann.nn_index(random_dataset, 1, checks=params["checks"])

            for i in range(self.k):
                group_idx = numpy.argwhere(nn == i)
                if len(group_idx) == 0:
                    centers[i] = self.dataset[ random.sample( xrange( len(self.dataset) ), 1 ), :]
                else:
                    centers[i] = numpy.sum( random_dataset[group_idx,:], 0 ) / len(group_idx)

            print 'total distance: {}'.format(numpy.sum(distance))

        del self.dataset
        self.centers = centers
        self.labels = nn

    def get_centers(self):
        return self.centers.tolist()
    
    def get_labels(self):
        return self.labels.tolist()

    def predict(self, testset):
        flann = pyflann.FLANN()
        testset = numpy.array(testset)

        params = flann.build_index(self.centers, algorithm="autotuned", target_precision=0.9, log_level = "warning")
        nn, dummy = flann.nn_index(testset, 1, checks=params["checks"])

        return nn.tolist()

    # def _build_centers_dict(self, centers):
    #     centers_dict = {}
    #     counter = 0
    #     for c in centers:
    #         centers_dict[str(c)] = counter
    #         counter += 1
    #     return centers_dict
