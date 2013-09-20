import sys
import tornado.ioloop
import tornado.web
import tornado.httpserver
import tornado.httputil
import json
import cPickle, os, sys, Image
import numpy as np

sys.path.append("/home/yjj/cuda-convnet/")
from shownet import *

class Tutuso:
    def __init__(self):
        self.op = ShowConvNet.get_options_parser()
        self.op.set_value('load_file', '/home/yjj/cuda-convnet/save/ConvNet__2013-07-24_01.03.19/')
        self.op, self.load_dic = self.parse_options(self.op)
        #print op.options['layer_def'].value
        #op.set_value('gpu', str(op.get_value('gpu')[0]))
        self.op.set_value('gpu', "0")

        #following block is used to set ConvNet__2013-07-05_23.59.53's options to local
        self.op.set_value('dp_type', 'cifar-cropped')
        self.op.set_value('data_path', '/home/yjj/cifar-10-py-colmajor/')
        self.op.set_value('save_path', '/home/yjj/cuda-convnet/save/')
        self.op.set_value('test_batch_range', '1')
        self.op.set_value('train_batch_range', '1')
        self.op.set_value('layer_def', '/home/yjj/cuda-convnet/example-layers/layers-zdb-nips.cfg')
        self.op.set_value('layer_params', '/home/yjj/cuda-convnet/example-layers/layer-params-imagenet.cfg')
        self.op.set_value('write_features', 'probs')
        self.op.set_value('feature_path', '/home/yjj/cuda-convnet/save/tmp')
        self.op.set_value('test_batch_range', '1-1')
        self.op.print_values()

        self.model = ShowConvNet(self.op, self.load_dic)

    def get_feature_of_file(self, filename):
        batched_img = self.make_batch(filename)
        data = [batched_img['data'] , batched_img['labels']]
        print data[0].shape[0]
        print data[0].shape[1]
        print data[1].shape[0]
        print data[1].shape[1]
        ftr = self.model.get_features_of_data(data)['data'][0]
        return ftr

    def make_batch(self, filename):
        print filename
        sz = 224

        data = []
        img = Image.open(filename);
        img = img.resize((sz,sz),Image.BILINEAR);
        img = np.array(img, order='C');
        img = np.fliplr(np.rot90(img, k=3))
        img = img.T.flatten('C')
        img = img.transpose()

        cp_num = 128

        data = map(lambda x:[x]*cp_num, img)
        data = np.matrix(data, dtype = 'float32')
        print data.shape
        print sys.getsizeof(data)

        dict = {}
        dict['batch_label'] = 'test img';
        dict['labels'] = np.matrix([0]*cp_num);
        dict['data'] = data
        dict['filename'] = [filename]*cp_num;

        #f = open('data_batch_8888','wb')
        #cPickle.dump(dict,f,1)
        #f.close()
        return dict

    def parse_options(self, op):
        try:
            load_dic = None
            options = op.parse_no_exception()
            if options["load_file"].value_given:
                load_dic = IGPUModel.load_checkpoint(options["load_file"].value)
                old_op = load_dic["op"]
                old_op.merge_from(op)
                op = old_op
            op.eval_expr_defaults()
            return op, load_dic
        except OptionMissingException, e:
            print e
            op.print_usage()
        except OptionException, e:
            print e
        except UnpickleError, e:
            print "Error loading checkpoint:"
            print e
        sys.exit()

if __name__ == '__main__':
    tu = Tutuso()
    print tu.get_feature_of_file("/home/zdb/work/deep_learning/online_models/comb_model/server/static/images/122/638.jpg")
