# Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# 
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as n
import os
from time import time, asctime, localtime, strftime, sleep
from numpy.random import randn, rand
from numpy import s_, dot, tile, zeros, ones, zeros_like, array, ones_like
from util import *
from data import *
from options import *
from math import ceil, floor, sqrt
from data import DataProvider, dp_types
import sys
import shutil
import platform
import threading
import copy
from os import linesep as NL

class ModelStateException(Exception):
    pass

# GPU Model interface
class IGPUModel:
    def __init__(self, model_name, op, load_dic, filename_options=None, dp_params={}):
        # these are input parameters
        self.model_name = model_name
        self.op = op
        self.options = op.options
        # shownet.get_gpus needs self.op
        self.device_ids = []
        self.get_gpus(op.get_value('gpu'))
        self.gpu_num = len(self.device_ids)
        self.load_dic = load_dic
        self.filename_options = filename_options
        self.dp_params = dp_params
        self.fill_excused_options()
        self.dp_mutex = threading.Lock()
        self.update_mutex = threading.Lock()
        #assert self.op.all_values_given()
        
        for o in op.get_options_list():
            setattr(self, o.name, o.value)

        # these are things that the model must remember but they're not input parameters
        if load_dic:
            self.model_state = load_dic["model_state"]
            #the following block deal with old single-gpu save file loaded in multi-gpus code
            if not self.model_state.has_key('master_layers'):
                self.model_state['master_layers'] = self.model_state['layers']
            if not self.model_state.has_key('gpu_layers'):
                self.model_state['gpu_layers'] = []
                for i in range(self.gpu_num):
                    self.model_state['gpu_layers'].append(copy.deepcopy(self.model_state['layers']))
            self.save_file = self.options["load_file"].value
            if not os.path.isdir(self.save_file):
                self.save_file = os.path.dirname(self.save_file)
        else:
            self.model_state = {}
            if filename_options is not None:
                self.save_file = model_name + "_" + '_'.join(['%s_%s' % (char, self.options[opt].get_str_value()) for opt, char in filename_options]) + '_' + strftime('%Y-%m-%d_%H.%M.%S')
            self.model_state["train_outputs"] = []
            self.model_state["test_outputs"] = []
            self.model_state["epoch"] = 1
            self.model_state["batchnum"] = self.train_batch_range[0]

        self.init_data_providers()
        if load_dic: 
            self.train_data_provider.advance_batch()
            
        # model state often requries knowledge of data provider, so it's initialized after
        try:
            self.init_model_state()
        except ModelStateException, e:
            print e
            sys.exit(1)
        for var, val in self.model_state.iteritems():
            setattr(self, var, val)
            
        self.import_model()
        self.init_model_lib()
        
    def import_model(self):
        print "========================="
        print "Importing %s C++ module" % ('_' + self.model_name)
        self.libmodel = __import__('_' + self.model_name)
                   
    def fill_excused_options(self):
        pass
    
    def init_data_providers(self):
        self.dp_params['convnet'] = self
        try:
            self.test_data_provider = DataProvider.get_instance(self.data_path, self.test_batch_range,
                                                                type=self.dp_type, dp_params=self.dp_params, test=True)
            self.train_data_provider = DataProvider.get_instance(self.data_path, self.train_batch_range,
                                                                     self.model_state["epoch"], self.model_state["batchnum"],
                                                                     type=self.dp_type, dp_params=self.dp_params, test=False)
        except DataProviderException, e:
            print "Unable to create data provider: %s" % e
            self.print_data_providers()
            sys.exit()
        
    def init_model_state(self):
        pass
       
    def init_model_lib(self):
        pass

    def init_trainers(self):
        self.trainers = []
        for i in range(self.gpu_num):
            self.trainers.append(Trainer(self, i, self.device_ids[i]))

    def start(self):
        if self.test_only:
            self.test_outputs += [self.get_test_error()]
            self.print_test_results()
            sys.exit(0)
        self.train()

    def train(self):
        print "========================="
        print "Training %s" % self.model_name
        self.op.print_values()
        print "========================="
        self.print_model_state()
        print "Running on CUDA device(s) %s" % ", ".join("%d" % d for d in self.device_ids)
        print "Current time: %s" % asctime(localtime())
        print "Saving checkpoints to %s" % os.path.join(self.save_path, self.save_file)
        print "========================="

        self.init_trainers()

        for i in range(self.gpu_num):
            self.trainers[i].start()

        for i in range(self.gpu_num):
            self.trainers[i].join()

        self.cleanup()

    def cleanup(self):
        sys.exit(0)
            
    def print_model_state(self):
        pass
    
    def print_iteration(self):
        print "\t%d.%d..." % (self.epoch, self.batchnum),

    def print_test_results(self):
        batch_error = self.test_outputs[-1][0]
        print "%s\t\tTest error: %.6f" % (NL, batch_error),
        
    def conditional_save(self):
        batch_error = self.test_outputs[-1][0]
        if batch_error > 0 and batch_error < self.max_test_err:
            self.save_state()
        else:
            print "\tTest error > %g, not saving." % self.max_test_err,

    def aggregate_test_outputs(self, test_outputs):
        test_error = tuple([sum(t[r] for t in test_outputs) / (1 if self.test_one else len(self.test_batch_range)) for r in range(len(test_outputs[-1]))])
        return test_error

    def get_test_error(self):
        next_data = self.get_next_batch(train=False)
        test_outputs = []
        while True:
            data = next_data
            self.start_batch(data, train=False)
            load_next = not self.test_one and data[1] < self.test_batch_range[-1]
            if load_next: # load next batch
                next_data = self.get_next_batch(train=False)
            test_outputs += [self.finish_batch()]
            if self.test_only: # Print the individual batch results for safety
                print "batch %d: %s" % (data[1], str(test_outputs[-1]))
            if not load_next:
                break
            sys.stdout.flush()
    
    def set_var(self, var_name, var_val):
        setattr(self, var_name, var_val)
        self.model_state[var_name] = var_val
        return var_val
        
    def get_var(self, var_name):
        return self.model_state[var_name]
        
    def has_var(self, var_name):
        return var_name in self.model_state

    def save_state(self):
        for att in self.model_state:
            if hasattr(self, att):
                self.model_state[att] = getattr(self, att)

        dic = {"model_state": self.model_state,
               "op": self.op}

        checkpoint_dir = os.path.join(self.save_path, self.save_file)
        checkpoint_file = "%d.%d" % (self.epoch, self.batchnum)
        checkpoint_file_full_path = os.path.join(checkpoint_dir, checkpoint_file)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        pickle(checkpoint_file_full_path, dic,compress=self.zip_save)

        for f in sorted(os.listdir(checkpoint_dir), key=alphanum_key):
            if sum(os.path.getsize(os.path.join(checkpoint_dir, f2)) for f2 in os.listdir(checkpoint_dir)) > self.max_filesize_mb*1024*1024 and f != checkpoint_file:
                os.remove(os.path.join(checkpoint_dir, f))
            else:
                break

    @staticmethod
    def load_checkpoint(load_dir):
        if os.path.isdir(load_dir):
            return unpickle(os.path.join(load_dir, sorted(os.listdir(load_dir), key=alphanum_key)[-1]))
        return unpickle(load_dir)

    @staticmethod
    def get_options_parser():
        op = OptionsParser()
        op.add_option("f", "load_file", StringOptionParser, "Load file", default="", excuses=OptionsParser.EXCLUDE_ALL)
        op.add_option("train-range", "train_batch_range", RangeOptionParser, "Data batch range: training")
        op.add_option("test-range", "test_batch_range", RangeOptionParser, "Data batch range: testing")
        op.add_option("data-provider", "dp_type", StringOptionParser, "Data provider", default="default")
        op.add_option("test-freq", "testing_freq", IntegerOptionParser, "Testing frequency", default=25)
        op.add_option("epochs", "num_epochs", IntegerOptionParser, "Number of epochs", default=500)
        op.add_option("data-path", "data_path", StringOptionParser, "Data path")
        op.add_option("save-path", "save_path", StringOptionParser, "Save path")
        op.add_option("max-filesize", "max_filesize_mb", IntegerOptionParser, "Maximum save file size (MB)", default=5000)
        op.add_option("max-test-err", "max_test_err", FloatOptionParser, "Maximum test error for saving")
        op.add_option("num-gpus", "num_gpus", IntegerOptionParser, "Number of GPUs", default=1)
        op.add_option("test-only", "test_only", BooleanOptionParser, "Test and quit?", default=0)
        op.add_option("zip-save", "zip_save", BooleanOptionParser, "Compress checkpoints?", default=0)
        op.add_option("test-one", "test_one", BooleanOptionParser, "Test on one batch at a time?", default=1)
        op.add_option("gpu", "gpu", ListOptionParser(IntegerOptionParser), "GPU override", default=OptionExpression("[0] * 1"))
        return op

    @staticmethod
    def print_data_providers():
        print "Available data providers:"
        for dp, desc in dp_types.iteritems():
            print "    %s: %s" % (dp, desc)

    def get_gpus(self, ids):
        self.device_ids = ids;

    @staticmethod
    def parse_options(op):
        try:
            load_dic = None
            options = op.parse()
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

    def get_next_batch(self, train=True):
        dp = self.train_data_provider
        if not train:
            dp = self.test_data_provider
        return self.parse_batch_data(dp.get_next_batch(), train=train)

    def parse_batch_data(self, batch_data, train=True):
        return batch_data[0], batch_data[1], batch_data[2]['data']

    def finish_gpu_batch(self, device_id):
        return self.libmodel.finishBatch(device_id)

class Trainer(threading.Thread):
    def __init__(self, gpumodel, index, device_id):
        threading.Thread.__init__(self)
        self.test_data_provider = gpumodel.test_data_provider
        self.train_data_provider = gpumodel.train_data_provider
        self.gpumodel = gpumodel
        self.device_index = index
        self.device_id = device_id
        self.libmodel = gpumodel.libmodel
        self.epoch = 1
        self.num_epochs = gpumodel.num_epochs
        self.layers = gpumodel.layers
        self.train_outputs = []
        self.test_outputs = []
        self.model = None

    def run(self):
        next_data = self.get_next_batch()
        batch_error_avg = 1.0
        counter = 1.0

        while self.epoch <= self.num_epochs:
            counter += 1.0

            data = next_data
            self.epoch, self.batchnum = data[0], data[1]

            compute_time_py = time()
            self.start_batch(data)
 
            # overlapping
            next_data = self.get_next_batch()

            batch_output = self.finish_batch()
            if batch_output is None:
                print "!!!! device{}: batch_output is None, exit.".format(self.device_id)
                exit(0)

            self.train_outputs += [batch_output]

            batch_error = self.train_outputs[-1][0]['logprob'][1]
            batch_error_avg = (batch_error * counter * 0.1 + batch_error_avg * counter) / (counter * 1.1)

            self.print_iteration()
            self.print_train_results()
            print "device" + str(self.device_id) + " ",

            self.gpumodel.update_mutex.acquire()
            self.model.train_outputs += [batch_output] # don't know if model.train_outputs is used in shownet.py; just stay as the origin version
            self.model.epoch = self.epoch
            self.model.batchnum = self.batchnum # these two data indices is uesed in conditional save and load again
            self.update_work(batch_error, batch_error_avg)
            self.gpumodel.update_mutex.release()

            self.print_train_time(time() - compute_time_py)
            sys.stdout.flush()

            if self.get_num_batches_done() % self.model.testing_freq == 0:
                next_data_copy = copy.deepcopy(next_data)
                self.test_outputs += [self.get_test_error()]
                self.model.test_outputs += [self.get_test_error()]
                self.print_test_results()
                self.print_test_status()
                self.model.conditional_save()
                print ''
                next_data = next_data_copy

 
    def get_next_batch(self, train=True):
        self.gpumodel.dp_mutex.acquire()
        dp = self.train_data_provider
        if not train:
            dp = self.test_data_provider
        tmp = self.parse_batch_data(dp.get_next_batch(), train=train)
        self.gpumodel.dp_mutex.release()
        return tmp
    
    def parse_batch_data(self, batch_data, train=True):
        return batch_data[0], batch_data[1], batch_data[2]['data']
            
    def print_iteration(self):
        print "\t%d.%d..." % (self.epoch, self.batchnum),

    def start_batch(self, batch_data, train=True):
        self.libmodel.startBatch(batch_data[2], not train, self.device_id)
    
    def finish_batch(self):
        return self.libmodel.finishBatch(self.device_id)
    
    def print_train_results(self):
        batch_error = self.train_outputs[-1][0]
        if not (batch_error > 0 and batch_error < 2e20):
            print "Crazy train error: %.6f" % batch_error
            self.cleanup()
        print "Train error: %.6f " % (batch_error),

    def cleanup(self):
        sys.exit(0)
    
    def get_num_batches_done(self):
        return len(self.gpumodel.train_batch_range) * (self.epoch - 1) + self.batchnum - self.gpumodel.train_batch_range[0] + 1

    def sync_with_host(self):
        self.libmodel.syncWithHost(self.device_id)

    def update_device(self, layers):
        self.libmodel.updateDevice(layers, self.device_id)

    def print_test_status(self):
        status = (len(self.test_outputs) == 1 or self.test_outputs[-1][0] < self.test_outputs[-2][0]) and "ok" or "WORSE"
        print status,

    def print_train_time(self, compute_time_py):
        print "(%.3f sec)" % (compute_time_py)

    def get_test_error(self):
        next_data = self.get_next_batch(train=False)
        test_outputs = []
        while True:
            data = next_data
            self.start_batch(data, train=False)
            load_next = not self.gpumodel.test_one and data[1] < self.gpumodel.test_batch_range[-1]
            if load_next: # load next batch
                next_data = self.get_next_batch(train=False)
            test_outputs += [self.finish_batch()]
            if self.gpumodel.test_only: # Print the individual batch results for safety
                print "batch %d: %s" % (data[1], str(test_outputs[-1]))
            if not load_next:
                break
            sys.stdout.flush()
            
        return self.gpumodel.aggregate_test_outputs(test_outputs)

    def update_work(self):
        pass
