# -*- coding: utf-8 -*-
import cPickle as pickle

class Database:
    def __init__(self):
        self.filenames = []
        self.tags = []
        self.tags_count = {}
        self.files_count = {}
        self.tags_filenames = {}
        self.filenames_tags = {}

    def add_filename(self, file_name):
        self.filenames.append(file_name)
        self.filenames_tags[file_name] = []
        self.files_count[file_name] = 0

    def add_tag(self, tag):
        self.tags.append(tag)
        self.tags_count[tag] = 0
        self.tags_filenames[tag] = []

    def add_relation(self, tag, file_name):
        self.tags_count[tag] += 1
        self.files_count[file_name] += 1
        self.tags_filenames[tag].append(file_name)
        self.filenames_tags[file_name].append(tag)

    def save(self, file_name):
        pickle.dump(self, open(file_name, 'wb'))
