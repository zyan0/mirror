# -*- coding: utf-8 -*-
import sqlite3
import os
import cPickle as pickle
from database import Database
from nltk.stem.porter import PorterStemmer
import string

DATABASE = 'database.pkl'

def is_ascii(val):
    try:
        val.decode('ascii')
        return True
    except:
        return False

def main():
    db = Database()
    porter = PorterStemmer()
    
    os.chdir('static/mirflickr')
    for file_name in os.listdir("."):
        if file_name.endswith(".jpg"):
            if file_name in db.filenames:
                continue

            db.add_filename(file_name)
            print file_name

            file_number = file_name.replace('.jpg', '').replace('im', '')
            f = open('meta/tags/tags{}.txt'.format(file_number))
            for tag in f:
                tag = tag.strip()
                if not is_ascii(tag):
                    continue
                tag = porter.stem(tag)
                if tag not in db.tags:
                    db.add_tag(tag)
                db.add_relation(tag, file_name)

    os.chdir('..')
    os.chdir('..')
    db.save(DATABASE)

if __name__ == '__main__':
    main()
