# -*- coding: utf-8 -*-
import os
from flask import Flask
from flask import url_for
from flask import render_template
from flask import request
from flask import redirect
from flask import g
from werkzeug import secure_filename
import nltk
import string
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from database import Database
import cPickle as pickle
import math

UPLOAD_FOLDER = 'static/media'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
DATABASE = 'database.pkl'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/search', methods=['GET', 'POST'])
def search():
    input_query = request.form['query']
    db = get_db()
    results = None
    
    queries = query_preprocess(input_query.split())
    scores = {}
    
    for query in queries:
        tags = [tag for tag in db.tags if tag == query]
        result = set()
        
        for tag in tags:
            result = set.union(result, set(db.tags_filenames[tag]))
            
        if results == None:
            results = result
        elif len(result) == 0:
            pass
        else:
            results_new = set.intersection(results, result)
            if len(results_new) == 0:
                for f in result:
                    scores[f] = -1
                results = set.union(results, result)
            else:
                results = results_new

    if results != None:
        for filename in results:
            scores[filename] = math.log(100.0 / float(db.files_count[filename]))
            for tag in db.filenames_tags[filename]:
                for query in queries:
                    if tag == query:
                        scores[filename] += math.log(float(len(db.tags)) / float(db.tags_count[tag]))
    
        results = sorted(results, key=lambda x: -scores[x])
    else:
        results = []
        
    return render_template('result.html', results = results, query = input_query, scores = scores)

@app.route('/similar', methods=['GET', 'POST'])
def similar():
    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename.lower()):
            filename = secure_filename(file.filename.lower()).lower()
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            fileurl = app.config['UPLOAD_FOLDER'] + '/' + filename

def get_db():
    if not hasattr(g, 'db'):
        g.db = pickle.load(open(DATABASE, 'rb'))
    return g.db

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def query_preprocess(val):
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(string.punctuation)
    stopwords = set(stopwords)
    porter = PorterStemmer()
    return [porter.stem(x) for x in val if x not in stopwords and len(x) > 1]

if __name__ == '__main__':
    app.run(debug=True)
