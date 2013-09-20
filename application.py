# -*- coding: utf-8 -*-
import os
from flask import Flask
from flask import url_for
from flask import render_template
from flask import request
from flask import redirect
from flask import g
from werkzeug import secure_filename
import string
from database import Database
import cPickle as pickle
from tutuso import Tutuso

UPLOAD_FOLDER = 'static/media'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
BASE_PATH = '/home/zdb/work/deep_learning/online_models/comb_model/'
APP_PATH = '/home/yan/mirror'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = None

@app.route('/')
def index():
    global model
    if model == None:
        model = Tutuso()

    return render_template('index.html')

@app.route('/similar', methods=['POST'])
def similar():
    file = request.files['image']
    if file and allowed_file(file.filename.lower()):
        filename = secure_filename(file.filename.lower()).lower()
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        fileurl = app.config['UPLOAD_FOLDER'] + '/' + filename
        file_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # codes below are based on zdb's work
        results = _similar(file_location)

        return render_template('similar.html', results = results, distances = None, original = fileurl)

    return redirect(url_for('index'))

@app.route('/similar/<filename>', methods=['GET'])
def similar_given_filename(filename):
    if filename and allowed_file(filename.lower()):
        fileurl = app.config['UPLOAD_FOLDER'] + '/' + filename
        file_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # codes below are based on zdb's work
        results = _similar(file_location)

        return render_template('similar.html', results = results, distances = None, original = fileurl)

    return redirect(url_for('index'))

def _similar(file_location):
    global model
    if model == None:
        model = Tutuso()

    fea = model.get_feature_of_file(file_location)
    write_feature(fea)
    os.system('cd ' + BASE_PATH + 'code1 && ./cal_kNN && cd ' + APP_PATH)

    results = []
    for filename in open(BASE_PATH + 'code1/returned_name'):
        results.append('static/' + filename[:-1])

    return results

def get_db():
    if not hasattr(g, 'db'):
        g.db = pickle.load(open(DATABASE, 'rb'))
    return g.db

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def write_feature(fea):
    f = open(BASE_PATH + 'query_fea/query_fea1', 'w')
    j = list(fea)
    for k in j:
        f.write(str(k) + " ")
    f.write('\n')
    f.close()

if __name__ == '__main__':
    app.debug = True
    app.threaded = True
    app.run(host='0.0.0.0')
