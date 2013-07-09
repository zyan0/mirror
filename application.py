import os
from flask import Flask
from flask import url_for
from flask import render_template
from flask import request
from flask import redirect
import sqlite3
from flask import g
from werkzeug import secure_filename

UPLOAD_FOLDER = 'static/media'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
DATABASE = 'database.db'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/search', methods=['GET', 'POST'])
def search():
    input_query = request.form['query']
    c = get_db().cursor()
    results = None
    
    queries = input_query.split()
    for query in queries:
        c.execute("SELECT * FROM tags WHERE content LIKE '{}%' LIMIT 100".format(query))
        tags = c.fetchall()
        result = set()
        for tag in tags:
            c.execute("SELECT pid FROM tags_images WHERE tid = {}".format(tag[0]))
            tags_images = c.fetchall()
            for t_i in tags_images:
                pid = t_i[0]
                c.execute("SELECT file_name FROM images WHERE id = {}".format(pid))
                try:
                    result = set.union(result, set(c.fetchone()))
                except:
                    pass
        if results == None:
            results = result
        else:
            results = set.intersection(results, result)

    return render_template('result.html', results = results, query = input_query)

@app.route('/similar', methods=['GET', 'POST'])
def similar():
    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename.lower()):
            filename = secure_filename(file.filename.lower()).lower()
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            fileurl = app.config['UPLOAD_FOLDER'] + '/' + filename

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = connect_db()
    return db

def connect_db():
    return sqlite3.connect(DATABASE)

@app.before_request
def before_request():
    g.db = connect_db()

@app.teardown_request
def teardown_request(exception):
    if hasattr(g, 'db'):
        g.db.close()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)
