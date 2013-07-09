from flask import Flask
from flask import url_for
from flask import render_template
from flask import request
import sqlite3
from flask import g
app = Flask(__name__)

DATABASE = 'database.db'

SQL = '''
SELECT *
FROM tags
LEFT JOIN tags_images
ON tags.id = tags_images.tid AND tid = 11
'''

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/search', methods=['GET', 'POST'])
def search():
    query = request.form['query']
    c = get_db().cursor()
    c.execute("SELECT * FROM tags WHERE content LIKE '{}%'".format(query))
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

    return render_template('result.html', results = result)

@app.route('/similar', methods=['GET', 'POST'])
def similar():
    if request.method == 'POST':
        return 'POST'
    else:
        return 'GET'

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

if __name__ == '__main__':
    app.run(debug=True)
