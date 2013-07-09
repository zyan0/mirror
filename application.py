from flask import Flask
from flask import url_for
from flask import render_template
from flask import request
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/search', methods=['GET', 'POST'])
def search():
    return render_template('result.html')

@app.route('/similar', methods=['GET', 'POST'])
def similar():
    if request.method == 'POST':
        return 'POST'
    else:
        return 'GET'

if __name__ == '__main__':
    app.run(debug=True)
