impoer flask 
from flask import Flask, request, render_template
import torch


app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')

def make_predictions():

    if request.method=='POST':

        file = request.files['image']
        if not file:
            return render_template('index.html', label = 'No file')

        

