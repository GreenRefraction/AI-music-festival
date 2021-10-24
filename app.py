from flask import Flask, render_template, request
from MLlib.DSP import get_data
import time, os 
app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('index.html')


@app.route("/merge", methods=['GET','POST'])
def enter_data():
    if request.method == 'POST':
        # get files from request
        file1 = request.files['file1']
        file2 = request.files['file2']
        # save files to disk
        upload_folder = 'uploads'
        file1.save(os.path.join(upload_folder, file1.filename))
        file2.save(os.path.join(upload_folder, file2.filename))
        # get data from files
        for x in get_data('./uploads/'):
            print(x)
        
        
        
        
    return render_template('index.html')