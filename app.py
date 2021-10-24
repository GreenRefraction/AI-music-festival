from flask import Flask, render_template, request, send_file, send_from_directory, redirect, url_for
from MLlib.DSP import get_data, arry2mid
from MLlib.Merger import merge_midi_matrix
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
        vec1 = next((x for i,x in enumerate(get_data('./uploads/')) if i==0), None)
        vec2 = next((x for i,x in enumerate(get_data('./uploads/')) if i==1), None)
        print(vec1)
        print(vec2)
        arr_merged = merge_midi_matrix(vec1,vec2)
        merged_mid = arry2mid(arr_merged)
        merged_mid.save('./uploads/merged.mid')
        
        return redirect(url_for('get_files', path='merged.mid'))  
    return render_template('index.html')


# Specify directory to download from . . .
# 

DOWNLOAD_DIRECTORY = "./uploads"

@app.route('/get-files/<path:path>',methods = ['GET','POST'])
def get_files(path):

    """Download a file."""
    try:
        return send_from_directory(DOWNLOAD_DIRECTORY, path, as_attachment=True)
    except FileNotFoundError:
        abort(404)