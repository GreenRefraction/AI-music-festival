from flask import Flask, render_template, request
import time
app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('index.html')


@app.route("/merge", methods=['GET','POST'])
def enter_data():
    if request.method == 'POST':
        # get files from request
        time.sleep(10)
        print(request.files['file1'].filename)
        print(request.files['file2'].filename)
    return render_template('index.html')