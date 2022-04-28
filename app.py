from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import warnings
import pickle
warnings.filterwarnings('ignore')
from feature import generate_data_set

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        url = request.form["url"]
        x = np.array(generate_data_set(url)).reshape(1,30)
        with open('./clustering_pickle', 'rb') as f:
            clst = pickle.load(f)
        clst_res = str(clst.predict(x)[0])
        path = './model/model_{}_pickle'.format(clst_res)
        with open(path, 'rb') as m:
            clf = pickle.load(m)
        result = clf.predict(x)[0]
        return jsonify({"Result":str(result)})

if __name__ == "__main__":
    app.run(debug=False)