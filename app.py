import numpy as np
from flask import Flask, request, render_template
import pickle

# create flask app
app = Flask(__name__)

# load the pickle model
model = pickle.load(open("Model.pkl", "rb"))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict')
def pre():
    return render_template("predict.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    float_features = [float(x) for x in request.form.values()]  # convert integer into floats
    features = [np.array(float_features)]  # convert floats into array and store
    prediction = model.predict(features)
    final = model.predict_proba(features)
    output = '{0:.{1}f}'.format(final[0][1], 2)

    if float(prediction) == 1.0:
        result = "This person is more likely to have a malignant cancer with probability value {} ".format(output)
    else:
        result = "This person is more likely to have a benign cancer with probability value {} ".format(output)

    return render_template("predict.html", pred=result)


@app.route("/treatments")
def treatments():
    return render_template("treatments.html")


@app.route("/lifestyle")
def lifestyle():
    return render_template("lifestyle.html")


if __name__ == "__main__":
    app.run(debug=True)
