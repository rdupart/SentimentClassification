from flask import Flask, render_template, request
import pickle
import os
import numpy as np
import sqlite3
from vectorizer import vect  # Import your vectorizer

app = Flask(__name__)

# Load trained model
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'classifier.pkl'), 'rb'))
db = os.path.join(cur_dir, 'reviews.sqlite')

# Prediction function
def classify(document):
    label = {0: 'negative', 1: 'positive'}
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y], proba

@app.route('/')
def index():
    return render_template('reviewform.html')

@app.route('/results', methods=['POST'])
def results():
    review = request.form['moviereview']
    y, proba = classify(review)
    return render_template('results.html', content=review, prediction=y, probability=round(proba*100, 2))

if __name__ == '__main__':
    app.run(debug=True)


# python app.py to run