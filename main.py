from flask import Flask, jsonify
import re
import nltk
import pickle
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

app = Flask(__name__)

classifier_path = os.environ['CLASSIFIER_PATH']
bow_path = os.environ['BOW_PATH']

metrics = {
    "n_predictions": 0,
    "n_positive": 0,
    "current_model": "latest"
}

nltk.download('stopwords')
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

cv, model = None


def load_version(v):
    global cv, model
    # Load data
    with open(os.path.join(bow_path, 'sentiment-model-' + v, 'rb')) as f:
        cv = pickle.load(f)
    model = joblib.load(os.path.join(classifier_path, 'classifier-model-' + v))


@app.route('/model/<version>')
def select_version(version):
    if version == "latest":
        load_version(str(len(os.listdir(classifier_path))))
    else:
        try:
            v = int(version)
            if v > len(os.listdir(classifier_path)) or v < 1:
                raise Exception("Wrong version specified")
            load_version(str(v))
        except:
            return "Wrong version specified", 400
    metrics["current_model"] = version
    return "Success", 200


@app.route('/models')
def list_versions():
    res = []
    for i in range(len(os.listdir(classifier_path))):
        res.append(str(i + 1))
    res[-1] = res[-1] + "(latest)"
    return jsonify(res)


@app.route('/<data>')
def fetch_model(data):
    X = pre_process([data])
    return jsonify(predict(X).tolist())


def transform(corpus):
    return cv.transform(corpus).toarray()


def pre_process(data):
    corpus = []

    for i in range(len(data)):
        review = re.sub('[^a-zA-Z]', ' ', data[i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)
    return transform(corpus)


def predict(X):
    y = model.predict(X)
    metrics["n_predictions"] += 1
    metrics["n_positive"] += 1 if y == 1 else 0
    return y


@app.route('/metrics')
def fetch_model(data):
    return jsonify(metrics)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)
    select_version("latest")
