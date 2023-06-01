from flask import Flask, jsonify, make_response
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
}

nltk.download('stopwords')
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

cv, model = None, None


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
    return "Success", 200


@app.route('/models')
def list_versions():
    res = []
    for i in range(len(os.listdir(classifier_path))):
        res.append(str(i + 1))
    res[-1] = res[-1] + "(latest)"
    return jsonify(res)


@app.route('/metrics')
def get_metrics():
    positive_ratio = 0 if metrics["n_predictions"] == 0 else metrics["n_positive"]/metrics["n_predictions"]
    res = '''
    # HELP n_predictions The total number of predictions made
    # TYPE n_predictions counter
    n_predictions{{}} {n_predictions}

    # HELP sentiment Ratio of positive/negative predictions
    # TYPE sentiment gauge
    sentiment{{type = "1"}} {positive_ratio}
    sentiment{{type = "0"}} {negative_ratio}
    '''.format(n_predictions=metrics["n_predictions"], positive_ratio=positive_ratio, negative_ratio=1-positive_ratio)
    response = make_response(res, 200)
    response.mimetype = "text/plain"
    return response


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


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)
    select_version("latest")
