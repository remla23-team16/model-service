from flask import Flask, request, jsonify
import re
import nltk
import pickle
import joblib
import sklearn
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/<data>', methods=['GET'])
@cross_origin()
def fetch_model(data):
    X = pre_process([data])
    return jsonify(predict(X).tolist())


def transform(corpus):
    bow_path = 'c1_BoW_Sentiment_Model.pkl'
    with open(bow_path, 'rb') as f:
        cv = pickle.load(f)
    return cv.transform(corpus).toarray()


def pre_process(data):
    nltk.download('stopwords')
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

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
    model = joblib.load('c2_Classifier_Sentiment_Model')
    y = model.predict(X)
    return y


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)
