FROM python:3.7-slim

WORKDIR /app

COPY main.py .
COPY c1_BoW_Sentiment_Model.pkl .
COPY c2_Classifier_Sentiment_Model .
COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["python","main.py"]