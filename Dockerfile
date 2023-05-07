From python:3.7-slim

workdir /model-service

COPY . /model-service/

RUN pip install -r requirements.txt

CMD ["python","main.py"]