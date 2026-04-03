FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV TF_ENABLE_ONEDNN_OPTS=0

CMD gunicorn app:app --bind 0.0.0.0:$PORT