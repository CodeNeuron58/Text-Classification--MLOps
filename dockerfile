FROM python:3.10

WORKDIR /app

COPY Flask_App/ /app/
COPY models/vectorizer.pkl /app/models/vectorizer.pkl
COPY models/model.pkl /app/models/model.pkl


RUN pip install -r requirement.txt && \
    python -m nltk.downloader stopwords wordnet

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
