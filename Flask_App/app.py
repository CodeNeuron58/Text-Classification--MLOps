from flask import Flask, render_template, request
from preprocessing_utility import normalize_text
import pickle

app = Flask(__name__)


model = pickle.load(open("models/model.pkl", "rb"))

vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')



@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    text = normalize_text(text)
    text = vectorizer.transform([text])
    text = model.predict(text)
    return  render_template('home.html', prediction=text[0]) # Return the predicted text


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')