from flask import Flask, render_template, request
import mlflow
from preprocessing_utility import normalize_text
import pickle
import dagshub

dagshub.init(repo_owner='CodeNeuron58', repo_name='Text-Classification--MLOps', mlflow=True)

app = Flask(__name__)


# model_name = "LR model"
# model_version = 1

# # Option 1: Get the latest production version
# latest_production_version = client.get_latest_versions(name=model_name, stages=["Production"])[0].version
# model_uri = f"models:/{model_name}/{latest_production_version}"

model_uri = f"runs:/20fa0da1a7ad4ebbbe28eb671974573c/model"
model = mlflow.pyfunc.load_model(model_uri)

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
    app.run(debug=True)