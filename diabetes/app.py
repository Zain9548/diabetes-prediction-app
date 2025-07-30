import pickle
import numpy as np
from flask import Flask, request, render_template

# Load the trained model
with open('maine_model (1).pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Convert form inputs to float
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]

    # Predict using the loaded model
    prediction = model.predict(final_features)
    output = prediction[0]

    # Send the result to frontend
    if output == 1:
        return render_template('index.html', prediction_text='The person is likely to have diabetes.')
    else:
        return render_template('index.html', prediction_text='The person is not likely to have diabetes.')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
