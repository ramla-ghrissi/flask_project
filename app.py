# app.py #
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

#  load the pre-trained model and other necessary components  #
classifier_top_features = joblib.load(r'C:\Users\ramla\OneDrive\Desktop\Kais_DS\PYTHON_DS\checkpoints\fraud_project\classifier_top_features.joblib')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        #retrieve user inputs
        user_inputs = [float(request.form[f'input{i+1}']) for i in range(10)] #LIST COMPREHENSION 

        #prediction using the model
        prediction = classifier_top_features.predict(np.array([user_inputs]))

        return render_template('result.html', prediction=prediction[0])

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
