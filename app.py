# app.py : the Flask backend for the Titanic Survival Predictor

import pickle                      # to load the saved model
import pandas as pd                # to put the passenger details into a table
from flask import Flask, render_template, request

# Create the Flask app
app = Flask(__name__)

# Load the trained Random Forest model you saved in Colab.
# The file titanic_rf_model.pkl must be in the same folder as this app.py
with open('titanic_rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

# These are the exact columns, in the exact order, the model was trained on
COLUMNS = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
           'Sex_male', 'Embarked_Q', 'Embarked_S']


# Home page: shows the empty form
@app.route('/')
def home():
    return render_template('index.html', result=None, form={}, error=None)


# Predict page: runs when the user taps the "Check my fate" button
@app.route('/predict', methods=['POST'])
def predict():
    form = request.form  # everything the user typed or selected

    try:
        # Read the numbers from the form
        pclass = int(form['pclass'])
        age = float(form['age'])
        sibsp = int(form['sibsp'])
        parch = int(form['parch'])
        fare = float(form['fare'])

        # Read the text choices
        sex = form['sex']            # 'male' or 'female'
        embarked = form['embarked']  # 'S', 'C' or 'Q'

        # Basic checks so the model does not get nonsense
        if pclass not in (1, 2, 3):
            raise ValueError('Please choose a class.')
        if not 0 < age <= 100:
            raise ValueError('Age must be between 1 and 100.')
        if sibsp < 0 or parch < 0 or fare < 0:
            raise ValueError('Numbers cannot be negative.')
        if sex not in ('male', 'female'):
            raise ValueError('Please choose male or female.')
        if embarked not in ('S', 'C', 'Q'):
            raise ValueError('Please choose a port.')

    except (KeyError, ValueError) as problem:
        # If anything is missing or wrong, show the form again with a message
        message = str(problem) if isinstance(problem, ValueError) else 'Please fill in every field.'
        return render_template('index.html', result=None, form=form, error=message)

    # Turn text into 0/1, the same way we did with get_dummies in Colab.
    # Cherbourg (C) was the dropped port, so it is 0 in both port columns.
    row = {
        'Pclass': pclass,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Sex_male': 1 if sex == 'male' else 0,
        'Embarked_Q': 1 if embarked == 'Q' else 0,
        'Embarked_S': 1 if embarked == 'S' else 0,
    }

    # Put the row into a one-line table with the right column order
    passenger = pd.DataFrame([row], columns=COLUMNS)

    # Ask the model: 0 = did not survive, 1 = survived
    prediction = int(model.predict(passenger)[0])

    # How sure is the model? (chance of survival, as a percentage)
    chance = round(float(model.predict_proba(passenger)[0][1]) * 100, 1)

    result = {'survived': prediction == 1, 'chance': chance}
    return render_template('index.html', result=result, form=form, error=None)


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
