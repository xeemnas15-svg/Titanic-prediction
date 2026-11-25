# Titanic Survival Predictor

A beginner data science project built during the **SheBuilds AI Fellowship**. I cleaned the Titanic dataset, analysed it, trained two machine learning models, and turned the best one into a small web app that predicts whether a passenger would have survived.

## What this project covers
1. **Data cleaning:** handling missing values and dropping columns that do not help
2. **Analysis and charts:** bar chart, histogram, pie charts and scatter plot
3. **Modelling:** Logistic Regression and Random Forest, compared on unseen data
4. **Saving the model:** the Random Forest saved with `pickle`
5. **Web app:** a Flask backend with an HTML and CSS front end

## Project structure
```
titanic-survival-predictor/
    app.py                  Flask backend (loads the model and makes predictions)
    train_model.py          Cleans the data, trains both models, saves the Random Forest
    titanic_rf_model.pkl    The saved model (created by train_model.py)
    requirements.txt        Packages the web app needs (also used by Render)
    requirements-notebook.txt  Extra packages for the notebook charts
    data/
        Titanic-Dataset.csv The dataset (891 passengers)
    templates/
        index.html          The web page
    notebooks/              The Colab notebook (cleaning, charts, models)
    screenshots/            Pictures of the web app
```

## Data cleaning
| Column | Problem | Fix |
|---|---|---|
| Age | 177 missing | Filled with the median age |
| Embarked | 2 missing | Filled with the most common port |
| Cabin | 687 missing (about 77%) | Dropped the column |

PassengerId, Name and Ticket were dropped because they are only labels. Sex and Embarked were converted into 0/1 columns.

## Key findings
- 61.6% of passengers died and 38.4% survived.
- Women survived at about 74%, men at about 19%.
- Survival by class: 1st class 62%, 2nd class 47%, 3rd class 24%.
- Small children had a better chance of survival.
- The model relied most on sex, then fare, age and passenger class.

## Model results
Data split: 80% training (712 passengers) and 20% testing (179 passengers).

| Model | Accuracy |
|---|---|
| Logistic Regression | 81% |
| Random Forest | 82% |

The Random Forest was slightly better, so it was saved and used in the web app.

## How to run the web app

1. Clone or download this repository and open the folder in a terminal.
2. Create a virtual environment and install the packages:
   ```
   python -m venv venv
   venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
   On Mac or Linux, use `venv/bin/python` instead of `venv\Scripts\python.exe`.
3. Train and save the model on your computer (this avoids scikit-learn version warnings):
   ```
   venv\Scripts\python.exe train_model.py
   ```
4. Start the app:
   ```
   venv\Scripts\python.exe app.py
   ```
5. Open (https://titanic-prediction-qxi2.onrender.com/) in your browser.

## Using the app
Choose the ticket class, sex, age, family members aboard, fare (in 1912 British pounds) and port of boarding, then press **Check my fate**. The app shows whether the model predicts survival and the percentage chance.

## Deploying on Render
1. Push this repo to GitHub, including `titanic_rf_model.pkl`.
2. On render.com, choose **New**, then **Web Service**, and connect the repo.
3. Use these settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
4. Create the service and wait for the build to finish.

The model was trained with scikit-learn 1.9.1, so `requirements.txt` pins that version.

## Limitations
- 177 missing ages were filled with the median, so many passengers look the same age.
- The two models differ by only 1 percentage point, so neither is clearly better.
- Results come from one train/test split and could change slightly with a different split.
- The app predicts patterns in historical data. It is a learning project, not a real prediction about people.

## Tools used
Python, pandas, scikit-learn, matplotlib, seaborn, Flask, HTML and CSS, Google Colab, VS Code

## Author
Zainab Muhammad Nasir, SheBuilds AI Fellowship (Brain Builders Youth Development Initiative, Kaduna)
