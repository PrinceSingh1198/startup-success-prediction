from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('/Users/princesingh/Desktop/startup_pred/random_forest_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    age_first_funding_year = float(request.form['age_first_funding_year'])
    age_last_funding_year = float(request.form['age_last_funding_year'])
    age_first_milestone_year = float(request.form['age_first_milestone_year'])
    age_last_milestone_year = float(request.form['age_last_milestone_year'])
    relationships = float(request.form['relationships'])
    funding_rounds = float(request.form['funding_rounds'])
    funding_total_usd = float(request.form['funding_total_usd'])
    milestones = float(request.form['milestones'])
    avg_participants = float(request.form['avg_participants'])

    # Create a list with the input values
    input_data = [
        age_first_funding_year,
        age_last_funding_year,
        age_first_milestone_year,
        age_last_milestone_year,
        relationships,
        funding_rounds,
        funding_total_usd,
        milestones,
        avg_participants
    ]

    # Make a prediction using the loaded model
    prediction = model.predict([input_data])[0]

    # Map the predicted label to a meaningful output
    if prediction == 1:
        result = 'Acquired'
    else:
        result = 'Closed'

    # Render the prediction result
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
