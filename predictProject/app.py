from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('weather_model.joblib')
# Load the trained Regression model
regression_model = joblib.load('precip_model.joblib')
#=========================LOADING MODELS=======================================


@app.route('/', methods=['GET', 'POST'])
def predict_classification(): 

    if request.method == 'POST':
        # Get user input from the form
        prepicitation = float(request.form.get('prepicitation'))
        temp_max = float(request.form.get('temp_max'))
        temp_min = float(request.form.get('temp_min'))
        wind = float(request.form.get('wind'))

        # Make predictions using the loaded model
        input_data = [[prepicitation, temp_max, temp_min, wind]]
        prediction = model.predict(input_data)

        return render_template('index.html', prediction=prediction[0])

    return render_template('index.html', prediction=None)

pass

# New route for regression predictions
@app.route('/regression', methods=['GET', 'POST'])
def predict_regression():

    if request.method == 'POST':
        # Retrieve input data from the web form
        year = float(request.form['year'])
        month = float(request.form['month'])
        specific_humidity = float(request.form['specific_humidity'])
        relative_humidity = float(request.form['relative_humidity'])
        temperature = float(request.form['temperature'])

        # Make predictions using the loaded regression model
        input_data = [[year, month, specific_humidity, relative_humidity, temperature]]
        regression_prediction = regression_model.predict(input_data)

        # Return the regression prediction to the user (use a different template)
        return render_template('regression.html', regression_prediction=regression_prediction[0])

    return render_template('regression.html', regression_prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
    