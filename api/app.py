from flask import Flask, render_template, request
import joblib

app = Flask(
    __name__,
    template_folder="../templates")

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

history = []

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/checker')
def checker():
    return render_template('checker.html')

@app.route('/predict', methods=['POST'])
def predict():

    text = request.form['message']

    data = vectorizer.transform([text])

    prediction = model.predict(data)

    probability = model.predict_proba(data)

    confidence = round(max(probability[0]) * 100, 2)

    if prediction[0] == 1:
        result = "🚨 Spam Message"
    else:
        result = "✅ Not Spam"

    history.append({
        "message": text,
        "result": result,
        "confidence": confidence
    })

    return render_template(
        'result.html',
        prediction=result,
        confidence=confidence,
        history=history
    )

if __name__ == "__main__":
    app.run()