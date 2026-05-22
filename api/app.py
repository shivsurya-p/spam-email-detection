from flask import Flask, render_template, request
import joblib
import re

app = Flask(
    __name__,
    template_folder="../templates"
)

# LOAD MODEL
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# STORE HISTORY
history = []

# SPAM KEYWORDS
spam_keywords = [
    "free",
    "winner",
    "won",
    "cash",
    "lottery",
    "claim",
    "urgent",
    "offer",
    "click",
    "buy now",
    "discount",
    "bonus",
    "crypto",
    "loan",
    "verify",
    "otp",
    "gift",
    "prize",
    "congratulations",
    "reward",
    "limited offer",
    "earn money",
    "free recharge",
    "selected",
    "exclusive",
    "jackpot",
    "call now",
    "subscribe",
    "money"
]

# CLEAN TEXT
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)

    return text

# HOME PAGE
@app.route('/')
def home():

    return render_template('home.html')

# CHECKER PAGE
@app.route('/checker')
def checker():

    return render_template('checker.html')

# PREDICTION
@app.route('/predict', methods=['POST'])
def predict():

    # USER INPUT
    text = request.form['message']

    # CLEAN TEXT
    cleaned_text = clean_text(text)

    # SPAM SCORE
    score = 0

    for word in spam_keywords:

        if word in cleaned_text:

            score += 1

    # VECTORIZE
    data = vectorizer.transform([cleaned_text])

    # MODEL PREDICTION
    prediction = model.predict(data)

    # CONFIDENCE
    probability = model.predict_proba(data)

    confidence = round(max(probability[0]) * 100, 2)

    # FINAL RESULT
    if prediction[0] == 1 or score >= 2:

        result = "🚨 Spam Detected"

    else:

        result = "✅ Safe Message"

    # SAVE HISTORY
    history.append({
        "message": text,
        "result": result,
        "confidence": confidence
    })

    # SHOW RESULT
    return render_template(
        'result.html',
        prediction=result,
        confidence=confidence,
        history=history,
        message=text
    )

# RUN SERVER
if __name__ == "__main__":

    app.run(debug=True)