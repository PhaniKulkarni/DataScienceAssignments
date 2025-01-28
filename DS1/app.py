from flask import Flask, request, jsonify
import pickle
from data_cleaning import clean_text  # Import the cleaning function

# Initialize Flask app
app = Flask(__name__)

# Load the model and vectorizer
with open("./models/model.pkl", "rb") as model_file:
    model, vectorizer = pickle.load(model_file)

@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Input: JSON with field `review_text`.
    Output: JSON with field `sentiment_prediction` ("positive"/"negative").
    """
    try:
        # Extract JSON data
        data = request.get_json()

        if "review_text" not in data:
            return jsonify({"error": "Field `review_text` is required"}), 400

        # Input preprocessing
        raw_text = data["review_text"]
        cleaned_text = clean_text(raw_text)  # Apply cleaning
        review_vectorized = vectorizer.transform([cleaned_text]).toarray()

        # Predict sentiment
        prediction = model.predict(review_vectorized)[0]
        print(prediction)

        return jsonify({
            "sentiment_prediction": prediction,
            "cleaned_text": cleaned_text  # Optional: Include cleaned text for debugging
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
