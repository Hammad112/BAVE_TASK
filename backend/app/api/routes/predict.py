import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from app.models import PredictionResponse,PredictionRequest

# Load the trained Logistic Regression model and the TF-IDF vectorizer
# These should be loaded outside the route functions to avoid reloading on each request
model: LogisticRegression = joblib.load("./logistic_regression_model.pkl")
tfidf_vectorizer: TfidfVectorizer = joblib.load("./tfidf_vectorizer.pkl")

# Create the APIRouter instance
router = APIRouter()

# Prediction endpoint
@router.post("/", response_model=PredictionResponse)
def predict_category(request: PredictionRequest) -> Any:
    """
    Predict the category based on the title and description.
    """
    # Combine title and description
    combined_text = ' '.join([request.title, request.description])

    # Vectorize the combined text using the loaded TF-IDF vectorizer
    vectorized_text = tfidf_vectorizer.transform([combined_text])

    # Predict the category using the loaded Logistic Regression model
    prediction = model.predict(vectorized_text)

    # Convert the prediction to a human-readable format
    prediction_label = prediction[0]

    return PredictionResponse(category=prediction_label)

# Get prediction by ID (if storing predictions in a database)
@router.get("/{id}", response_model=PredictionResponse)
def get_prediction(id: uuid.UUID) -> Any:
    """
    Get prediction by ID.
    """
    # Here you would normally fetch the prediction from the database using the ID
    # Example:
    # prediction = session.get(Prediction, id)
    # if not prediction:
    #     raise HTTPException(status_code=404, detail="Prediction not found")

    raise HTTPException(status_code=501, detail="Fetching by ID not implemented")

# Additional CRUD routes can be implemented similarly, depending on your requirements
