
from pathlib import Path

import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Project folders
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "datasets" / "heart_disease.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "heart_disease_model.joblib"

# Dataset columns
FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

TARGET = "num"


def main():
    print("Loading Heart Disease dataset...")

    # Treat '?' as missing data
    data = pd.read_csv(DATA_PATH, na_values=["?"])

    # Convert all columns to numeric
    data = data.apply(pd.to_numeric, errors="coerce")

    # Remove rows where the target is missing
    data = data.dropna(subset=[TARGET])

    # Convert target into binary classification:
    # 0 = no disease
    # 1, 2, 3, 4 = disease present
    data[TARGET] = (data[TARGET] > 0).astype(int)

    X = data[FEATURES]
    y = data[TARGET]

    print(f"Total records: {len(data)}")
    print(f"Records with no disease: {(y == 0).sum()}")
    print(f"Records with disease: {(y == 1).sum()}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Build the ML pipeline
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nModel evaluation")
    print("----------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["No Disease", "Disease Present"],
            zero_division=0,
        )
    )

    # Save trained model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved successfully at:\n{MODEL_PATH}")


if __name__ == "__main__":
    main()
