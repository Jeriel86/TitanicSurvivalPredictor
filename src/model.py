import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd


def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Cross-validate
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {scores.mean():.4f}")

    # Train the model
    model.fit(X_train, y_train)

    # Save the trained model to a file
    joblib.dump(model, 'models/model_v1.joblib')

    return model


def predict(model, X_test):
    # Make predictions
    return model.predict(X_test)


def save_submission(predictions, passenger_ids, filename):
    # Create the submission DataFrame
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })

    # Save to CSV
    submission.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")
