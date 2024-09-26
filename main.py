from src.data_preprocessing import preprocess_data
from src.model import train_model, predict, save_submission
from src.utils import load_data


# Load data
train_data, test_data = load_data()

# Preprocess data
X_train, y_train, X_test = preprocess_data(train_data, test_data)

# Train model
model = train_model(X_train, y_train)

# Make predictions on the test set
predictions = predict(model, X_test)

# Save the submission file
save_submission(predictions, test_data['PassengerId'], 'submissions/submission.csv')