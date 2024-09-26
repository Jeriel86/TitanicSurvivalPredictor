import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def preprocess_data(train_data, test_data):
    # Combine datasets for consistent preprocessing
    combined = pd.concat([train_data.drop('Survived', axis=1), test_data], sort=False)

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')

    # Impute 'Age' and 'Fare'
    combined['Age'] = imputer.fit_transform(combined[['Age']])
    combined['Fare'] = imputer.fit_transform(combined[['Fare']])

    # Fill missing values for 'Embarked'
    combined['Embarked'] = combined['Embarked'].fillna('S')

    # Drop columns that aren't useful or contain too many missing values
    combined = combined.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    # Encode categorical features
    le = LabelEncoder()
    combined['Sex'] = le.fit_transform(combined['Sex'])
    combined['Embarked'] = le.fit_transform(combined['Embarked'])

    # Feature scaling
    scaler = StandardScaler()
    combined[['Age', 'Fare']] = scaler.fit_transform(combined[['Age', 'Fare']])

    # Feature engineering: Adding FamilySize
    combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1

    # Split the data back into training and test sets
    X_train = combined[:len(train_data)]
    X_test = combined[len(train_data):]
    y_train = train_data['Survived']

    return X_train, y_train, X_test
