
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
url = 'lungcancer.csv'  # Replace 'your_dataset.csv' with the actual path or URL of your CSV file
df = pd.read_csv(url)

# Cleaning the data
# Assuming there might be missing values, let's handle them by dropping rows with missing values
df = df.dropna()

# Map 'YES' to 2 and 'NO' to 1
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 2, 'NO': 1})

# Assuming there might be categorical variables that need to be encoded
# If 'GENDER' is a categorical variable, you might want to encode it
# For example, using one-hot encoding:
df = pd.get_dummies(df, columns=['GENDER'], drop_first=True)

df['AGE'] = (df['AGE'] - df['AGE'].min()) / (df['AGE'].max() - df['AGE'].min())


# Select features (independent variables) and target variable
X = df.drop(['LUNG_CANCER'], axis=1)  # Exclude the target variable
y = df['LUNG_CANCER']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

# Now, you can use the model to predict whether an individual has lung cancer
# For example, if you have a new set of features 'new_data', you can use:
# new_prediction = model.predict([new_data])
# print(f'Predicted Lung Cancer Status: {new_prediction[0]}')

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['NO', 'YES'], yticklabels=['NO', 'YES'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
