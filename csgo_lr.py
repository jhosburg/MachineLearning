import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
df = pd.read_csv('csgo_round_snapshots.csv')

# Preprocess the data
df = pd.get_dummies(df, columns=['map'])

# Define features and target variable
features = df.drop(['round_winner'], axis=1)
target = df['round_winner']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)  # or a higher value
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['CT', 'T'], yticklabels=['CT', 'T'])
plt.title('Confusion Matrix for Logistic Regression')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Group by map and round winner, then count the occurrences
map_win_counts = df.groupby(['map', 'round_winner']).size().unstack().fillna(0)

# Plot the bar graph
sns.set(style="whitegrid")
map_win_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Map Wins for CT and T Side')
plt.xlabel('Map')
plt.ylabel('Number of Wins')
plt.legend(title='Round Winner', loc='upper right', bbox_to_anchor=(1.25, 1))
plt.show()
