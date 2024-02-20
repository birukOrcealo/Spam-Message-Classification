import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

# Load the spam dataset
spam_data = pd.read_csv('spam.csv', usecols=['v1', 'v2'], encoding='latin-1')  # Only load the 'Type' and 'Message' columns
spam_data.columns = ['label', 'message']  # Rename columns for clarity

#convert the label data to binary format
spam_data['label']= spam_data['label'].map({'ham':0, 'spam':1})
spam_data.head()

y=spam_data['label']
x=spam_data['message']

#splite the data into training and testing data 
X_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# Create a text processing and classification pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),  # Convert messages to TF-IDF features
    ('logreg', LogisticRegression())  # Apply logistic regression
])

pipeline.fit(X_train, y_train)

y_pred=pipeline.predict(x_test)

print(accuracy_score(y_test,y_pred))
print(precision_score(y_test,y_pred))
print(recall_score(y_test,y_pred))
print(roc_auc_score(y_test, pipeline.predict_proba(x_test)[:,1]))

# Output the confusion matrix
classess=[0,1]
cm = confusion_matrix(y_test, y_pred)
result=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classess)
result.plot(cmap='Blues')
plt.show()