import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("data/tree_addhealth.csv")
df = df.dropna()
df.columns = map(str.upper, df.columns)

df.dtypes
df.describe()

# Split into training and testing sets
# TODO: just unselect vars
X = df[[
    'BIO_SEX','HISPANIC','WHITE','BLACK','NAMERICAN','ASIAN','AGE','ALCEVR1',
    'ALCPROBS1','MAREVER1','COCEVER1','INHEVER1','CIGAVAIL','DEP1','ESTEEM1',
    'VIOL1','PASSIST','DEVIANT1','SCHCONN1','GPA1','EXPEL1','FAMCONCT',
    'PARACTV', 'PARPRES'
]]
y = df.TREG1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.4, random_state=1234
)


print(X_train.shape)
print(X_test.shape) 
print(y_train.shape) 
print(y_test.shape)

# Build random forest model on training data
classifier = RandomForestClassifier(n_estimators=25)
classifier.fit(X_train,y_train)

pred = classifier.predict(X_test)

print(confusion_matrix(y_test,pred))
print(accuracy_score(y_test, pred))

# Fit an Extra Trees model to the data
classifier2 = ExtraTreesClassifier()
classifier2.fit(X_train,y_train)

# Display the relative importance of each attribute
print(classifier2.feature_importances_)

# Running a different number of trees, observe accuracy
n_trees = list(range(1, 26))

def my_random_forest(n_tree):
   classifier = RandomForestClassifier(n_estimators=n_tree)
   classifier.fit(X_train, y_train)
   pred = classifier.predict(X_test)
   return accuracy_score(y_test, pred)

accuracies = list(map(my_random_forest, n_trees))

plt.plot(n_trees, accuracies)
plt.show()
