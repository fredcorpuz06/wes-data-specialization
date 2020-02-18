import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix
import pydotplus # for graphing

# Load the dataset
df = pd.read_csv("data/tree_addhealth.csv")
df.columns = map(str.upper, data.columns)
df["MALE"] = df['BIO_SEX'] == 1
df = df.dropna()
 
df.dtypes
df.describe()

# Split into training and testing sets
# TODO: just unselect vars
X = df[[
    'BIO_SEX','HISPANIC','WHITE','BLACK','NAMERICAN','ASIAN','age','ALCEVR1',
    'ALCPROBS1','marever1','cocever1','inhever1','cigavail','DEP1','ESTEEM1',
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

# Build model on training data
classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)

pred = classifier.predict(X_test)

print(confusion_matrix(y_test,pred))
print(accuracy_score(y_test, pred))

# Displaying the decision tree
# TODO: find different way to graph a tree
tree_data = export_graphviz(classifier, out_file=None)
graph = pydotplus.graph_from_dot_data(tree_data)

# Save decision tree graph as a png file 
with open('tree_out.png', 'wb') as f:
    f.write(graph.create_png())





