import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoLarsCV
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv("data/tree_addhealth.csv")
df = df.dropna()
df.columns = map(str.upper, df.columns)

df.dtypes
df.describe()

# Data Management
recode1 = {1:1, 2:0}
df['MALE']= df['BIO_SEX'].map(recode1)

# Select predictor variables and target variable as separate data sets  
X = df[[
    'MALE','HISPANIC','WHITE','BLACK','NAMERICAN','ASIAN','AGE','ALCEVR1',
    'ALCPROBS1','MAREVER1','COCEVER1','INHEVER1','CIGAVAIL','DEP1','ESTEEM1',
    'VIOL1','PASSIST','DEVIANT1','GPA1','EXPEL1','FAMCONCT','PARACTV','PARPRES'
]]

y = df.SCHCONN1

# standardize predictors to have mean=0 and sd=1
X = X.apply(lambda x: scale(x.astype("float64")), axis=0)

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.4, random_state=1234
)


print(X_train.shape)
print(X_test.shape) 
print(y_train.shape) 
print(y_test.shape)

# specify the lasso regression model
classifier = LassoLarsCV(cv=10, precompute=False)
cli.fit(X_train, y_train)

# print variable names and regression coefficients
dict(zip(predictors.columns, model.coef_))

# plot coefficient progression
m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')

# plot mean square error for each fold
m_log_alphascv = np.log10(model.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, model.cv_mse_path_, ':')
plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()
plt.xlabel('log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')
         

# MSE from training and test data
train_error = mean_squared_error(tar_train, model.predict(pred_train))
test_error = mean_squared_error(tar_test, model.predict(pred_test))
print ('training data MSE')
print(train_error)
print ('test data MSE')
print(test_error)

# R-square from training and test data
rsquared_train=model.score(pred_train,tar_train)
rsquared_test=model.score(pred_test,tar_test)
print ('training data R-square')
print(rsquared_train)
print ('test data R-square')
print(rsquared_test)
