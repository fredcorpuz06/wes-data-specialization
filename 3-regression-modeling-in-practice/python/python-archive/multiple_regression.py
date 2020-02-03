# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:46:09 2015

@author: jrose01
"""
import numpy
import pandas
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn

# bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%.2f'%x)

# increase # of rows of data printed 
pandas.set_option('display.max_rows', 1000)
#call in data set
data = pandas.read_csv('gapminder.csv')

data.columns

# print variable types for all variables
data.dtypes

# numeric variables that are read into python
# from the csv file as strings (objects) with empty cells should be 
# converted back to numeric format using convert_objects function
data['internetuserate'] = data['internetuserate'].convert_objects(convert_numeric=True)
data['incomeperperson'] = data['incomeperperson'].convert_objects(convert_numeric=True)
data['urbanrate'] = data['urbanrate'].convert_objects(convert_numeric=True)
data['lifeexpectancy'] = data['lifeexpectancy'].convert_objects(convert_numeric=True)
data['relectricperperson'] = data['relectricperperson'].convert_objects(convert_numeric=True)
data['alcconsumption'] = data['alcconsumption'].convert_objects(convert_numeric=True)
data['femaleemployrate'] = data['femaleemployrate'].convert_objects(convert_numeric=True)
data['armedforcesrate'] = data['armedforcesrate'].convert_objects(convert_numeric=True)
data['co2emissions'] = data['co2emissions'].convert_objects(convert_numeric=True)

data['co2emissions']=pandas.qcut(data.co2emissions, 4, labels=["1=25%tile", "2=50%tile", "3=75%tile", "4=100%tile"])
c3=data['co2emissions'].value_counts(sort=False, dropna=True)
print(c3)
data['logvar']=numpy.log(data['incomeperperson'])
c4=data['incomeperperson'].value_counts(sort=False, dropna=True)
print(c4)
# listwise deletion of missing values
sub1 = data[['urbanrate', 'femaleemployrate', 'internetuserate']].dropna()

# run the 2 scatterplots together to get both linear and second order fit lines
# Using seaborn package
scat1 = seaborn.regplot(x="urbanrate", y="femaleemployrate", scatter=True, data=sub1)
plt.xlabel('Urbanization Rate')
plt.ylabel('Female Employment Rate')

# fit second order polynomial
scat1 = seaborn.regplot(x="urbanrate", y="femaleemployrate", scatter=True, order=2, data=sub1)
plt.xlabel('Urbanization Rate')
plt.ylabel('Female Employment Rate')

# center quantitative IVs for regression analysis
sub1['urbanrate_c'] = (sub1['urbanrate'] - sub1['urbanrate'].mean())
sub1['internetuserate_c'] = (sub1['internetuserate'] - sub1['internetuserate'].mean())
sub1[["urbanrate_c", "internetuserate_c"]].describe()

# Fit and summarize OLS model
del I
reg1 = smf.ols('femaleemployrate ~ urbanrate_c', data=sub1).fit()
print (reg1.summary())

# polynomial regression
reg2 = smf.ols('femaleemployrate ~ urbanrate_c + I(urbanrate_c**2)', data=sub1).fit()
print (reg2.summary())

# adding internet use rate
reg3 = smf.ols('femaleemployrate  ~ urbanrate_c + I(urbanrate_c**2) + internetuserate_c', 
               data=sub1).fit()
print (reg3.summary())

#Q-Q plot for normality
fig4=sm.qqplot(reg3.resid, line='r')

# simple plot of residuals
stdres=pandas.DataFrame(reg3.resid_pearson)
plt.plot(stdres, 'o', ls='None')
l = plt.axhline(y=0, color='r')
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')


# additional regression diagnostic plots
fig2 = plt.figure(figsize=(12,8))
fig2 = sm.graphics.plot_regress_exog(reg3,  "internetuserate_c", fig=fig2)

# leverage plot
fig3=sm.graphics.influence_plot(reg3, size=8)
print(fig3)



# include interactions
print ("OLS regression model predicting internet use with interaction")
reg2 = smf.ols('internetuserate ~ urbanrate_c*incomeperperson_c', data=data).fit()
print (reg2.summary())

# graph interaction with different markers for interaction variable
sns.lmplot(x="urbanrate", y="internetuserate", hue="smoker", data=tips,
           markers=["o", "x"], palette="Set1");
plt.xlabel('Urban Rate (Mean = 0)')
plt.ylabel('Internet Use')




# Fit regression model (using the natural log of one of the regressors)
#results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data=dat).fit()

# Inspect the results
#print results.summary()