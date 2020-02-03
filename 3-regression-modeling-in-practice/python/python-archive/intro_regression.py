# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:46:09 2015

@author: jrose01
"""
import numpy as numpyp
import pandas as pandas
import statsmodels.api
import seaborn
import scipy
import statsmodels.formula.api as smf

# bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%.2f'%x)

#call in data set
data = pd.read_csv('P:/QAC/qac201/Develop/2015/Data sets/Gapminder (moved out of studies for 2015)/Data/gapminder.csv')

# convert variables to numeric format using convert_objects function
data['internetuserate'] = data['internetuserate'].convert_objects(convert_numeric=True)
data['urbanrate'] = data['urbanrate'].convert_objects(convert_numeric=True)

data_clean=data.dropna()

print ('association between urbanrate and internetuserate')
print (scipy.stats.pearsonr(data_clean['urbanrate'], data_clean['internetuserate']))

scat1 = seaborn.regplot(x="urbanrate", y="internetuserate", scatter=True, data=data)
plt.xlabel('Urbanization Rate')
plt.ylabel('Internet Use Rate')
plt.title ('Scatterplot for the Association Between Urban Rate and Internet Use Rate')
print(scat1)

print ("OLS regression model for the association between urban rate and internet use rate")
reg1 = smf.ols('internetuserate ~ urbanrate', data=data).fit()
print (reg1.summary())

