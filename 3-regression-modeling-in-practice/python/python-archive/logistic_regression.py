# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 09:54:18 2015

@author: jrose01
"""

import numpy
import pandas
import statsmodels.api as sm
import seaborn
import statsmodels.formula.api as smf 

# bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%.2f'%x)

data = pandas.read_csv('nesarc_pds.csv', low_memory=False)

#setting variables you will be working with to numeric
data['IDNUM'] = data['IDNUM'].convert_objects(convert_numeric=True)
data['TAB12MDX'] = data['TAB12MDX'].convert_objects(convert_numeric=True)
data['MAJORDEPLIFE'] = data['MAJORDEPLIFE'].convert_objects(convert_numeric=True)
data['NDSymptoms'] = data['NDSymptoms'].convert_objects(convert_numeric=True)
data['DYSLIFE'] = data['DYSLIFE'].convert_objects(convert_numeric=True)

data['SOCPDLIFE'] = data['SOCPDLIFE'].convert_objects(convert_numeric=True)
data['S3AQ3C1'] = data['S3AQ3C1'].convert_objects(convert_numeric=True)
data['AGE'] = data['AGE'].convert_objects(convert_numeric=True)
data['SEX'] = data['SEX'].convert_objects(convert_numeric=True)
data['S6Q1'] = data['S6Q1'].convert_objects(convert_numeric=True)
data['S6Q2'] = data['S6Q2'].convert_objects(convert_numeric=True)
data['S6Q3'] = data['S6Q3'].convert_objects(convert_numeric=True)
data['S6Q7'] = data['S6Q7'].convert_objects(convert_numeric=True)
data['S6Q61'] = data['S6Q61'].convert_objects(convert_numeric=True)
data['S6Q62'] = data['S6Q62'].convert_objects(convert_numeric=True)
data['S6Q63'] = data['S6Q63'].convert_objects(convert_numeric=True)
data['S6Q64'] = data['S6Q64'].convert_objects(convert_numeric=True)
data['S6Q65'] = data['S6Q65'].convert_objects(convert_numeric=True)
data['S6Q66'] = data['S6Q66'].convert_objects(convert_numeric=True)
data['S6Q67'] = data['S6Q67'].convert_objects(convert_numeric=True)
data['S6Q68'] = data['S6Q68'].convert_objects(convert_numeric=True)
data['S6Q69'] = data['S6Q69'].convert_objects(convert_numeric=True)
data['S6Q610'] = data['S6Q610'].convert_objects(convert_numeric=True)
data['S6Q611'] = data['S6Q611'].convert_objects(convert_numeric=True)
data['S6Q612'] = data['S6Q612'].convert_objects(convert_numeric=True)
data['S6Q613'] = data['S6Q613'].convert_objects(convert_numeric=True)

data['S3AQ3C1']=data['S3AQ3C1'].replace(99, numpy.nan)

# run this code to do the NDsymptoms regression
sub1=data[(data['AGE']<=25) & (data['CHECK321']==1) & (data['S3AQ3B1']==1) & 
(data['IDNUM']!=20346) & (data['IDNUM']!=36471) & (data['IDNUM']!=28724)]

# run this code to do all other regression analyses
sub1=data[(data['AGE']<=25) & (data['CHECK321']==1) & (data['S3AQ3B1']==1)]

c1 = sub1["MAJORDEPLIFE"].value_counts(sort=False, dropna=False)
print(c1)
c2 = sub1["AGE"].value_counts(sort=False, dropna=False)
print(c2)
# binary nictoine dependence
def NICOTINEDEP (x):
   if x['TAB12MDX']==1:
      return 1
   else: 
      return 0
sub1['NICOTINEDEP'] = sub1.apply (lambda x: NICOTINEDEP (x), axis=1)
print (pandas.crosstab(sub1['TAB12MDX'], sub1['NICOTINEDEP']))

# rename variables
sub1.rename(columns={'S3AQ3C1': 'numbercigsmoked'}, inplace=True)

c6 = sub1["numbercigsmoked"].value_counts(sort=False, dropna=False)
print(c6)

def PANIC (x1):
    if ((x1['S6Q1']==1 and x1['S6Q2']==1) or (x1['S6Q2']==1 and x1['S6Q3']==1) or 
    (x1['S6Q3']==1 and x1['S6Q61']==1) or (x1['S6Q61']==1 and x1['S6Q62']==1) or 
    (x1['S6Q62']==1 and x1['S6Q63']==1) or (x1['S6Q63']==1 and x1['S6Q64']==1) or 
    (x1['S6Q64']==1 and x1['S6Q65']==1) or (x1['S6Q65']==1 and x1['S6Q66']==1) or 
    (x1['S6Q66']==1 and x1['S6Q67']==1) or (x1['S6Q67']==1 and x1['S6Q68']==1) or 
    (x1['S6Q68']==1 and x1['S6Q69']==1) or (x1['S6Q69']==1 and x1['S6Q610']==1) or 
    (x1['S6Q610']==1 and x1['S6Q611']==1) or (x1['S6Q611']==1 and x1['S6Q612']==1) or 
    (x1['S6Q612']==1 and x1['S6Q613']==1) or (x1['S6Q613']==1 and x1['S6Q7']==1) or 
    x1['S6Q7']==1):
        return 1
    else:
        return 0
sub1['PANIC'] = sub1.apply (lambda x1: PANIC (x1), axis=1)
c7 = sub1["PANIC"].value_counts(sort=False, dropna=False)
print(c7)

############################################################################
#END DATA MANAGEMENT - TMI FOR LEARNERS - RUN THEN CUT MOST FOR SCREENSHOTS 
############################################################################

# depression
reg1 = smf.ols('NDSymptoms ~ MAJORDEPLIFE', data=sub1).fit()
print (reg1.summary())

# group means & sd
sub2 = sub1[['NDSymptoms', 'MAJORDEPLIFE']].dropna()
print ("Mean")
ds1 = sub2.groupby('MAJORDEPLIFE').mean()
print (ds1)
print ("Standard deviation")
ds2 = sub2.groupby('MAJORDEPLIFE').std()
print (ds2)

# bivariate bar graph
seaborn.factorplot(x="MAJORDEPLIFE", y="NDSymptoms", data=sub2, kind="bar", ci=None)
plt.xlabel('Major Life Depression')
plt.ylabel('Mean Number Nicotine Dependence Symptoms')

# adding number of cigarettes smoked  
# center quantitative IVs for regression analysis
sub1['numbercigsmoked_c'] = (sub1['numbercigsmoked'] - sub1['numbercigsmoked'].mean())
print (sub1['numbercigsmoked_c'].mean()) 

reg2 = smf.ols('NDSymptoms ~ MAJORDEPLIFE + numbercigsmoked_c', data=sub1).fit()
print (reg2.summary())

# dysphoria 
reg3 = smf.ols('NDSymptoms ~ DYSLIFE', data=sub1).fit()
print (reg3.summary())

# dysphoria & depression
reg4 = smf.ols('NDSymptoms ~ DYSLIFE + MAJORDEPLIFE', data=sub1).fit()
print (reg4.summary())


# dysphoria & depression + other covariates
# center age
sub1['age_c']=(sub1['AGE'] - sub1['AGE'].mean())
print (sub1['age_c'].mean()) 

reg5 = smf.ols('NDSymptoms ~ DYSLIFE + MAJORDEPLIFE + numbercigsmoked_c + age_c + SEX', data=sub1).fit()
print (reg5.summary())













