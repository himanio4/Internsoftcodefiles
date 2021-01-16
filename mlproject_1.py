import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("advertising.csv")
data.head()
fig, axs=plt.subplots(1,3,sharey=True)
data.plot(kind='scatter',x='TV',y='Sales',ax=axs[0],figsize=(16,8))
data.plot(kind='scatter',x='Radio',y='Sales',ax=axs[1])
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[2])

feature_cols=['TV']
x=data[feature_cols]
y=data.Sales

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x,y)

print(lr.coef_)

result=6.97+0.0554*50
print(result)

x_new=pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
x_new.head()

preds=lr.predict(x_new)
preds


data.plot(kind='scatter',x='TV',y='Sales')
plt.plot(x_new,preds,c='red',linewidth=3)


import statsmodels.formula.api as smf
lr=smf.ols(formula='Sales ~ TV',data=data).fit()
lr.conf_int()
lr.pvalues
lr.rsquared


feature_cols=['TV','Radio','Newspaper']
x=data[feature_cols]
y=data.Sales

lr=LinearRegression()
lr.fit(x,y)

print(lr.intercept_)
print(lr.coef_)


lm=smf.ols(formula='Sales ~ TV+Radio+Newspaper',data=data).fit()
lm.conf_int()
lm.summary()
