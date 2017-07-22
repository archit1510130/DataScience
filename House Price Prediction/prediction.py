import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import time
import seaborn as sns
from scipy.stats import skew
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

#print(train.info())
# as wee see here there are some features in the dataset that have too small num
# of values so they can easily remove than from the dataset


# so now firstly i m going to create a co-relation matrix that would tell abt the
# multicolinearity and other....

'''
corr=train.corr()
cols=corr.nlargest(15,'SalePrice')['SalePrice'].index
#print(type(cols)) will give a series ..
cm=train[cols].corr()
plt.figure(figsize=(14,14))
sns.heatmap(cm,cbar=True,square=True,annot=True,fmt=".2f",xticklabels=cols,yticklabels=cols)
'''
# here are some features which give the same info
# [GarageCars,GarageArea]
#[TotalBsmtSF,1stFlrSF]
#[TotRmsAbvGrd,GrLivArea]
# also we note the most corelated varibles with saleprice


# now lets discuss abt the missing data
t=train.isnull().sum().sort_values(ascending=False)
p=(train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
#print(p)


# so here we see many of features miss some values
#useless features=[PoolQC,MiscFeature,Alley,Fence,FireplaceQu]
train.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageArea','1stFlrSF','TotRmsAbvGrd'],axis=1,inplace=True)
test.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageArea','1stFlrSF','TotRmsAbvGrd'],axis=1,inplace=True)



# now data preprocessing
#price=pd.DataFrame({'price':train['SalePrice'],"logdata":np.log1p(train['SalePrice'])})
#price.hist()

# so log transform gratly affected the our result
train["SalePrice"]=np.log1p(train["SalePrice"])

numcols=train.dtypes[train.dtypes!='object'].index

sk=train[numcols].apply(lambda x: skew(x.dropna()))
sk=sk[sk>0.75].index
train[sk]=np.log1p(train[sk])
test[sk]=np.log1p(test[sk])
train=train.fillna(train.mean())
test=test.fillna(test.mean())

#print(train.head())


# now working with dummy variables
# as i can see here the data normalizing would play also a critical part because some features have large values than others 

# bt ill look in to this later


print(train.shape)
print(test.shape)
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
print(all_data.shape)
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

# working with models

from sklearn.linear_model import Ridge,Lasso,LassoCV

#X_train=train.drop('SalePrice',axis=1)


#from sklearn import preprocessing
#normalized_X = preprocessing.normalize(X_train)
#X_train=pd.DataFrame(normalized_X,columns=list(X_train.columns.tolist())) does not working well
from sklearn.model_selection import cross_val_score
#y=train["SalePrice"]

def rmse_cv(model):
    rmse=np.sqrt(-cross_val_score(model,X_train,y,scoring="neg_mean_squared_error",cv=10))
    return (rmse)
    

# as we know for regularization parameter if its high then there will be overfitting
# else if its low then there will be bad acccuarcy in final result
# so for checking the value of that parameter 
# we take some values and check with the help with the cross validation
 

def Ridge_model():
    alphas=[.02,.05,.50,1,2,3,5,7,10,15,20,30,50,60]
    score_ridge=[rmse_cv(Ridge(alpha=alpha)).mean() for alpha in alphas]
#print(score_ridge)
    score_ridge=pd.Series(score_ridge,index=alphas)
#print(score_ridge.sort_values())
# so according to this best alpha would be 5
    score_ridge.plot()
    print(score_ridge.min())

#Ridge_model()
# so by ridge regression we got about 0.1233 rmse ...that pretty good



#so now lets try with the LASSO MODEL:
# as we know abt the lasso model ...for the value of alpha its just opposite then the 
# ridge model and so for the low alpha it would provide better result for us...
# for complete detail about the LASSO pls visit the ANALYTICSVIDYA.com
# as we know lasso also does feaature selection so we are also going to see which feature has most important according to lasso




def LASSOMODEL():
    alphas=[0.000410,0.000425,0.000420,0.000435,0.000430,0.000440]
    model_lasso = LassoCV(alphas =alphas).fit(X_train, y)
    #print(rmse_cv(model_lasso).mean())
    coef = pd.Series(model_lasso.coef_, index = X_train.columns)
    # so best alpha would be 0.000425 and accuracy abt .120001 yeah its doing well compare then the ridge 
    #print(str(sum(coef!=0)),str(sum(coef==0)))
    # so 111 columns retain and 157 columns rejected
    lasso_preds = np.expm1(model_lasso.predict(X_test))
    #print(lasso_preds)
    solution=pd.DataFrame({"id":test.Id,"SalePrice":lasso_preds})
    solution.to_csv('answerfinal.csv',index=False)
    
#LASSOMODEL()    


# so by lasso we got .1200 accuracy ..........its pretty good bt now trying with the xgboost model


#XGBOOST:
    # BOOSTING :
        #XGBBOST model is based on boosting ....boosting is sequential process ....in which first 
         #we we apply the algorithm ontrain data and those who has large errors we build a another model
         # then apply the same algorithm for that and in the en final result is ensembles of all of them:
             













