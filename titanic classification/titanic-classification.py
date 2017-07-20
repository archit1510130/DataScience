import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import time
import seaborn as sns



df=pd.read_csv("train.csv",index_col=None)


# so at first ....we see here are too many categorical data....so now 
# we have to change all the categorical data in to numerical data using cat.codes 
# because cat_codes uses the realtionship b/w them.

# now first of all i m going to drop some columns that would be not useful in prediction
dropcolumns=['Fare','Cabin','Embarked','Ticket','Name','PassengerId']
# as we can see these column does not have any importance in the survival of any passenger
# no importance of FARE 
# no importance of what type of CABIN he has and so on
df.drop(dropcolumns,axis=1,inplace=True)
#so here we can see we have two columns for categorical data
#df.Sex=df.Sex.astype('category').cat.codes
#print(df.Sex)# male=1 female=0







pred_col=['Pclass','Sex','Age','SibSp','Parch']
# one thing or
# as we can see SibSp and parch both are telling the same thing 
df['pariwar']=df['SibSp']+df['Parch']
df.pariwar.loc[df.pariwar>1]=1
df.pariwar.loc[df.pariwar==0]=0
#print(df.pariwar)
pred_col=['Pclass','Sex','Age','pariwar']
# so here we left only 4 columns so lets do some visulization on these 4 columns

# first on pcclass

fig,(axis1,axis2)=plt.subplots(1,2,figsize=(10,5))
#sns.countplot(x="Pclass",data=df,ax=axis1)
#sns.factorplot("Pclass",'Survived',data=df)

pdummy=pd.get_dummies(df['Pclass'])     
pdummy.columns=(['class1','class2','class3'])
pdummy.drop('class3',axis=1,inplace=True)
df=df.join(pdummy)
df.drop('Pclass',axis=1,inplace=True)
#print(df.columns)


# now come to the sex
#print(df.groupby("Sex").count()) so male > female
pds=df[["Sex","Survived"]].groupby(["Sex"]).sum()
# go to 
def get_person(passe):
    age,sex=passe
    return "child" if age<16 else sex
df["person"]=df[["Age","Sex"]].apply(get_person,axis=1)
pds=pd.get_dummies(df.person)
pds.columns=(["child","female","male"])
pds.drop("male",axis=1,inplace=True)
df=df.join(pds)
#print(df.columns)
df.drop(["person","Sex","SibSp","Parch"],axis=1,inplace=True)
#print(df.columns)




# age 


size=df.Age.isnull().sum()
random_list=np.random.randint(df.Age.mean()-df.Age.std(),df.Age.mean()+df.Age.std(),size=size)
df.Age[np.isnan(df.Age)]=random_list
df.Age=df.Age.astype('int')
df["newage"]=pd.cut(df.Age,5)
#print(df[["newage","Survived"]].groupby("newage").sum())

ads=pd.get_dummies(df["newage"])
ads.columns=(["bht chote age","chote age","medium age","high","buddhe"])
ads.drop(["high","buddhe"],axis=1,inplace=True)
df=df.join(ads)
df.drop(["newage","Age"],axis=1,inplace=True)


# now do some machine learing

from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression


pred_col=["pariwar","class1","class2","child","female","bht chote age","chote age","medium age"]
x=df[pred_col]
y=df.Survived




def class_model_gridsearchCV(model,param_grid,x,y):
    clf=GridSearchCV(model,param_grid,cv=10,scoring='accuracy')
    clf.fit(x,y)
    print(clf.best_params_)
    print(clf.best_score_)
    
def classification_model(x,y):
    result={}
    def test_model(clf):
        scores=cross_val_score(clf,x,y,cv=10,scoring='accuracy')
        return scores.max()
    clf=KNeighborsClassifier(n_neighbors=3)
    result["KNEIGHBORS"]=test_model(clf)
    
    clf=RandomForestClassifier(n_estimators=100)
    result["RANDOMFOREST"]=test_model(clf)
    
    clf=DecisionTreeClassifier()
    result["DECISIONTREE"]=test_model(clf)
        
    result=pd.DataFrame.from_dict(result,orient='index')
    result.columns=["accuracy"]
    result=result.sort(columns=["accuracy"],ascending=False)
    #result.plot(kind="bar")
    
    print(result.head(3))


rndm=RandomForestClassifier(n_estimators=100)
knn=KNeighborsClassifier(n_neighbors=3)
dc = DecisionTreeClassifier()
lr=LogisticRegression()
sv=svm.SVC()


classification_model(x,y)






