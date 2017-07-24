import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import time
import math

import seaborn as sns
df=pd.read_csv("data.csv",header=0)
#print(df.head())
#print(df.info()) here we know unmed:32 has all null values
df.drop(['Unnamed: 32','id'],axis=1,inplace=True)
#print(df.info())
FAST_DRAW=True


# above the data can be divided into three parts.lets divied the features according to their category
f_mean=list(df.columns[1:11])
f_se=list(df.columns[11:20])
f_worst=list(df.columns[21:31])

#now do data analysis for each and every part
df['diagnosis']=df['diagnosis'].astype('category').cat.codes #mapping the categorical data in to numerical
#print(df['diagnosis'])
df['diagnosis'].value_counts().reset_index()
#print(df)
# so here even rate is 37% ..that is enough good for predicting
# so here no feature scaling needed
#print(df.describe()) here i think normaliztion should be done

#now there is corelation graph ...for removing multicolineatity and for feature selction

'''corr = df[f_mean].corr() # .corr is used for find corelation
plt.figure(figsize=(14,14))
sns.heatmap(corr, cbar = True,  square =True, annot=True, fmt= '.2f',annot_kws={'size': 15},
           xticklabels= f_mean, yticklabels= f_mean,
           cmap= 'coolwarm')
'''           
           
# so here we remove some features because of multicolineraity
     
#['perimeter_mean','area_mean''concvity_mean','concave points_mean',]
     
df.drop(['perimeter_mean','area_mean','concavity_mean','concave points_mean'],axis=1,inplace=True)
#print(len(df.columns)) 

#again lets for f_se
'''corr=df[f_se].corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':15},xticklabels=f_se,yticklabels=f_se)
'''
# so here also

#print(f_se)
#['perimeter_se','area_se','concavity_se','concave points_se']
df.drop(['perimeter_se','area_se','concavity_se','concave points_se'],axis=1,inplace=True)

#now for f_worst
'''corr=df[f_worst].corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':10},xticklabels=f_worst,yticklabels=f_worst)
'''
#print(f_worst)

df.drop(['perimeter_worst','area_worst','concavity_worst','concave points_worst'],axis=1,inplace=True)
#print(len(df.columns))

pred_var=df.columns.tolist()
pred_var.pop(0)




#so now we all set with the features ....huraaaaaaahhhhhhhhhhh




#lets do machine learning part




from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

def drawPlots(model, X_train, X_test, y_train, y_test, wintitle='Figure 1'):
  # INFO: A convenience function that would be used to plot the decision curve for any high-dimensional data


  padding = 3
  resolution = 0.5
  max_2d_score = 0

  y_colors = ['#ff0000', '#00ff00']
  my_cmap = mpl.colors.ListedColormap(['#ffaaaa', '#aaffaa'])
  colors = [y_colors[i] for i in y_train]
  num_columns = len(X_train.columns)

  fig = plt.figure()
  fig.canvas.set_window_title(wintitle)
  
  cnt = 0
  for col in range(num_columns):
    for row in range(num_columns):
      # Easy out
      if FAST_DRAW and col > row:
        cnt += 1
        continue

      ax = plt.subplot(num_columns, num_columns, cnt + 1)
      plt.xticks(())
      plt.yticks(())

      # Intersection:
      if col == row:
        plt.text(0.5, 0.5, X_train.columns[row], verticalalignment='center', horizontalalignment='center', fontsize=12)
        cnt += 1
        continue


      # Only select two features to display, then train the model
      X_train_bag = X_train.ix[:, [row,col]]
      X_test_bag = X_test.ix[:, [row,col]]
      model.fit(X_train_bag, y_train)

      # Create a mesh to plot in
      x_min, x_max = X_train_bag.ix[:, 0].min() - padding, X_train_bag.ix[:, 0].max() + padding
      y_min, y_max = X_train_bag.ix[:, 1].min() - padding, X_train_bag.ix[:, 1].max() + padding
      xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                           np.arange(y_min, y_max, resolution))

      # Plot Boundaries
      plt.xlim(xx.min(), xx.max())
      plt.ylim(yy.min(), yy.max())

      # Prepare the contour
      Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
      Z = Z.reshape(xx.shape)
      plt.contourf(xx, yy, Z, cmap=my_cmap, alpha=0.8)
      plt.scatter(X_train_bag.ix[:, 0], X_train_bag.ix[:, 1], c=colors, alpha=0.5)


      score = round(model.score(X_test_bag, y_test) * 100, 3)
      plt.text(0.5, 0, "Score: {0}".format(score), transform = ax.transAxes, horizontalalignment='center', fontsize=8)
      max_2d_score = score if score > max_2d_score else max_2d_score

      cnt += 1

  print ("Max 2D Score: ", max_2d_score)
  #fig.set_tight_layout(True)
        
X=df[pred_var]
y=df.diagnosis

def model_selection(x,y):
    result={}
    parameter={}
    
    
    def class_model_gridsearchCV(model,param_grid):
        clf=GridSearchCV(model,param_grid,cv=10,scoring='accuracy')
        clf.fit(x,y)
        param=[clf.best_params_,clf.best_score_]
        return param
    
    
    def classification_model(model):
        scores=cross_val_score(model,x,y,cv=10,scoring='accuracy')
        score=[scores.mean()]
        return score
    
    
    clf=KNeighborsClassifier(n_neighbors=3)
    result["Kneighbors"]=classification_model(clf)
    param_grid={"n_neighbors":range(3,8)}
    parameter["kneighors"]=class_model_gridsearchCV(clf,param_grid)
    
    clf=RandomForestClassifier(n_estimators=101)
    param_grid={"n_estimators":range(101,108)}
    result["RandomForest"]=classification_model(clf)
    parameter["RANDOMFOREST"]=class_model_gridsearchCV(clf,param_grid)
    
    clf= svm.SVC()
    result["SVM"]=classification_model(clf)
    
    
    result=pd.DataFrame.from_dict(result,orient='index')
    result.columns=["score"]
    parameter=pd.DataFrame.from_dict(parameter,orient='index')
    parameter.columns=["parametr","score"]
    print(result)
    print(parameter)

    
    
model_selection(X,y)





 






  