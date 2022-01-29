import math
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go

#  load dataset
df_tr = pd.read_csv('aug_train.csv')
df_te = pd.read_csv('aug_test.csv')

#  information about given data
df_tr.head()
df_tr.shape

#  checking null values in given train dataset
df_tr.isnull().sum()

#  Dropping rows with missing values would significantly reduce the dataset and
#  underfitting could occur - too many missing values
#  Demonstrating a few other ways to handle with missing values: 
#  fill missing values with 0, fill missing values with mode, fill missing values with median.
df_tr['company_size']=[1 if company_size=="<10" 
                       else 2 if company_size=="10/49"
                       else 3 if company_size=="50-99"
                       else 4 if company_size=="100-500"        #do this manually to explicitly tell the model how to group company_size
                       else 5 if company_size=="500-999"
                       else 6 if company_size=="1000-4999"
                       else 7 if company_size=="5000-9999"
                       else 8 if company_size=="10000+"
                       else 0 for company_size in df_tr["company_size"]]

experience_new = {'<1': 0, '>20': 21}                           #do this manually to explicitly tell the model how to group experience_new
df_tr = df_tr.replace({"experience": experience_new})           

#  median is used if the data comprised of outliers or more frequent value, we can see that from boxplot
df_tr['experience'].fillna(df_tr['experience'].median(), inplace=True)

df_tr['last_new_job'] = df_tr['last_new_job'].fillna("NA")
lnj = {'>4': 5, 'never': 6}                                     #do this manually to explicitly tell the model how to group last_new_job
df_tr = df_tr.replace({"last_new_job": lnj})

df_tr['gender'] = df_tr['gender'].fillna("NA")                  #missing values fill with NA

#  missing values fill with mode because mode is used when the data having more
#  occurences of a particular value or more frequent value
#  This method is used for the categorical variable
df_tr['company_type'].fillna(df_tr['company_type'].mode()[0], inplace=True)         
df_tr['major_discipline'].fillna(df_tr['major_discipline'].mode()[0], inplace=True)       
df_tr['education_level'].fillna(df_tr['education_level'].mode()[0], inplace=True)
df_tr['enrolled_university'].fillna(df_tr['enrolled_university'].mode()[0], inplace=True)

#  the same procedure for test dataset
df_te.isnull().sum()

df_te['company_size']=[1 if company_size=="<10"
                       else 2 if company_size=="10/49"
                       else 3 if company_size=="50-99"
                       else 4 if company_size=="100-500"
                       else 5 if company_size=="500-999"
                       else 6 if company_size=="1000-4999"
                       else 7 if company_size=="5000-9999"
                       else 8 if company_size=="10000+"
                       else 0 for company_size in df_te["company_size"]]

experience_new = {'<1': 0, '>20': 21}
df_te = df_te.replace({"experience": experience_new})
df_te['experience'].fillna(df_te['experience'].median(), inplace=True)

df_te['last_new_job'] = df_te['last_new_job'].fillna("NA")
lnj = {'>4': 5, 'never': 6}
df_te = df_te.replace({"last_new_job": lnj})

df_te['gender'] = df_te['gender'].fillna("NA")
df_te['company_type'].fillna(df_te['company_type'].mode()[0], inplace=True)
df_te['major_discipline'].fillna(df_te['major_discipline'].mode()[0], inplace=True)
df_te['education_level'].fillna(df_te['education_level'].mode()[0], inplace=True)
df_te['enrolled_university'].fillna(df_te['enrolled_university'].mode()[0], inplace=True)




#  EDA - Exploratory Data Analysis 

#  Distribution of target - 75% candidates are not searching for new job
plt.figure(1)

bars = df_tr['target'].value_counts()
graph = plt.bar(['Not searching for new job', 'Searching for new job'], bars.values)
plt.title('Distribution of target')
plt.xlabel('Target values')
plt.ylabel('Count')
i = 0
for p in graph:
    w = p.get_width()
    h = p.get_height()
    x, y = p.get_xy()
    plt.text(x+w/2,y+h*1.01+1,str(round(100*bars[i]/df_tr['target'].count()))+'%', ha = 'center')
    i += 1

plt.close() #comment this line if you want to show figure 1

#  The top 4 cities represent more than half of the candidates
plt.figure(2)

city = df_tr['city'].value_counts().head(5)
graph = plt.bar([str(i) for i in city.keys()], city.values,color = 'red')
plt.title('Cities with the most candidates (Top 5)')
plt.xlabel('City')
plt.ylabel('Count')
i = 0
for p in graph:
     w = p.get_width()
     h = p.get_height()
     x, y = p.get_xy()
     plt.text(x+w/2,y+h*1.01+1,str(round(100*city.values[i]/df_tr['target'].count()))+'%',ha = 'center')
     i += 1

plt.close() #comment this line if you want to show figure 2

#  Males make up a majority of participants
graph = go.Figure(go.Pie(labels = df_tr['gender'].value_counts().keys().to_list(),values = df_tr['gender'].value_counts().to_list(),hole = 0.5))
#graph.update_layout(title_text='Gender distribution', title_x=0.5) #uncomment this line if you want to show figure 3

#  The percentage of people looking for a new job is almost the same for all genders. 
#  Gender has no effect on changing the proportion of job seekers
graph = px.histogram(data_frame = df_tr, x = 'gender', color = 'target', title = "Relationship between target and gender")
#graph.show() #uncomment this line if you want to show figure 4

#  As education level increases, the percentage of females also increases 
graph=px.histogram(data_frame = df_tr, x = 'education_level', color = 'gender', title = "Gender education level")
#graph.show() #uncomment this line if you want to show figure 5

#  The higher education level means the greater amount of candidates with relevant experience
graph=px.histogram(data_frame = df_tr, x = 'education_level', color = 'relevent_experience', title = "Relationship between education level and relevent experience")
#graph.show() #uncomment this line if you want to show figure 6

#  STEM is the most common discipline for data scientists
graph = go.Figure(go.Pie(labels = df_tr['major_discipline'].value_counts().keys().to_list(),values = df_tr['major_discipline'].value_counts().to_list(),hole = 0.5))
#graph.update_layout(title_text='Major discipline', title_x=0.5) #uncomment this line if you want to show figure 7 

#  Most of the candidates have more than 20 years of experience
graph = px.histogram(data_frame = df_tr, y = 'experience', title = "Experience")
#graph.show() #uncomment this line if you want to show figure 8

#  The less work experience, the probability that the candidate is searching 
#  for a new job is higher
graph = px.histogram(data_frame = df_tr, x = 'experience', color = 'target', title = "Relationship between experience and target")
#graph.show() #uncomment this line if you want to show figure 9

#  The proportion of candidates looking for a new job is consistent across all company sizes
graph = px.histogram(data_frame = df_tr, x = 'company_size', color = 'target', title = "Relationship between company size and target")
#graph.show() #uncomment this line if you want to show figure 10

#  Strong relationship between city and city development index
graph = px.histogram(data_frame = df_tr, x = 'city', color = 'target', title = "Relationship between city and target")
#graph.show() #uncomment this line if you want to show figure 11
graph = px.histogram(data_frame = df_tr, x = 'city_development_index', color = 'target', title = "Relationship between city development index and target")
#graph.show() #uncomment this line if you want to show figure 12

#  The percentage of people looking for a new job is almost the same for all education levels
graph = px.histogram(data_frame = df_tr, x = 'education_level', color = 'target', title = "Relationship between education level and target")
#graph.show() #uncomment this line if you want to show figure 13

#  Similar distribution for both types of candidates 
graph = px.histogram(data_frame = df_tr, x = 'last_new_job', color = 'target', title = "Relationship between last new job and target")
#graph.show() #uncomment this line if you want to show figure 14

#  The distribution of job seekers is equal  
graph = px.histogram(data_frame = df_tr, x = 'relevent_experience', color = 'target', title = "Relationship between relevant experience and target")
#graph.show() #uncomment this line if you want to show figure 15

#  The distribution of job seekers is equal  
graph = px.histogram(data_frame = df_tr, x = 'major_discipline', color = 'target', title = "Relationship between major discipline and target")
#graph.show() #uncomment this line if you want to show figure 16

#  Strong connection between variables 
graph = px.histogram(data_frame = df_tr, x = 'enrolled_university', color = 'target', title = "Relationship between enrolled university and target")
#graph.show() #uncomment this line if you want to show figure 17

#  Strong connection between variables 
graph = px.histogram(data_frame = df_tr, x = 'company_type', color = 'target', title = "Relationship between company type and target")
#graph.show() #uncomment this line if you want to show figure 18

#  Correlation Heatmap
fig, ax = plt.subplots(figsize=(7,7))
sns.heatmap(df_tr.corr(),ax=ax,annot=True)
plt.title('Correlation Heatmap', weight='bold',fontsize=10)
plt.close() #comment this line if you want to show figure 19


#  Model training
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score,StratifiedKFold
import pickle


train_data = df_tr.copy()
test_df = df_te.copy()

#  deleting features from train and test dataset that do not affect prediction
#  deleting city because in EDA we have seen that city and city_develompent_index have a very strong relationship
#  enrollee_id does not affect prediction
del train_data['enrollee_id']
del train_data['city']

enrolled_id_test_df = test_df['enrollee_id']
del test_df['enrollee_id']
del test_df['city']

#  converting categorical variables into dummy/indicator variables
train_data = pd.get_dummies(train_data)
test_df = pd.get_dummies(test_df)

X = train_data.drop('target', axis=1)
y = train_data['target']

#  evaluating the performance of algorithm
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 44)

#first model: kNN - kNN makes highly accurate predictions and handles big datasets well
knn = KNeighborsClassifier(n_neighbors = 20)
knn.fit(X_train, y_train)
knn_prediction = knn.predict(X_test)

#  metrics: accuracy - a standard, commonly-used performance metric
#  recall - summarizes how well the positive class was predicted - good for
#  imbalanced classification because they focus on one class
#  precision - summarizes the fraction of examples assigned to the positive
#  class that belong to the positive class - good for imbalanced classification because they focus on one class
#  roc auc - can be optimistic under a severe class imbalance when the number 
#  of examples in the minority class is small
accuracy_knn = accuracy_score(y_test, knn_prediction)*100
recall_knn = recall_score(y_test, knn_prediction)*100
precision_knn = precision_score(y_test, knn_prediction)*100
roc_knn = roc_auc_score(y_test, knn_prediction)*100

#  second model: dtree - requires less effort for data preparation during
#  pre-processing, very intuitive and easy to explain, missing values in the data
#  also do not affect the process of building a decision tree to any considerable extent
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
dtree_prediction = dtree.predict(X_test)

#  metrics for the second model
accuracy_dtree = accuracy_score(y_test, dtree_prediction)*100
recall_dtree = recall_score(y_test, dtree_prediction)*100
precision_dtree = precision_score(y_test, dtree_prediction)*100
roc_dtree = roc_auc_score(y_test, dtree_prediction)*100

#  third model: rfc - works well with both categorical and continuous variables,
#  can automatically handle missing values, reduces overfitting problem in decision
#  trees and reduces the variance and therefore improves the accuracy
rfc = RandomForestClassifier(random_state = 0, n_estimators = 256, criterion = 'gini', max_features = 'auto', max_depth = 16)
rfc.fit(X_train, y_train)
rfc_prediction = rfc.predict(X_test)

#  metrics for the third model
accuracy_rfc = accuracy_score(y_test, rfc_prediction)*100
recall_rfc = recall_score(y_test, rfc_prediction)*100
precision_rfc = precision_score(y_test, rfc_prediction)*100
roc_rfc = roc_auc_score(y_test, rfc_prediction)*100

#  improvement of the third model - SMOTE (Synthetic Minority Oversampling Technique)
#  oversampling method to solve the imbalance problem
sm = SMOTE(random_state = 44)
X_res, y_res = sm.fit_resample(X_train, y_train)
steps = [('over', SMOTE()), ('model', RandomForestClassifier())]
SM_rf = Pipeline(steps = steps)
SM_rf.fit(X_res, y_res)
SM_rf_prediction = SM_rf.predict(X_test)

#  metrics for the improvement of the third model
accuracy_SM_rf = accuracy_score(y_test, SM_rf_prediction)*100
recall_SM_rf = recall_score(y_test, SM_rf_prediction)*100
precision_SM_rf = precision_score(y_test, SM_rf_prediction)*100
roc_SM_rf = roc_auc_score(y_test, SM_rf_prediction)*100

#  view models and metrics in tabular form
data = {'Accuracy (%)': ["{0:.2f}".format(accuracy_knn), "{0:.2f}".format(accuracy_dtree), "{0:.2f}".format(accuracy_rfc), "{0:.2f}".format(accuracy_SM_rf)], 'Recall (%)': ["{0:.2f}".format(recall_knn),"{0:.2f}".format(recall_dtree), "{0:.2f}".format(recall_rfc), "{0:.2f}".format(recall_SM_rf)], 'Precision (%)': ["{0:.2f}".format(precision_knn), "{0:.2f}".format(precision_dtree), "{0:.2f}".format(precision_rfc), "{0:.2f}".format(precision_SM_rf)], 'ROC AUC (%)': ["{0:.2f}".format(roc_knn), "{0:.2f}".format(roc_dtree), "{0:.2f}".format(roc_rfc), "{0:.2f}".format(roc_SM_rf)]}
df = pd.DataFrame(data, index = ['KNN', 'Decision Tree', 'Random Forest', 'SMOTE Random Forest'])
print(df)

#  Fine-tune model using GridSearchCV

#  define list of parameters
max_depth = [2, 8, 16]
n_estimators = [64, 128, 256]

#  enables searching over any sequence of parameter settings
param_grid = dict(max_depth = max_depth, n_estimators = n_estimators)
dfrst = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth)

#  the parameters of the estimator used to apply these methods are optimized by 
#  cross-validated grid-search over a parameter grid
grid = GridSearchCV(estimator = dfrst, param_grid = param_grid, cv = 5)
grid_results = grid.fit(X_train, y_train)

#  print best result
print("\nThe best result after tuning parameters is: ", "{0:.2f}".format(grid.best_score_*100), "%")
print("The best estimator is: ", grid.best_estimator_)


#  cross validation - StratifiedKFold works perfectly well for Imbalanced Data:
#  Each fold in stratified cross-validation will have a representation of data
#  of all classes in the same ratio as in the whole dataset 
stratifiedkf = StratifiedKFold(n_splits = 5)
score = cross_val_score(rfc, X, y, cv = stratifiedkf)

print("\nCross Validation Scores are {}".format(score))
print("Average Cross Validation score: {0:.2f}".format(score.mean()*100), "%")

#  Comment on stratified cross-validation results:
#  The cross validation scores are very similar and consistent. Stratified cv 
#  provides a more reliable accuracy evaluation.

#  The best model is SMOTE Random Forest. This model has the highest recall and
#  roc auc scores and high accuracy

#  saving the best model
filename = 'SMOTE_rfc.sav'
pickle.dump(dtree, open(filename, 'wb'))

#  exporting test data frame to csv
final_results = SM_rf.predict(test_df)
final_df = pd.DataFrame(data={'target': final_results }, index=enrolled_id_test_df)
final_df.to_csv('final_results.csv')

#  function which takes employee_id as input and returns predicted target
def get_predicted_target(employee_id):
  df = pd.read_csv('final_results.csv', index_col='enrollee_id')
  return int(df.loc[employee_id, :][0])

#  mini demo of function:
employee_ids = [32403, 9858, 31806, 21465, 12994, 10856]
for id in employee_ids:
  print('ID = ', id, ', target = ', get_predicted_target(id))
