import math
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go


#load dataset
df_tr = pd.read_csv('aug_train.csv')
df_te = pd.read_csv('aug_test.csv')

#information about given data
df_tr.head()
df_tr.shape

#checking null values in given train dataset
df_tr.isnull().sum()

#dropping rows with missing values would significantly reduce the dataset and underfitting could occur - too many missing values
#demonstrating a few other ways to handle with missing values: fill missing values with 0, fill missing values with mode, fill missing values with median.
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

#median is used if the data comprised of outliers or more frequent value, we can see that from boxplot
df_tr['experience'].fillna(df_tr['experience'].median(), inplace=True)

df_tr['last_new_job'] = df_tr['last_new_job'].fillna("NA")
lnj = {'>4': 5, 'never': 6}                                     #do this manually to explicitly tell the model how to group last_new_job
df_tr = df_tr.replace({"last_new_job": lnj})

df_tr['gender'] = df_tr['gender'].fillna("NA")                  #missing values fill with NA

#missing values fill with mode because mode is used when the data having more occurences of a particular value or more frequent value
#this method is used for the categorical variable
df_tr['company_type'].fillna(df_tr['company_type'].mode()[0], inplace=True)         
df_tr['major_discipline'].fillna(df_tr['major_discipline'].mode()[0], inplace=True)       
df_tr['education_level'].fillna(df_tr['education_level'].mode()[0], inplace=True)
df_tr['enrolled_university'].fillna(df_tr['enrolled_university'].mode()[0], inplace=True)

#the same procedure for test dataset
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




#EDA - Exploratory Data Analysis 

#Distribution of target - 75% candidates are not searching for new job
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

#The top 4 cities represent more than half of the candidates
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

#Males make up a majority of participants
graph = go.Figure(go.Pie(labels = df_tr['gender'].value_counts().keys().to_list(),values = df_tr['gender'].value_counts().to_list(),hole = 0.5))
#graph.update_layout(title_text='Gender distribution', title_x=0.5) #uncomment this line if you want to show figure 3

#The percentage of people looking for a new job is almost the same for all genders. Gender has no effect on changing the proportion of job seekers
graph = px.histogram(data_frame = df_tr, x = 'gender', color = 'target', title = "Relationship between target and gender")
#graph.show() #uncomment this line if you want to show figure 4

#As education level increases, the percentage of females also increases 
graph=px.histogram(data_frame = df_tr, x = 'education_level', color = 'gender', title = "Gender education level")
#graph.show() #uncomment this line if you want to show figure 5

#The higher education level means the greater amount of candidates with relevant experience
graph=px.histogram(data_frame = df_tr, x = 'education_level', color = 'relevent_experience', title = "Relationship between education level and relevent experience")
#graph.show() #uncomment this line if you want to show figure 6

#STEM is the most common discipline for data scientists
graph = go.Figure(go.Pie(labels = df_tr['major_discipline'].value_counts().keys().to_list(),values = df_tr['major_discipline'].value_counts().to_list(),hole = 0.5))
#graph.update_layout(title_text='Major discipline', title_x=0.5) #uncomment this line if you want to show figure 7 

#Most of the candidates have more than 20 years of experience
graph = px.histogram(data_frame = df_tr, y = 'experience', title = "Experience")
#graph.show() #uncomment this line if you want to show figure 8

#The less work experience, the probability that the candidate is searching for a new job is higher
graph = px.histogram(data_frame = df_tr, x = 'experience', color = 'target', title = "Relationship between experience and target")
#graph.show() #uncomment this line if you want to show figure 9

#The proportion of candidates looking for a new job is consistent across all company sizes
graph = px.histogram(data_frame = df_tr, x = 'company_size', color = 'target', title = "Relationship between company size and target")
#graph.show() #uncomment this line if you want to show figure 10

#Strong relationship between city and city development index
graph = px.histogram(data_frame = df_tr, x = 'city', color = 'target', title = "Relationship between city and target")
#graph.show() #uncomment this line if you want to show figure 11
graph = px.histogram(data_frame = df_tr, x = 'city_development_index', color = 'target', title = "Relationship between city development index and target")
#graph.show() #uncomment this line if you want to show figure 12

#The percentage of people looking for a new job is almost the same for all education levels
graph = px.histogram(data_frame = df_tr, x = 'education_level', color = 'target', title = "Relationship between education level and target")
#graph.show() #uncomment this line if you want to show figure 13

#Similar distribution for both types of candidates 
graph = px.histogram(data_frame = df_tr, x = 'last_new_job', color = 'target', title = "Relationship between last new job and target")
#graph.show() #uncomment this line if you want to show figure 14

#The distribution of job seekers is equal  
graph = px.histogram(data_frame = df_tr, x = 'relevent_experience', color = 'target', title = "Relationship between relevant experience and target")
#graph.show() #uncomment this line if you want to show figure 15

#The distribution of job seekers is equal  
graph = px.histogram(data_frame = df_tr, x = 'major_discipline', color = 'target', title = "Relationship between major discipline and target")
#graph.show() #uncomment this line if you want to show figure 16

#Strong connection between variables 
graph = px.histogram(data_frame = df_tr, x = 'enrolled_university', color = 'target', title = "Relationship between enrolled university and target")
#graph.show() #uncomment this line if you want to show figure 17

#Strong connection between variables 
graph = px.histogram(data_frame = df_tr, x = 'company_type', color = 'target', title = "Relationship between company type and target")
#graph.show() #uncomment this line if you want to show figure 18

#Correlation Heatmap
fig, ax = plt.subplots(figsize=(7,7))
sns.heatmap(df_tr.corr(),ax=ax,annot=True)
plt.title('Correlation Heatmap', weight='bold',fontsize=10)
plt.close() #comment this line if you want to show figure 19