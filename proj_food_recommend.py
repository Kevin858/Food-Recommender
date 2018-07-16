import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm
import statsmodels.api as sma
from pandas.core import datetools
from sklearn.metrics import f1_score
# import healthcareai
# import healthcareai.trained_models.trained_supervised_model as tsm_plots
# import healthcareai.common.database_connections as hcai_db
#
# from healthcareai import SupervisedModelTrainer

from sklearn.feature_selection import SelectKBest, f_regression
import itertools

def input_transformer_recommend(gender, age, df): # replace first row
    # initialized 0
    col = df.columns.values.tolist()
    zeros = np.zeros((1,len(col)))
    input_data = pd.DataFrame(data=zeros, columns=col)

    if gender == "Male":
        input_data.gender_Male = 1
        input_data.gender_Female = 0
    else:
        input_data.gender_Male = 0
        input_data.gender_Female = 1
    # age
    input_data.ageInWeek = age * 52
    # symptom

    return input_data

def input_transformer_food(gender, age, symptom, df):  # replace first row
    # initialized 0
    col = df.columns.values.tolist()
    col.pop(0)  # remove fdaName, the 1st element
    zeros = np.zeros((1, len(col)))
    input_data = pd.DataFrame(data=zeros, columns=col)

    if gender == "Male":
        # input_data.iloc[1, :] = 0
        input_data['gender_Male'] = [1]
        input_data['gender_Female'] = [0]
        # df.genderFemale.iloc[:,1] = 0
    else:
        input_data["gender_Male"] = [0]
        input_data["gender_Female"] = [1]

    # age
    input_data.ageInWeek = age * 52

    # symptom
    symptom_string = symptom
    input_data.filter(regex=symptom_string, axis=1)
    input_data[symptom_string] = 1

    return input_data

def merge_dummy_column(df, column_in):

    str_upper = column_in
    str_lower = column_in.lower()
    str_lower_col = str_lower #+ "_col"

    disablity_cols = [col for col in df.columns if str_upper in col]
    df[str_lower_col] = df[disablity_cols].max(axis=1)
    return df

# # main function

mode = 1
# food search
# mode 1: (age,gender,symptom) => food

# recommendation
# mode 2: (age,gender) => food
# mode 3: (age,gender) => symptom

df = pd.read_csv("CAERS_ASCII_2004_2017Q2.csv")#("food.csv") # CAERS_ASCII_2004_2017Q2.csv
headers = df.dtypes.index
df.columns = df.columns.str.replace (' #', 'ID')

#Dropping all duplicate rows but keeping the first row

df = df.dropna(axis=0, how='any')
df = df[df.CI_Gender != "Not Available"]
df = df[df.CI_Gender != "Not Reported"]
df = df[df.CI_Gender != "Unknown"]
df = df.drop_duplicates(subset='RA_ReportID')

df = df.rename(columns={"AEC_One Row Outcomes":"Symptom","PRI_FDA Industry Code":"fdaCode", 'PRI_FDA Industry Name': 'fdaName'})

df.columns = df.columns.str.strip().str.replace(' ', '')
df.columns = df.columns.str.replace('/', '')
df.columns = df.columns.str.replace('.', '')
df.columns = df.columns.str.replace('-', '_')
df.columns = df.columns.str.replace('_', '')
df.columns = df.columns.str.replace('(', '')
df.columns = df.columns.str.replace(')', '')
df.columns = df.columns.str.replace('.', '')

## Convert df['caers created date'] and df ['eventStartDate'] to datetime format
df['eventCreateDate'] = pd.to_datetime(df['RACAERSCreatedDate'])
df['eventStartDate'] = pd.to_datetime(df['AECEventStartDate'])

# Compute the time Difference Between the eventStartDate and the RACAERSCreatedDate
df['timeDiff'] = (df['eventCreateDate'] - df['eventStartDate']).dt.days

# Group the Time difference by different periods
time_list = (0,180,366,1098,1830,3660)
time_cat = ['less than 6 Months','less than 1 year','1 to 3 years','3 to 5 years','5 to 10 Years']
df['timeGroup'] = pd.cut(df['timeDiff'],time_list,labels= time_cat)
df['timeGroup'] = df['timeGroup'].astype('str')
df['timeGroup'] = df['timeGroup'].replace('nan','Unknown')

#Let Split the data into two dataset...one that has age unit and one that don't('Not Available' as value)
df = df[df['CIAgeUnit']!= 'Not Available']

# Convert the age unit to week, which means year = 52 week, month= 4 weeks, day = 1/7 ~= 0.143 week, decade = 520 weeks
age_change = {'Year(s)':52,'Month(s)':4,'Week(s)':1,'Day(s)':0.143,'Decade(s)':520}
df['ageByWeek'] = df['CIAgeUnit'].map(age_change)

#Multiply the age at adverse by ageByWeek to have Age in a single unit which is weeks
df['ageInWeek'] = df['CIAgeatAdverseEvent'] * df['ageByWeek']


# CALCULATE
df = df.rename(columns={"symptom":"Symptom"})

df['symptom1'] = df.Symptom.str.split(', ').str.get(0)
df['symptom2'] = df.Symptom.str.split(', ').str.get(1)
df['symptom3'] = df.Symptom.str.split(', ').str.get(2)
df['symptom4'] = df.Symptom.str.split(', ').str.get(3)
df['symptom5'] = df.Symptom.str.split(', ').str.get(4)
df['symptom6'] = df.Symptom.str.split(', ').str.get(5) # redundant

# split to the maximum

headers = df.dtypes.index
train = df

# feature set :
# gender : CIGender (category)
# age : timeGroup (numerical)

# estimate value 1:
# food class: fdaCode (category)

# estimate value 2:
# symptom: symptom 1 to 6 (category)


obj_df = train.iloc[:,17:] # partial select, symptom1 to symptom6
x_gender = train.loc[:,"CIGender"]
x_timeGroup = train.loc[:,"ageInWeek"]#[:,"timeGroup"]
cols_to_transform_gender = ['CIGender']
df_with_dummies_gender = pd.get_dummies(x_gender, columns = cols_to_transform_gender , prefix='gender')

cols_to_transform_time = ['ageInWeek']
df_with_dummies_time = pd.get_dummies(x_timeGroup, columns = cols_to_transform_time , prefix='time')

cols_to_transform = ['symptom1', 'symptom2','symptom3', 'symptom4','symptom5', 'symptom6']

df_with_dummies = pd.get_dummies(obj_df, columns = cols_to_transform , prefix='symptom')

df_with_dummies.columns = df_with_dummies.columns.str.replace('_', '') # extra one  blank , solved
df_with_dummies.columns = df_with_dummies.columns.str.replace('symptom', '') # extra one  blank , solved

df = df_with_dummies
col_list = list(df.columns.values)
col_name = set(df.columns.astype(str))

#print(col_name)

df_row, df_col = df.shape # restore original shape of dataframe

# merge dummy columns into non-repeated column
for col in col_name:
    df = merge_dummy_column(df, col)

df_symptom = df.iloc[:,df_col+1:]


df_a = train.loc[:,"fdaName"] # fda code or fda name
df_b = df_with_dummies_gender #df.filter(regex=("CIGender*"))
df_c = train.loc[:,"ageInWeek"]#df_with_dummies_time#train.loc[:,"ageByWeek"]#df_with_dummies_time #df.filter(regex=("timeGroup*")), train.loc[:,"ageByWeek"]
#df_d = df_symptom #df.filter(regex=("symptom*"))
#df_d = obj_df.loc[:,"symptom1"]
df_d = df_symptom  # df.filter(regex=("symptom*"))
df_a = df_a.astype('str')

#
# # SKLEARN
# df_x = pd.concat([df_b,df_c],axis=1) # combine two Series into Dataframe
# #df_y = df_a # symptom vs food name, df_d vs df_a
# df_y = df_d

if mode == 1: # y = food
    #df_a = train.loc[:, "fdaName"]  # fda code or fda name
    #df_a = df_a.astype('str')

    # df = pd.concat([df_a,df_b],axis=1) # combine two Series into Dataframe
    #df = pd.concat([df_a, df_b, df_c, df_d], axis=1)  # combine two Series into Dataframe

    # SKLEARN
    df_x = pd.concat([df_b, df_c, df_d], axis=1)  # combine two Series into Dataframe
    df_y = train.loc[:, "fdaName"]  # fda code or fda name

if mode == 2:  # y = food
    df_x = pd.concat([df_b, df_c], axis=1)  # combine two Series into Dataframe
    df_y = train.loc[:,"fdaName"]

if mode == 3: # y = symptom
    df_x = pd.concat([df_b, df_c], axis=1)  # combine two Series into Dataframe
    df_y = obj_df.loc[:, "symptom1"]



df_x.columns = df_x.columns.str.replace(' ', '')
df_x.columns = df_x.columns.str.replace('/', '')
df_x.columns = df_x.columns.str.replace('.', '')
df_x.columns = df_x.columns.str.replace('-', '')
df_x.columns = df_x.columns.str.replace('_', '')
df_x.columns = df_x.columns.str.replace('(', '')
df_x.columns = df_x.columns.str.replace(')', '')


# k nearest neightbor (KNN)
from sklearn.cross_validation import train_test_split

# create design matrix X and target vector y
X = df_x #np.array(df.ix[:, 0:4]) 	# end index is exclusive
y = df_y #np.array(df['class']) 	# another way of indexing a pandas df

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# loading library
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score


# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=64, weights="distance")

# fitting the model
knn.fit(X_train, y_train)

# predict the response
pred = knn.predict(X_test)

#f1 = f1_score(y_test, pred, average='micro')
#print(pred.iloc(1,1))


# evaluate accuracy
print(accuracy_score(y_test, pred))
print(classification_report(y_test, pred)) #, target_names=target_names))
#print(f1)



# test our input
gender = "Male"
age = 12
symptom = "death"


# search food
if mode == 1:
    df_input = input_transformer_food(gender, age, symptom, X_test)  # replace first row

# Food , RECOMMENDATION
else:
    df_input = input_transformer_recommend(gender, age, X_test) # replace first row

X_test = X_test.iloc[1:]
X_test.append(df_input)

# predict the response, print out symptom name
pred_user = knn.predict(X_test)
outputFood = pred_user[-1]

print(outputFood)


#print(df_input.shape)
#print(X_test.shape)

# predict the response, print out symptom name
#pred_user_food = knn.predict(X_test)
#print(pred_user_food[-1])




# FIND K parameter
# # creating list of K for KNN
# neighbors = [4,8,16,32,64]#list(range(1,50))
#
# # empty list that will hold cv scores
# cv_scores = []
#
# # perform 10-fold cross validation
# for k in neighbors:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
#     cv_scores.append(scores.mean())
#
# # changing to misclassification error
# MSE = [1 - x for x in cv_scores]
#
# print(MSE)
# # determining best k
# optimal_k = neighbors[MSE.index(min(MSE))]
# optimal_k = neighbors[MSE.index(min(MSE))]
#
# print("The optimal number of neighbors is %d" % optimal_k)
#
# #plot misclassification error vs k
# plt.plot(neighbors, MSE)
# plt.xlabel('Number of Neighbors K')
# plt.ylabel('Misclassification Error')
# plt.show()

