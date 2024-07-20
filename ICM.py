# importing important libraries
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from datetime import date
import streamlit as st
import sklearn
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the Excel file
ICM = 'C:/Users/Hp/OneDrive/Desktop/python/Copper_Set.xlsx'
df = pd.read_excel(ICM)

# convert the data type from object to numeric
df['quantity tons'] = pd.to_numeric(df['quantity tons'], errors='coerce')
df['item_date_1'] = pd.to_datetime(df['item_date'], format='%Y%m%d', errors='coerce').dt.date
df['delivery date_1'] = pd.to_datetime(df['delivery date'], format='%Y%m%d', errors='coerce').dt.date

# Some rubbish values are present in ‘Material_ref’ which starts with ‘00000’ value which should be converted into null
df['material_ref'] = df['material_ref'].apply(lambda x: np.nan if str(x).startswith('00000') else x)

# material ref have more than 55% are null values and id have all are unique values. so we have drop both columns.
df.drop(columns=['id','material_ref'], inplace=True)

# quantity and selling price values are not below 0. so we convert to null for below 0 values.
df['quantity tons'] = df['quantity tons'].apply(lambda x: np.nan if x <= 0 else x)
df['selling_price'] = df['selling_price'].apply(lambda x: np.nan if x <= 0 else x)

# Handling null values using median and mode
# median - middle value in dataset (asc/desc), mode - value that appears most frequently in dataset
# object datatype using mode
df['item_date'].fillna(df['item_date'].mode().iloc[0], inplace=True)
df['item_date_1'].fillna(df['item_date_1'].mode().iloc[0], inplace=True)
df['status'].fillna(df['status'].mode().iloc[0], inplace=True)
df['delivery date'].fillna(df['delivery date'].mode().iloc[0], inplace=True)
df['delivery date_1'].fillna(df['delivery date_1'].mode().iloc[0], inplace=True)

# #numerical datatype using median
df['quantity tons'].fillna(df['quantity tons'].median(), inplace=True)
df['customer'].fillna(df['customer'].median(), inplace=True)
df['country'].fillna(df['country'].median(), inplace=True)
df['application'].fillna(df['application'].median(), inplace=True)
df['thickness'].fillna(df['thickness'].median(), inplace=True)
df['selling_price'].fillna(df['selling_price'].median(), inplace=True)

# convert categorical data into numerical data - using map and ordinal encoder methods
df['status'] = df['status'].map({'Lost': 0, 'Won': 1, 'Draft': 2, 'To be approved': 3, 'Not lost for AM': 4,
                                 'Wonderful': 5, 'Revised': 6, 'Offered': 7, 'Offerable': 8})
df['item type'] = OrdinalEncoder().fit_transform(df[['item type']])

# #Copper_Sets
# # find outliers - box plot & skewed data - hist plot and violin plot

# def plot(df, column):
#     plt.figure(figsize=(20,5))
#     plt.subplot(1,3,1)
#     sns.boxplot(data=df, x=column)
#     plt.title(f'Box Plot for {column}')

#     plt.subplot(1,3,2)
#     sns.histplot(data=df, x=column, kde=True, bins=50)
#     plt.title(f'Distribution Plot for {column}')

#     plt.subplot(1,3,3)
#     sns.violinplot(data=df, x=column)
#     plt.title(f'Violin Plot for {column}')
#     plt.show()
# #for i in ['quantity tons', 'customer', 'country', 'item type', 'application', 'thickness', 'width', 'selling_price']:
#     #plot(df, i)
    
# # quantity tons, thickness and selling price data are skewd. so using the log transformation method to handle the skewness data
df1 = df.copy()
df1['quantity tons_log'] = np.log(df1['quantity tons'])
df1['thickness_log'] = np.log(df1['thickness'])
df1['selling_price_log'] = np.log(df1['selling_price'])

# # after log transformation the data are normally distributed and reduced the skewness. [hist plot and violin plot]
# #for i in ['quantity tons_log', 'thickness_log', 'width', 'selling_price_log']:
#     #plot(df1, i)

# # Outliers Handling - Interquartile Range (IQR) method
df2 = df1.copy()

# # Using IQR and clip() methods to handle the outliers and add a new column of dataframe
def outlier(df, column):
    iqr = df[column].quantile(0.75) - df[column].quantile(0.25)
    upper_threshold = df[column].quantile(0.75) + (1.5 * iqr)
    lower_threshold = df[column].quantile(0.25) - (1.5 * iqr)
    df[column] = df[column].clip(lower_threshold, upper_threshold)
    
# # (Ex: lower threshold = 5 and upper threshold = 20)
# # above upper threshold values (>20) are converted to upper threshold value (20) in features
# # below lower threshold values (<5)  are converted to lower threshold value (5)  in features
outlier(df2, 'quantity tons_log')
outlier(df2, 'thickness_log')
outlier(df2, 'selling_price_log')
outlier(df2, 'width')

# # transform the outliers to within range using IQR and clip() methods - box plot
# # for i in ['quantity tons_log', 'thickness_log', 'width', 'selling_price_log']:
# #     plot(df2, i)
    
# # after adding the new column of 'quantity tons_log', 'thickness_log', 'selling_price_log', drop the existing columns
df3 = df2.drop(columns=['quantity tons', 'thickness', 'selling_price'])

# # Need to verify any columns are highly correlated using Heatmap. If any columns correalaion value >= 0.7 (absolute value), drop the columns.
# # col = ['quantity tons_log','customer','country','status','application','width','product_ref','thickness_log','selling_price_log']
# # df_heatmap = df2[col].corr()
# # sns.heatmap(df_heatmap, annot=True)

# # Wrong Delivery Date Handling
df4 = df3.copy()

# find the difference between item and delivery date and add the new column of dataframe
df4['Date_difference'] = (pd.to_datetime(df4['delivery date_1']) - pd.to_datetime(df4['item_date_1'])).dt.days

# convert the data type using pandas
df4['item_date_1'] = pd.to_datetime(df4['item_date_1'])

# split the day, month, and year from 'item_date_1' column and add dataframe (This data also help us to prediction)
df4['item_date_day'] = df4['item_date_1'].dt.day
df4['item_date_month'] = df4['item_date_1'].dt.month
df4['item_date_year'] = df4['item_date_1'].dt.year

# split the non-negative value of 'Date_difference' column in separate dataframe
df_f1 = df4[df4['Date_difference']>=0]

# after split, the index values are unordered. so need to reset the index to ascending order from 0
df_f1 = df_f1.reset_index(drop=True)

# split the negative value of 'Date_difference' column in another dataframe
df_f2 = df4[df4['Date_difference']<0]

# after split, the index values are unordered. so need to reset the index to ascending order from 0
df_f2 = df_f2.reset_index(drop=True)

# # find best algorithm for prediction based on R2, mean absolute error, mean squared error and root mean squared error values
# # def machine_learning_delivery_date(df, algorithm):

# #     x = df.drop(columns=['item_date_1','delivery date_1','Date_difference'], axis=1)
# #     y = df['Date_difference']
# #     x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

# #     model = algorithm().fit(x_train, y_train)
# #     y_pred = model.predict(x_test)

# #     mse = mean_squared_error(y_test, y_pred)
# #     rmse = np.sqrt(mse)
# #     r2 = r2_score(y_test, y_pred)
# #     mae = mean_absolute_error(y_test, y_pred)

# #     metrics = {'Algorithm': str(algorithm).split("'")[1].split(".")[-1],
# #                'R2': r2,
# #                'Mean Absolute Error': mae,
# #                'Mean Squared Error': mse,
# #                'Root Mean Squared Error': rmse}

# #     return metrics
# # machine_learning_delivery_date(df_f1, DecisionTreeRegressor)
# # machine_learning_delivery_date(df_f1, ExtraTreesRegressor)
# # machine_learning_delivery_date(df_f1, RandomForestRegressor)
# # machine_learning_delivery_date(df_f1, XGBRegressor)

# # train the model by using Random Forest Regression algorithm to predict 'Date difference'
# # 'item_date_1','delivery date_1' - this columns are non-numerical and cannot passed, so skip the columns in model training and prediction.
# def ml_date_difference():

#     # train the model by using correct delivery date (df_f1) dataframe
#     x = df_f1.drop(columns=['item_date_1','delivery date_1','Date_difference'], axis=1)
#     y = df_f1['Date_difference']
#     x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

#     model = RandomForestRegressor().fit(x_train, y_train)

#     # predict the 'Date_difference' of df_f2 columns using model
#     y_pred_list = []

#     for index, row in df_f2.iterrows():
#         input_data = row.drop(['item_date_1','delivery date_1','Date_difference'])
#         y_pred = model.predict([input_data])
#         y_pred_list.append(y_pred[0])

#     return y_pred_list

# Machine learning model predict the date difference of (df_f2) datafame
# date_difference = ml_date_difference()

# convert float values into integer using list comprehension method
# date_difference1 = [int(round(i,0)) for i in date_difference]

# add 'Date_difference' column in the dataframe
# df_f2['Date_difference'] = pd.DataFrame(date_difference1)

# calculate delivery date (item_date + Date_difference = delivery_date)
def find_delivery_date(item_date, date_difference):
    result_date = item_date + timedelta(days=date_difference)
    delivery_date = result_date.strftime("%Y-%m-%d")
    return delivery_date

# find out the delivery date and add to dataframe
df_f2['item_date_1'] = pd.to_datetime(df_f2['item_date_1'])
df_f2['delivery date_1'] = df_f2.apply(lambda x: find_delivery_date(x['item_date_1'], x['Date_difference']), axis=1)

# # Finally concatinate the both dataframe into single dataframe
df_final = pd.concat([df_f1,df_f2], axis=0, ignore_index=True)

# split the day, month, and year from 'delivery_date_1' column and add dataframe (This data also help us to prediction)
df_final['delivery date_1'] = pd.to_datetime(df_final['delivery date_1'])
df_final['delivery_date_day'] = df_final['delivery date_1'].dt.day
df_final['delivery_date_month'] = df_final['delivery date_1'].dt.month
df_final['delivery_date_year'] = df_final['delivery date_1'].dt.year

# finally drop the item_date, delivery_date and date_difference columns
df_final.drop(columns=['item_date','delivery date','item_date_1','delivery date_1','Date_difference'], inplace=True)

# #Classification Method - Predict Status
df_c = df_final.copy()

# # filter the status column values only 1 & 0 rows in a new dataframe ['Won':1 & 'Lost':0]
df_c = df_c[(df_c.status == 1) | (df_c.status == 0)]

# # check no of rows (records) of each 1 and 0 in dataframe
df_c['status'].value_counts()

# # in status feature, the 'Won' and 'Lost' value difference is very high. So we need to oversampling to reduce the difference

x = df_c.drop('status', axis=1)
y = df_c['status']

x_new, y_new = SMOTETomek().fit_resample(x,y)
# #x.shape, y.shape, x_new.shape, y_new.shape

# # check the accuracy of training and testing using metrics
# # algorithm.__name__  - it return the algorithm name

# # def machine_learning_classification(x_new,y_new, algorithm):

# #     x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size=0.2, random_state=42)
# #     model = algorithm().fit(x_train, y_train)

# #     y_pred_train = model.predict(x_train)
# #     y_pred_test = model.predict(x_test)

# #     accuracy_train = metrics.accuracy_score(y_train, y_pred_train)
# #     accuracy_test = metrics.accuracy_score(y_test, y_pred_test)

# #     # algo = str(algorithm).split("'")[1].split(".")[-1]
# #     accuracy_metrics = {'algorithm'    : algorithm.__name__,
# #                         'accuracy_train': accuracy_train,
# #                         'accuracy_test' : accuracy_test}

# #     return accuracy_metrics
# # machine_learning_classification(x_new, y_new, DecisionTreeClassifier)
# # machine_learning_classification(x_new, y_new, ExtraTreesClassifier)
# # machine_learning_classification(x_new, y_new, RandomForestClassifier)
# # machine_learning_classification(x_new, y_new, XGBClassifier)

# # x_train, x_test, y_train, y_test= train_test_split(x_new, y_new, test_size=0.8, random_state=42)

# # param_grid = {
# #     'max_depth': [5, 10],
# #     'min_samples_split': [5, 10],
# #     'min_samples_leaf': [2, 4],
# #     'max_features': ['sqrt', 'log2']
# # }
# # grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=3, n_jobs=-1)
# # grid_search.fit(x_train, y_train)

# # # Using RandomizedSearchCV for quicker search
# # random_search = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=param_grid, n_iter=10, cv=3, n_jobs=-1, random_state=42)
# # random_search.fit(x_train, y_train)

# # Displaying best parameters and score
# # f"Best parameters found: {random_search.best_params_}"
# # f"Best cross-validation score: {random_search.best_score_}"

# # Training the model with the best parameters
# # best_model = random_search.best_estimator_
# # best_model.fit(x_train, y_train)

# # # Evaluating on the test set
# # y_pred_test = best_model.predict(x_test)
# # accuracy_test = metrics.accuracy_score(y_test, y_pred_test)
# #f"Test set accuracy: {accuracy_test}"

# # evaluate all the parameter combinations and return the best parameters based on score
# # grid_search.best_params_
# # grid_search.best_score_

# # passing the parameters in the random forest algorithm and check the accuracy for training and testing

# # x_train, x_test, y_train, y_test = train_test_split(x_new,y_new,test_size=0.2,random_state=42)

# # model = RandomForestClassifier(max_depth=20, max_features='sqrt', min_samples_leaf=1, min_samples_split=2).fit(x_train, y_train)
# # y_pred_train = model.predict(x_train)
# # y_pred_test = model.predict(x_test)
# # accuracy_train = metrics.accuracy_score(y_train, y_pred_train)
# # accuracy_test = metrics.accuracy_score(y_test, y_pred_test)
# #accuracy_train, accuracy_test

# # predict the status and check the accuracy using metrics

x_train, x_test, y_train, y_test = train_test_split(x_new,y_new,test_size=0.2,random_state=42)

model = RandomForestClassifier(max_depth=20, max_features='sqrt', min_samples_leaf=1, min_samples_split=2).fit(x_train, y_train)
# y_pred = model.predict(x_test)

# # confusion_matrix(y_true=y_test, y_pred=y_pred)
# # classification_report(y_true=y_test, y_pred=y_pred)

# # Receiver Operating Characteristic (ROC) Curve and Area Under the Curve (AUC)

# # FP,TP,threshold = roc_curve(y_true=y_test, y_score=y_pred)
# # auc_curve = auc(x=FP, y=TP)

# # plt.plot(FP, TP, label=f"ROC Curve (area={round(auc_curve, 2)}) ")
# # plt.plot([0, 1], [0, 1], 'k--')
# # plt.xlim([0.0, 1.0])
# # plt.ylim([0.0, 1.10])
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.legend(loc='lower right')
# #plt.show()

# will pass the user data manually to check the prediction of status ar our model

user_data = np.array([[30153963, 30, 6, 28, 952, 628377, 5.9, -0.96, 6.46, 1,4,2021,1,1,2021]])
y_p = model.predict(user_data)
if y_p[0] == 1:
    print('Won')
else:
    print('Lose')
user_data = np.array([[30223403, 78, 5, 10, 1500, 1668701718, 2.2, 0, 7.13, 1,4,2021,1,7,2021]])
y_p = model.predict(user_data)
if y_p[0] == 1:
    print('Won')
else:
    print('Lose')
    
# save the classification model by using pickle
with open('classification_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
# load pickle model to predict the status (like Trained brain).

with open('C:/Users/Hp/OneDrive/Desktop/python/classification_model.pkl', 'rb') as f:
    model = pickle.load(f)

user_data = np.array([[30223403, 78, 5, 10, 1500, 1668701718, 2.2, 0, 7.13, 1,4,2021,1,7,2021]])
y_p = model.predict(user_data)
if y_p[0] == 1:
    print('Won')
else:
    print('Lose')

# # Regression Method - Predict Selling Price
# # check the train and test accuracy using R2 (R-squared ---> coefficient of determination) to predict selling price

# # def machine_learning_regression(df, algorithm):

# #     x = df.drop(columns=['selling_price_log'], axis=1)
# #     y = df['selling_price_log']
# #     x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# #     model = algorithm().fit(x_train, y_train)
# #     y_pred_train = model.predict(x_train)
# #     y_pred_test = model.predict(x_test)
# #     r2_train = r2_score(y_train, y_pred_train)
# #     r2_test = r2_score(y_test, y_pred_test)

# #     # algo = str(algorithm).split("'")[1].split(".")[-1]
# #     accuracy_metrics = {'algorithm': algorithm.__name__,
# #                         'R2_train' : r2_train,
# #                         'R2_test'  : r2_test}

# #     return accuracy_metrics
# # machine_learning_regression(df_final, DecisionTreeRegressor)
# # machine_learning_regression(df_final, ExtraTreesRegressor)
# # machine_learning_regression(df_final, RandomForestRegressor)
# # machine_learning_regression(df_final, XGBRegressor)

# # Preparing the data
# x = df_final.drop(columns=['selling_price_log'], axis=1)
# y = df_final['selling_price_log']
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# # Defining the parameter grid
# # param_grid_r = {
# #     'max_depth': [5, 10],
# #     'min_samples_split': [5, 10],
# #     'min_samples_leaf': [2, 4],
# #     'max_features': ['sqrt', 'log2']
# # }

# # Using RandomizedSearchCV for quicker search
# #random_search = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=param_grid_r, n_iter=10, cv=3, n_jobs=-1, random_state=42)
# #random_search.fit(x_train, y_train)

# # Displaying best parameters and score
# # f"Best parameters found: {random_search.best_params_}"
# # f"Best cross-validation score: {random_search.best_score_}"

# # Training the model with the best parameters
# # best_model = random_search.best_estimator_
# # best_model.fit(x_train, y_train)

# # Evaluating on the test set
# #y_pred_test = best_model.predict(x_test)

# # Calculating R2 score and other regression metrics
# # r2_test = r2_score(y_test, y_pred_test)
# # mse_test = mean_squared_error(y_test, y_pred_test)
# # mae_test = mean_absolute_error(y_test, y_pred_test)

# # f"Test set R2 score: {r2_test}"
# # f"Test set Mean Squared Error: {mse_test}"
# # f"Test set Mean Absolute Error: {mae_test}"

# # best parameters for hypertuning the random forest algorithm for better accuracy in unseen data
# # random_search.best_params_
# # random_search.best_score_

# # pass the parameters and check the accuracy for both training and testing & overfitting
# x = df_final.drop(columns=['selling_price_log'], axis=1)
# y = df_final['selling_price_log']
# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# model = RandomForestRegressor(max_depth=20, max_features=None, min_samples_leaf=1, min_samples_split=2).fit(x_train, y_train)
# # y_pred_train = model.predict(x_train)
# # y_pred_test = model.predict(x_test)

# # r2_train = r2_score(y_train, y_pred_train)
# # r2_test = r2_score(y_test, y_pred_test)
# #r2_train, r2_test

# # predict the selling price with hypertuning parameters and calculate the accuracy using metrics
# x = df_final.drop(columns=['selling_price_log'], axis=1)
# y = df_final['selling_price_log']
# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# model = RandomForestRegressor(max_depth=20, max_features=None, min_samples_leaf=1, min_samples_split=2).fit(x_train, y_train)
# # y_pred = model.predict(x_test)

# # mse = mean_squared_error(y_test, y_pred)
# # rmse = np.sqrt(mse)
# # r2 = r2_score(y_test, y_pred)
# # mae = mean_absolute_error(y_test, y_pred)

# # metrics_r = {'R2': r2,
# #            'Mean Absolute Error': mae,
# #            'Mean Squared Error': mse,
# #            'Root Mean Squared Error': rmse}

# manually passed the user input and predict the selling price
user_data = np.array([[30202938,25,1,5,41,1210,1668701718,6.6,-0.2,1,4,2021,1,4,2021]])
y_pred = model.predict(user_data)
#y_pred[0]

# using Inverse Log Transformation to convert the value to original scale of the data (exp)
# np.exp(y_pred[0])

# save the regression model by using pickle
with open('regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
# load the pickle model to predict selling price
with open('C:/Users/Hp/OneDrive/Desktop/python/regression_model.pkl', 'rb') as f:
    model = pickle.load(f)
y_pred = model.predict(np.array([[30202938,25,1,5,41,1210,1668701718,6.6,-0.2,1,4,2021,1,4,2021]]))
# np.exp(y_pred[0])    

# #Streamlit Part

# # Streamlit page custom design

# def streamlit_config():

#     # page configuration
#     st.set_page_config(page_title='Industrial Copper Modeling')

#     # page header transparent color
#     page_background_color = """
#     <style>

#     [data-testid="stHeader"] 
#     {
#     background: rgba(0,0,0,0);
#     }

#     </style>
#     """
#     st.markdown(page_background_color, unsafe_allow_html=True)

# # title and position
# st.markdown(f'<h1 style="text-align: center;">Industrial Copper Modeling</h1>',
#             unsafe_allow_html=True)

# # custom style for submit button - color and width

# def style_submit_button():

#     st.markdown("""
#                     <style>
#                     div.stButton > button:first-child {
#                                                         background-color: #367F89;
#                                                         color: white;
#                                                         width: 70%}
#                     </style>
#                 """, unsafe_allow_html=True)

# # custom style for prediction result text - color and position

# def style_prediction():

#     st.markdown(
#             """
#             <style>
#             .center-text {
#                 text-align: center;
#                 color: #20CA0C
#             }
#             </style>
#             """,
#             unsafe_allow_html=True
#         )

# # user input options

# class options:

#     country_values = [25.0, 26.0, 27.0, 28.0, 30.0, 32.0, 38.0, 39.0, 40.0, 77.0, 
#                     78.0, 79.0, 80.0, 84.0, 89.0, 107.0, 113.0]

#     status_values = ['Won', 'Lost', 'Draft', 'To be approved', 'Not lost for AM',
#                     'Wonderful', 'Revised', 'Offered', 'Offerable']
#     status_dict = {'Lost':0, 'Won':1, 'Draft':2, 'To be approved':3, 'Not lost for AM':4,
#                 'Wonderful':5, 'Revised':6, 'Offered':7, 'Offerable':8}

#     item_type_values = ['W', 'WI', 'S', 'PL', 'IPL', 'SLAWR', 'Others']
#     item_type_dict = {'W':5.0, 'WI':6.0, 'S':3.0, 'Others':1.0, 'PL':2.0, 'IPL':0.0, 'SLAWR':4.0}

#     application_values = [2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0, 
#                         27.0, 28.0, 29.0, 38.0, 39.0, 40.0, 41.0, 42.0, 56.0, 58.0, 
#                         59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0]

#     product_ref_values = [611728, 611733, 611993, 628112, 628117, 628377, 640400, 
#                         640405, 640665, 164141591, 164336407, 164337175, 929423819, 
#                         1282007633, 1332077137, 1665572032, 1665572374, 1665584320, 
#                         1665584642, 1665584662, 1668701376, 1668701698, 1668701718, 
#                         1668701725, 1670798778, 1671863738, 1671876026, 1690738206, 
#                         1690738219, 1693867550, 1693867563, 1721130331, 1722207579]

# # Get input data from users both regression and classification methods

# class prediction:

#     def regression():

#         # get input from users
#         with st.form('Regression'):

#             col1,col2,col3 = st.columns([0.5,0.1,0.5])

#             with col1:

#                 item_date = st.date_input(label='Item Date', min_value=date(2020,7,1), 
#                                         max_value=date(2021,5,31), value=date(2020,7,1))
                
#                 quantity_log = st.text_input(label='Quantity Tons (Min: 0.00001 & Max: 1000000000)')

#                 country = st.selectbox(label='Country', options=options.country_values)

#                 item_type = st.selectbox(label='Item Type', options=options.item_type_values)

#                 thickness_log = st.number_input(label='Thickness', min_value=0.1, max_value=2500000.0, value=1.0)

#                 product_ref = st.selectbox(label='Product Ref', options=options.product_ref_values)
            
#             with col3:

#                 delivery_date = st.date_input(label='Delivery Date', min_value=date(2020,8,1), 
#                                             max_value=date(2022,2,28), value=date(2020,8,1))
                
#                 customer = st.text_input(label='Customer ID (Min: 12458000 & Max: 2147484000)')

#                 status = st.selectbox(label='Status', options=options.status_values)

#                 application = st.selectbox(label='Application', options=options.application_values)

#                 width = st.number_input(label='Width', min_value=1.0, max_value=2990000.0, value=1.0)

#                 st.write('')
#                 st.write('')
#                 button = st.form_submit_button(label='SUBMIT')
#                 style_submit_button()

#         # give information to users
#         col1,col2 = st.columns([0.65,0.35])
#         with col2:
#             st.caption(body='*Min and Max values are reference only')

#         # user entered the all input values and click the button
#         if button:
            
#             # load the regression pickle model
#             with open(r'models\regression_model.pkl', 'rb') as f:
#                 model = pickle.load(f)
            
#             # make array for all user input values in required order for model prediction
#             user_data = np.array([[customer, 
#                                 country, 
#                                 options.status_dict[status], 
#                                 options.item_type_dict[item_type], 
#                                 application, 
#                                 width, 
#                                 product_ref, 
#                                 np.log(float(quantity_log)), 
#                                 np.log(float(thickness_log)),
#                                 item_date.day, item_date.month, item_date.year,
#                                 delivery_date.day, delivery_date.month, delivery_date.year]])
            
#             # model predict the selling price based on user input
#             y_pred = model.predict(user_data)

#             # inverse transformation for log transformation data
#             selling_price = np.exp(y_pred[0])

#             # round the value with 2 decimal point (Eg: 1.35678 to 1.36)
#             selling_price = round(selling_price, 2)

#             return selling_price

#     def classification():

#         # get input from users
#         with st.form('Classification'):

#             col1,col2,col3 = st.columns([0.5,0.1,0.5])

#             with col1:

#                 item_date = st.date_input(label='Item Date', min_value=date(2020,7,1), 
#                                         max_value=date(2021,5,31), value=date(2020,7,1))
                
#                 quantity_log = st.text_input(label='Quantity Tons (Min: 0.00001 & Max: 1000000000)')

#                 country = st.selectbox(label='Country', options=options.country_values)

#                 item_type = st.selectbox(label='Item Type', options=options.item_type_values)

#                 thickness_log = st.number_input(label='Thickness', min_value=0.1, max_value=2500000.0, value=1.0)

#                 product_ref = st.selectbox(label='Product Ref', options=options.product_ref_values)

#             with col3:

#                 delivery_date = st.date_input(label='Delivery Date', min_value=date(2020,8,1), 
#                                             max_value=date(2022,2,28), value=date(2020,8,1))
                
#                 customer = st.text_input(label='Customer ID (Min: 12458000 & Max: 2147484000)')

#                 selling_price_log = st.text_input(label='Selling Price (Min: 0.1 & Max: 100001000)')

#                 application = st.selectbox(label='Application', options=options.application_values)

#                 width = st.number_input(label='Width', min_value=1.0, max_value=2990000.0, value=1.0)

#                 st.write('')
#                 st.write('')
#                 button = st.form_submit_button(label='SUBMIT')
#                 style_submit_button()
                
#         # give information to users
#         col1,col2 = st.columns([0.65,0.35])
#         with col2:
#             st.caption(body='*Min and Max values are reference only')

#         # user entered the all input values and click the button
#         if button:
            
#             # load the classification pickle model
#             with open(r'models\classification_model.pkl', 'rb') as f:
#                 model = pickle.load(f)
            
#             # make array for all user input values in required order for model prediction
#             user_data = np.array([[customer, 
#                                 country, 
#                                 options.item_type_dict[item_type], 
#                                 application, 
#                                 width, 
#                                 product_ref, 
#                                 np.log(float(quantity_log)), 
#                                 np.log(float(thickness_log)),
#                                 np.log(float(selling_price_log)),
#                                 item_date.day, item_date.month, item_date.year,
#                                 delivery_date.day, delivery_date.month, delivery_date.year]])
            
#             # model predict the status based on user input
#             y_pred = model.predict(user_data)

#             # we get the single output in list, so we access the output using index method
#             status = y_pred[0]

#             return status

# streamlit_config()

# tab1, tab2 = st.tabs(['PREDICT SELLING PRICE', 'PREDICT STATUS'])

# with tab1:

#     try:
    
#         selling_price = prediction.regression()

#         if selling_price:
#             # apply custom css style for prediction text
#             style_prediction()
#             st.markdown(f'### <div class="center-text">Predicted Selling Price = {selling_price}</div>', unsafe_allow_html=True)
#             st.balloons()
    
#     except ValueError:

#         col1,col2,col3 = st.columns([0.26,0.55,0.26])

#         with col2:
#             st.warning('##### Quantity Tons / Customer ID is empty')

# with tab2:

#     try:

#         status = prediction.classification()

#         if status == 1:
            
#             # apply custom css style for prediction text
#             style_prediction()
#             st.markdown(f'### <div class="center-text">Predicted Status = Won</div>', unsafe_allow_html=True)
#             st.balloons()
            
#         elif status == 0:
            
#             # apply custom css style for prediction text
#             style_prediction()
#             st.markdown(f'### <div class="center-text">Predicted Status = Lost</div>', unsafe_allow_html=True)
#             st.snow()
    
#     except ValueError:

#         col1,col2,col3 = st.columns([0.15,0.70,0.15])

#         with col2:
#             st.warning('##### Quantity Tons / Customer ID / Selling Price is empty')

import streamlit as st
from datetime import date
import numpy as np
import pickle

# Page configuration
st.set_page_config(page_title='Industrial Copper Modeling')

# Page header transparent color
page_background_color = """
<style>
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
</style>
"""
st.markdown(page_background_color, unsafe_allow_html=True)

# Title and position
st.markdown(f'<h1 style="text-align: center;">Industrial Copper Modeling</h1>', unsafe_allow_html=True)

# Custom style for submit button - color and width
def style_submit_button():
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #367F89;
            color: white;
            width: 70%;
        }
        </style>
    """, unsafe_allow_html=True)

# Custom style for prediction result text - color and position
def style_prediction():
    st.markdown("""
        <style>
        .center-text {
            text-align: center;
            color: #20CA0C;
        }
        </style>
    """, unsafe_allow_html=True)

# User input options
class options:
    country_values = [25.0, 26.0, 27.0, 28.0, 30.0, 32.0, 38.0, 39.0, 40.0, 77.0, 
                      78.0, 79.0, 80.0, 84.0, 89.0, 107.0, 113.0]
    status_values = ['Won', 'Lost', 'Draft', 'To be approved', 'Not lost for AM',
                     'Wonderful', 'Revised', 'Offered', 'Offerable']
    status_dict = {'Lost': 0, 'Won': 1, 'Draft': 2, 'To be approved': 3, 'Not lost for AM': 4,
                   'Wonderful': 5, 'Revised': 6, 'Offered': 7, 'Offerable': 8}
    item_type_values = ['W', 'WI', 'S', 'PL', 'IPL', 'SLAWR', 'Others']
    item_type_dict = {'W': 5.0, 'WI': 6.0, 'S': 3.0, 'Others': 1.0, 'PL': 2.0, 'IPL': 0.0, 'SLAWR': 4.0}
    application_values = [2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0, 
                          27.0, 28.0, 29.0, 38.0, 39.0, 40.0, 41.0, 42.0, 56.0, 58.0, 
                          59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0]
    product_ref_values = [611728, 611733, 611993, 628112, 628117, 628377, 640400, 
                          640405, 640665, 164141591, 164336407, 164337175, 929423819, 
                          1282007633, 1332077137, 1665572032, 1665572374, 1665584320, 
                          1665584642, 1665584662, 1668701376, 1668701698, 1668701718, 
                          1668701725, 1670798778, 1671863738, 1671876026, 1690738206, 
                          1690738219, 1693867550, 1693867563, 1721130331, 1722207579]

# Get input data from users both regression and classification methods
class prediction:
    def regression():
        # Get input from users
        with st.form('Regression'):
            col1, col2, col3 = st.columns([0.5, 0.1, 0.5])
            with col1:
                item_date = st.date_input(label='Item Date', min_value=date(2020, 7, 1), 
                                          max_value=date(2021, 5, 31), value=date(2020, 7, 1))
                quantity_log = st.text_input(label='Quantity Tons (Min: 0.00001 & Max: 1000000000)')
                country = st.selectbox(label='Country', options=options.country_values)
                item_type = st.selectbox(label='Item Type', options=options.item_type_values)
                thickness_log = st.number_input(label='Thickness', min_value=0.1, max_value=2500000.0, value=1.0)
                product_ref = st.selectbox(label='Product Ref', options=options.product_ref_values)
            with col3:
                delivery_date = st.date_input(label='Delivery Date', min_value=date(2020, 8, 1), 
                                              max_value=date(2022, 2, 28), value=date(2020, 8, 1))
                customer = st.text_input(label='Customer ID (Min: 12458000 & Max: 2147484000)')
                status = st.selectbox(label='Status', options=options.status_values)
                application = st.selectbox(label='Application', options=options.application_values)
                width = st.number_input(label='Width', min_value=1.0, max_value=2990000.0, value=1.0)
                st.write('')
                st.write('')
                button = st.form_submit_button(label='SUBMIT')
                style_submit_button()
        # Give information to users
        col1, col2 = st.columns([0.65, 0.35])
        with col2:
            st.caption(body='*Min and Max values are reference only')
        # User entered all input values and clicked the button
        if button:
            # Load the regression pickle model
            with open(r'models\regression_model.pkl', 'rb') as f:
                model = pickle.load(f)
            # Make array for all user input values in required order for model prediction
            user_data = np.array([[customer, country, options.status_dict[status], options.item_type_dict[item_type], 
                                   application, width, product_ref, np.log(float(quantity_log)), 
                                   np.log(float(thickness_log)), item_date.day, item_date.month, item_date.year, 
                                   delivery_date.day, delivery_date.month, delivery_date.year]])
            # Model predict the selling price based on user input
            y_pred = model.predict(user_data)
            # Inverse transformation for log transformation data
            selling_price = np.exp(y_pred[0])
            # Round the value to 2 decimal points (Eg: 1.35678 to 1.36)
            selling_price = round(selling_price, 2)
            return selling_price

    def classification():
        # Get input from users
        with st.form('Classification'):
            col1, col2, col3 = st.columns([0.5, 0.1, 0.5])
            with col1:
                item_date = st.date_input(label='Item Date', min_value=date(2020, 7, 1), 
                                          max_value=date(2021, 5, 31), value=date(2020, 7, 1))
                quantity_log = st.text_input(label='Quantity Tons (Min: 0.00001 & Max: 1000000000)')
                country = st.selectbox(label='Country', options=options.country_values)
                item_type = st.selectbox(label='Item Type', options=options.item_type_values)
                thickness_log = st.number_input(label='Thickness', min_value=0.1, max_value=2500000.0, value=1.0)
                product_ref = st.selectbox(label='Product Ref', options=options.product_ref_values)
            with col3:
                delivery_date = st.date_input(label='Delivery Date', min_value=date(2020, 8, 1), 
                                              max_value=date(2022, 2, 28), value=date(2020, 8, 1))
                customer = st.text_input(label='Customer ID (Min: 12458000 & Max: 2147484000)')
                selling_price_log = st.text_input(label='Selling Price (Min: 0.1 & Max: 100001000)')
                application = st.selectbox(label='Application', options=options.application_values)
                width = st.number_input(label='Width', min_value=1.0, max_value=2990000.0, value=1.0)
                st.write('')
                st.write('')
                button = st.form_submit_button(label='SUBMIT')
                style_submit_button()
        # Give information to users
        col1, col2 = st.columns([0.65, 0.35])
        with col2:
            st.caption(body='*Min and Max values are reference only')
        # User entered all input values and clicked the button
        if button:
            # Load the classification pickle model
            with open(r'models\classification_model.pkl', 'rb') as f:
                model = pickle.load(f)
            # Make array for all user input values in required order for model prediction
            user_data = np.array([[customer, country, options.item_type_dict[item_type], application, width, 
                                   product_ref, np.log(float(quantity_log)), np.log(float(thickness_log)), 
                                   np.log(float(selling_price_log)), item_date.day, item_date.month, item_date.year, 
                                   delivery_date.day, delivery_date.month, delivery_date.year]])
            # Model predict the status based on user input
            y_pred = model.predict(user_data)
            # We get the single output in list, so we access the output using index method
            status = y_pred[0]
            return status

tab1, tab2 = st.tabs(['PREDICT SELLING PRICE', 'PREDICT STATUS'])

with tab1:
    try:
        selling_price = prediction.regression()
        if selling_price:
            # Apply custom css style for prediction text
            style_prediction()
            st.markdown(f'### <div class="center-text">Predicted Selling Price = {selling_price}</div>', unsafe_allow_html=True)
            st.balloons()
    except ValueError:
        col1, col2, col3 = st.columns([0.26, 0.55, 0.26])
        with col2:
            st.warning('##### Quantity Tons / Customer ID is empty')

with tab2:
    try:
        status = prediction.classification()
        if status == 1:
            # Apply custom css style for prediction text
            style_prediction()
            st.markdown(f'### <div class="center-text">Predicted Status = Won</div>', unsafe_allow_html=True)
            st.balloons()
        elif status == 0:
            # Apply custom css style for prediction text
            style_prediction()
            st.markdown(f'### <div class="center-text">Predicted Status = Lost</div>', unsafe_allow_html=True)
            st.snow()
    except ValueError:
        col1, col2, col3 = st.columns([0.15, 0.70, 0.15])
        with col2:
            st.warning('##### Quantity Tons / Customer ID / Selling Price is empty')
