import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# preperation
from sklearn.model_selection import train_test_split,KFold
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression,SelectFromModel
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder



# models
import random
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostClassifier
#from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
# Evaluation
import math
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error
import sklearn.metrics as metrics
import pickle

# Preprocess the data

def labelEncoder(gearbox, fuelType, notRepairedDamage):

    pkl_file1 = open('models/gearbox_encoder.pkl', 'rb')
    l1 = pickle.load(pkl_file1)
    pkl_file1.close()
    gearbox = l1.transform(gearbox)

    pkl_file2 = open('models/fuelType_encoder.pkl', 'rb')
    l2 = pickle.load(pkl_file2)
    pkl_file2.close()
    fuelType = l2.transform(fuelType)

    pkl_file3 = open('models/notRepairedDamage_encoder.pkl', 'rb')
    l3 = pickle.load(pkl_file3)
    pkl_file3.close()
    notRepairedDamage= l3.transform(notRepairedDamage)

    return [gearbox, fuelType, notRepairedDamage]



def scale(gearbox, fuelType, notRepairedDamage):

    price_df = pd.DataFrame([gearbox], columns= ['gearbox'])
    price_df['fuelType'] = fuelType
    price_df['notRepairedDamage'] = notRepairedDamage
    file = open('models/scale_data.pkl', 'rb')
    l3 = pickle.load(file)
    file.close()
    price_df = l3.transform(price_df)
    return price_df

# Predict Function

def predict_fuc(data,model):
    y_pred = model.predict(data)
    print(y_pred)
    return y_pred

def runner(car_specifics):
    data = pd.DataFrame.from_dict(car_specifics)
    gearbox = data['gearbox']
    fuelType = data['fuelType']
    notRepairedDamage = data['notRepairedDamage']
    gearbox, fuelType, notRepairedDamage = labelEncoder(gearbox, fuelType, notRepairedDamage)
    raw_data = scale(gearbox, fuelType, notRepairedDamage)
    raw_data1 = pd.DataFrame(raw_data, columns = ['gearbox', 'fuelType', 'notRepairedDamage'])
    loaded_model = pickle.load(open("models/final_model.pkl", "rb"))
    result  = predict_fuc(raw_data1,loaded_model)
    # Unscaling the Result
    file = open('models/scale_data_y.pkl', 'rb')
    l = pickle.load(file)
    file.close()
    price = l.inverse_transform(result)
    return price
