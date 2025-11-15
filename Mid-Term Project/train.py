import pandas as pd 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import numpy as np
from sklearn.feature_extraction import DictVectorizer


# 1. Load dataset
df = pd.read_csv('fruit_classification_dataset.csv')


# 2. Split the dataset
df_full_train, df_test = train_test_split(df,test_size=0.2, random_state=1)
df_train, df_val =train_test_split(df_full_train, test_size=0.25, random_state=1)

# 3. Encode target variable
y_train = df_train.fruit_name.values
y_val = df_val.fruit_name.values

encoder = LabelEncoder()

y_train = encoder.fit_transform(y_train)
y_val = encoder.transform(y_val)


# Drop target
df_train = df_train.drop('fruit_name', axis=1)
df_val = df_val.drop('fruit_name', axis=1)


# One-hot encode categorical features
dv = DictVectorizer(sparse=True)

train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')

X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

# 5. Train XGBoost model (using your tuned parameters)
model = XGBClassifier(
    n_estimators=500,       # number of trees
    max_depth=10,           # depth of each tree
    eta=0.01,      # step size shrinkage
    subsample=0.6,          # sample ratio of training instance
    colsample_bytree=1.0,   # subsample ratio of columns when constructing each tree
    min_child_weight = 1,
    random_state=1,
    eval_metric='mlogloss', # for multi-class classification
)

model.fit(X_train, y_train)


#6. Save model, encoder, and column names
with open('model.bin', 'wb') as f_out:
   pickle.dump((encoder, dv, model), f_out)


print("Model trained and saved as model.bin")