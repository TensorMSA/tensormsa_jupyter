# Load in our libraries
import pandas as pd #http://pandas.pydata.org/ #Python Data Analysis Library
import numpy as np #NumPy is the fundamental package for scientific computing with Python. It contains among other thing
import re #정규식
import sklearn # http://scikit-learn.org/stable/ Simple and efficient tools for data mining and data analysis
import xgboost as xgb # Scalable and Flexible Gradient Boosting https://xgboost.readthedocs.io/en/latest/
import seaborn as sns  # https://seaborn.pydata.org/   seaborn: statistical data visualization
import matplotlib.pyplot as plt

import plotly.offline as py  #Visualize Data, Togethe #https://plot.ly/
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

# Going to use these 5 base models for the stacking
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold;
import tensorflow as tf

# Load in the train and test datasets
train = pd.read_csv('../data/titanic/train.csv')
test = pd.read_csv('../data/titanic/test.csv')

# Store our passenger ID for easy access
PassengerId = test['PassengerId']
train_passengerid = train['PassengerId']
#print(train.columns)
train_results_xgb = train[['PassengerId','Survived']]

full_data = [train, test]
print('train row {0}'.format(len(train.index)))
print('test row {0}'.format(len(test.index)))

# Some features of my own that I have added in
# Gives the length of the name
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)
# print(train['Name_length'])

# Feature that tells whether a passenger had a cabin on the Titanic
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

print(train['Has_Cabin'].head(10))

# Feature engineering steps taken from Sina
# Create new feature FamilySize as a combination of SibSp and Parch

# 새로운 Feature를 만듭니다.(친척과 자식)

for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Create new feature IsAlone from FamilySize
# IsAlone이 Feature를 새로 만든다.
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    # print(dataset['IsAlone'])

# Nan값을 S로 치환한다.
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# Nan값을 평균값으로 취소한다.
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
print(train['CategoricalFare'].values)
# Create a New feature CategoricalAge
# qcut는 같은 갯수로 나눔
# cut은 실수값의 경계선으로 자름

for dataset in full_data:
    age_avg = dataset['Age'].mean()  # 평균
    age_std = dataset['Age'].std()  # 표준편차 구하기 : 분산의 제곱근
    age_null_count = dataset['Age'].isnull().sum()
    print("start random age {0}".format(age_avg - age_std))
    print("end random age {0}".format(age_avg + age_std))

    # nan값을 처리 하는 로직.
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
train['CategoricalAge'] = pd.cut(train['Age'], 5)
print(pd.cut(train['Age'], 5).values)


# 정규식을 통해서 Lady.Dr Major등을 추출함
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


print(get_title('Cumings, Mrs. John Bradley (Florence Briggs Th...'))

# Title 컬럼 생성
# Create a new feature Title, containing the titles of passenger names
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print(dataset['Title'].head(3))

for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # Mapping Fare
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    # Mapping Age
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'];

print(train['Title'].head(3))
print(train['Embarked'].head(3))
print(train['Fare'].head(3))

# Feature selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_elements, axis=1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis=1)
test = test.drop(drop_elements, axis=1)


# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_train = train['Survived'].ravel()
x_train = train.drop(['Survived'], axis=1)

COLUMNS = ["Survived", "Pclass", "Sex", "Age", "Parch", "Fare", "Embarked", "Name_length", "Has_Cabin", "FamilySize"
    , "IsAlone", "Title"]

CONTINUOUS_COLUMNS = [ "Age", "Parch", "Fare", "Embarked", "Name_length", "FamilySize"
    , "Title","IsAlone"]

CATEGORICAL_COLUMNS = ["Pclass","Has_Cabin","Sex"]

LABEL_COLUMN = "Survived"


def build_estimator(model_dir, model_type):

    sparse_columns = {s: tf.contrib.layers.sparse_column_with_hash_bucket(s, hash_bucket_size=1000)
                      for s in CATEGORICAL_COLUMNS}
    wide_columns = [tf.contrib.layers.real_valued_column(c) for c in CONTINUOUS_COLUMNS]
    deep_columns = [tf.contrib.layers.embedding_column(_tv, dimension=8)
                    for _id, _tv in sparse_columns.items()]

    #deep_columns.extend(wide_columns)

    #tf.contrib.layers.embedding_column(workclass, dimension=8),
    #deep_columns.update(wide_columns)

    if model_type == "wide":
        m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                              feature_columns=wide_columns)
    elif model_type == "deep":
        m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                           feature_columns=deep_columns,
                                           hidden_units=[100, 50])
    else:
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[100, 50],
            fix_global_step_increment_bug=True)
    return m

def input_fn(x_train,y_train ):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(x_train[k].values) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {
      k: tf.SparseTensor(
          indices=[[i, 0] for i in range(x_train[k].size)],
          values=x_train[k].astype(str).values,
          dense_shape=[x_train[k].size, 1])
      for k in CATEGORICAL_COLUMNS if len(CATEGORICAL_COLUMNS) != 0}
  #categorical_cols_conti = {k: tf.constant(x_train[k].values) for k in CONTINUOUS_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  #feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  label = tf.constant(y_train)
  # Returns the feature columns and the label.
  return feature_cols, label


def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
  """Train and evaluate the model."""

  model_dir = tempfile.mkdtemp() if not model_dir else model_dir
  print("model directory = %s" % model_dir)

  m = build_estimator(model_dir, model_type)
  m.fit(input_fn=lambda: input_fn(x_train,y_train ), steps=train_steps)
  results = m.evaluate(input_fn=lambda: input_fn(x_train, y_train), steps=1)

  for key in sorted(results):
    print("%s: %s" % (key, results[key]))

  predict = m.predict(input_fn=lambda: input_fn(x_train, y_train))
  train['Survived_wd'] = list(predict)
  #print(train[['Survivied','result'].a])

  train['result'] = train[['Survived', 'Survived_wd']].apply(
      lambda x: 1 if x['Survived'] == x['Survived_wd'] else 0, axis=1)

  print("Wdnn {0}".format(train['result'].sum(axis=0) / len(train.index)))

#IsAlone


import argparse
import sys
import tempfile


parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument(
  "--model_dir",
  type=str,
  default="",
  help="Base directory for output models."
)
parser.add_argument(
  "--model_type",
  type=str,
  default="wide_n_deep",
  help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
)
parser.add_argument(
  "--train_steps",
  type=int,
  default=200,
  help="Number of training steps."
)
parser.add_argument(
  "--train_data",
  type=str,
  default="",
  help="Path to the training data."
)
parser.add_argument(
  "--test_data",
  type=str,
  default="",
  help="Path to the test data."
)
FLAGS, unparsed = parser.parse_known_args()

train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data)
