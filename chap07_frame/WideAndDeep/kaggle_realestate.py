import numpy as np
import pandas as pd
import tensorflow as tf

print('Loading data ...')

train = pd.read_csv('../train_2016_v2.csv')
prop = pd.read_csv('../properties_2016.csv')
sample = pd.read_csv('../sample_submission.csv')

#print("rowcount {0}".format(len(train.index)))

# print('Binding to float32')

# for c, dtype in zip(prop.columns, prop.dtypes):
#     if dtype == np.float64:
#         prop[c] = prop[c].astype(np.float32)

print('Creating training set ...')

df_train = train.merge(prop, how='left', on='parcelid')

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)

train_columns = x_train.columns

#print(x_train.dtypes)
#print(dir(x_train.dtypes))

CATEGORICAL_COLUMNS = [ _name for _name, _type in x_train.dtypes.iteritems() if _type == 'object' ]
#print(CATEGORICAL_COLUMNS)
# 여기서 column을 바꾸자

CONTINUOUS_COLUMNS = [ _name for _name, _type in x_train.dtypes.iteritems() if _type != 'object' ]
#print(CONTINUOUS_COLUMNS)

LABEL_COLUMN = 'logerror'


def build_estimator(model_dir, model_type):
    """Build an estimator."""
    # Sparse base columns.

    # ['airconditioningtypeid', 'architecturalstyletypeid', 'basementsqft', 'bathroomcnt', 'bedroomcnt', 'buildingclasstypeid'
    airconditioningtypeid = tf.contrib.layers.real_valued_column("airconditioningtypeid")
    architecturalstyletypeid = tf.contrib.layers.real_valued_column("architecturalstyletypeid")
    basementsqft = tf.contrib.layers.real_valued_column("basementsqft")
    bathroomcnt = tf.contrib.layers.real_valued_column("bathroomcnt")
    bedroomcnt = tf.contrib.layers.real_valued_column("bedroomcnt")
    buildingclasstypeid = tf.contrib.layers.real_valued_column("buildingclasstypeid")

    # , 'buildingqualitytypeid', 'calculatedbathnbr', 'decktypeid', 'finishedfloor1squarefeet', 'calculatedfinishedsquarefeet', 'finishedsquarefeet12', 'finishedsquarefeet13',
    buildingqualitytypeid = tf.contrib.layers.real_valued_column("buildingqualitytypeid")
    calculatedbathnbr = tf.contrib.layers.real_valued_column("calculatedbathnbr")
    decktypeid = tf.contrib.layers.real_valued_column("decktypeid")
    finishedfloor1squarefeet = tf.contrib.layers.real_valued_column("finishedfloor1squarefeet")
    calculatedfinishedsquarefeet = tf.contrib.layers.real_valued_column("calculatedfinishedsquarefeet")
    finishedsquarefeet12 = tf.contrib.layers.real_valued_column("finishedsquarefeet12")
    finishedsquarefeet13 = tf.contrib.layers.real_valued_column("finishedsquarefeet13")

    # 'finishedsquarefeet15', 'finishedsquarefeet50', 'finishedsquarefeet6', 'fips', 'fireplacecnt', 'fullbathcnt', 'garagecarcnt',

    finishedsquarefeet15 = tf.contrib.layers.real_valued_column("finishedsquarefeet15")
    finishedsquarefeet50 = tf.contrib.layers.real_valued_column("finishedsquarefeet50")
    finishedsquarefeet6 = tf.contrib.layers.real_valued_column("finishedsquarefeet6")
    fips = tf.contrib.layers.real_valued_column("fips")
    fireplacecnt = tf.contrib.layers.real_valued_column("fireplacecnt")
    fullbathcnt = tf.contrib.layers.real_valued_column("fullbathcnt")
    garagecarcnt = tf.contrib.layers.real_valued_column("garagecarcnt")

    # 'garagetotalsqft', 'heatingorsystemtypeid', 'latitude', 'longitude', 'lotsizesquarefeet', 'poolcnt', 'poolsizesum', 'pooltypeid10'

    garagetotalsqft = tf.contrib.layers.real_valued_column("garagetotalsqft")
    heatingorsystemtypeid = tf.contrib.layers.real_valued_column("heatingorsystemtypeid")
    latitude = tf.contrib.layers.real_valued_column("latitude")
    longitude = tf.contrib.layers.real_valued_column("longitude")
    lotsizesquarefeet = tf.contrib.layers.real_valued_column("lotsizesquarefeet")
    poolcnt = tf.contrib.layers.real_valued_column("poolcnt")
    garagecarcnt = tf.contrib.layers.real_valued_column("garagecarcnt")
    poolsizesum = tf.contrib.layers.real_valued_column("poolsizesum")
    pooltypeid10 = tf.contrib.layers.real_valued_column("pooltypeid10")

    # , 'pooltypeid2', 'pooltypeid7', 'propertylandusetypeid', 'rawcensustractandblock', 'regionidcity', 'regionidcounty',
    pooltypeid2 = tf.contrib.layers.real_valued_column("pooltypeid2")
    pooltypeid7 = tf.contrib.layers.real_valued_column("pooltypeid7")
    propertylandusetypeid = tf.contrib.layers.real_valued_column("propertylandusetypeid")
    rawcensustractandblock = tf.contrib.layers.real_valued_column("rawcensustractandblock")
    regionidcity = tf.contrib.layers.real_valued_column("regionidcity")
    regionidcounty = tf.contrib.layers.real_valued_column("regionidcounty")
    regionidneighborhood = tf.contrib.layers.real_valued_column("regionidneighborhood")

    # 'regionidneighborhood', 'regionidzip', 'roomcnt', 'storytypeid', 'threequarterbathnbr', 'typeconstructiontypeid', 'unitcnt', 'yardbuildingsqft17'
    regionidneighborhood = tf.contrib.layers.real_valued_column("regionidneighborhood")
    regionidzip = tf.contrib.layers.real_valued_column("regionidzip")
    roomcnt = tf.contrib.layers.real_valued_column("roomcnt")
    storytypeid = tf.contrib.layers.real_valued_column("storytypeid")
    threequarterbathnbr = tf.contrib.layers.real_valued_column("threequarterbathnbr")
    typeconstructiontypeid = tf.contrib.layers.real_valued_column("typeconstructiontypeid")
    unitcnt = tf.contrib.layers.real_valued_column("unitcnt")
    yardbuildingsqft17 = tf.contrib.layers.real_valued_column("yardbuildingsqft17")
    garagecarcnt = tf.contrib.layers.real_valued_column("garagecarcnt")

    # , 'yardbuildingsqft26', 'yearbuilt', 'numberofstories', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'assessmentyear', 'landtaxvaluedollarcnt',

    yardbuildingsqft26 = tf.contrib.layers.real_valued_column("yardbuildingsqft26")
    yearbuilt = tf.contrib.layers.real_valued_column("yearbuilt")
    numberofstories = tf.contrib.layers.real_valued_column("numberofstories")
    structuretaxvaluedollarcnt = tf.contrib.layers.real_valued_column("structuretaxvaluedollarcnt")
    taxvaluedollarcnt = tf.contrib.layers.real_valued_column("taxvaluedollarcnt")
    assessmentyear = tf.contrib.layers.real_valued_column("assessmentyear")
    landtaxvaluedollarcnt = tf.contrib.layers.real_valued_column("landtaxvaluedollarcnt")

    # 'taxamount', 'taxdelinquencyyear', 'censustractandblock']
    taxamount = tf.contrib.layers.real_valued_column("taxamount")
    taxdelinquencyyear = tf.contrib.layers.real_valued_column("taxdelinquencyyear")
    censustractandblock = tf.contrib.layers.real_valued_column("censustractandblock")

    # spsrse
    # ['hashottuborspa', 'fireplaceflag', 'taxdelinquencyflag']]
    hashottuborspa = tf.contrib.layers.sparse_column_with_keys(
        "hashottuborspa", keys=["False", "true"])
    fireplaceflag = tf.contrib.layers.sparse_column_with_keys(
        "fireplaceflag",  keys=["False", "true"])
    taxdelinquencyflag = tf.contrib.layers.sparse_column_with_keys(
        "taxdelinquencyflag", keys=["False", "N"])

    # Wide columns and deep columns.
    # continues columns feeding
    wide_columns = [airconditioningtypeid, architecturalstyletypeid, basementsqft, bathroomcnt, bedroomcnt,
                    buildingclasstypeid, buildingqualitytypeid, calculatedbathnbr, decktypeid, finishedfloor1squarefeet,
                    calculatedfinishedsquarefeet, finishedsquarefeet12, finishedsquarefeet13, finishedsquarefeet15,
                    finishedsquarefeet50, finishedsquarefeet6, fips, fireplacecnt, fullbathcnt, garagecarcnt,
                    garagetotalsqft, heatingorsystemtypeid, latitude, longitude, lotsizesquarefeet, poolcnt,
                    poolsizesum, pooltypeid10, pooltypeid2, pooltypeid7, propertylandusetypeid, rawcensustractandblock,
                    regionidcity, regionidcounty, regionidneighborhood, regionidzip, roomcnt, storytypeid,
                    threequarterbathnbr, typeconstructiontypeid, unitcnt, yardbuildingsqft17, yardbuildingsqft26,
                    yearbuilt, numberofstories, structuretaxvaluedollarcnt, taxvaluedollarcnt, assessmentyear,
                    landtaxvaluedollarcnt, taxamount, taxdelinquencyyear, censustractandblock]
    # categorical columns + continuous feeding
    deep_columns = [
        tf.contrib.layers.embedding_column(hashottuborspa, dimension=8),
        tf.contrib.layers.embedding_column(fireplaceflag, dimension=8),
        tf.contrib.layers.embedding_column(taxdelinquencyflag, dimension=8),
        airconditioningtypeid, architecturalstyletypeid, basementsqft, bathroomcnt, bedroomcnt, buildingclasstypeid,
        buildingqualitytypeid, calculatedbathnbr, decktypeid, finishedfloor1squarefeet, calculatedfinishedsquarefeet,
        finishedsquarefeet12, finishedsquarefeet13, finishedsquarefeet15, finishedsquarefeet50, finishedsquarefeet6,
        fips, fireplacecnt, fullbathcnt, garagecarcnt, garagetotalsqft, heatingorsystemtypeid, latitude, longitude,
        lotsizesquarefeet, poolcnt, poolsizesum, pooltypeid10, pooltypeid2, pooltypeid7, propertylandusetypeid,
        rawcensustractandblock, regionidcity, regionidcounty, regionidneighborhood, regionidzip, roomcnt, storytypeid,
        threequarterbathnbr, typeconstructiontypeid, unitcnt, yardbuildingsqft17, yardbuildingsqft26, yearbuilt,
        numberofstories, structuretaxvaluedollarcnt, taxvaluedollarcnt, assessmentyear, landtaxvaluedollarcnt,
        taxamount, taxdelinquencyyear, censustractandblock
    ]

    if model_type == "wide":
        m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                              feature_columns=wide_columns)
    elif model_type == "deep":
        m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                           feature_columns=deep_columns,
                                           hidden_units=[100, 50])
    elif model_type == "wdRegress":
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[100, 50],
            fix_global_step_increment_bug=True)
    else:
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[500, 200],
            fix_global_step_increment_bug=True)
    return m


def input_fn(x_train,y_train ):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(np.nan_to_num(x_train[k])) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols2 = {
    k: tf.SparseTensor(
        indices=[[i, 0] for i in range(x_train[k].size)],
        values=x_train[k].astype(str).fillna('').values,
        dense_shape=[x_train[k].size, 1])
        for k in CATEGORICAL_COLUMNS}


  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols2)
  # Converts the label column into a constant Tensor.
  label = tf.constant(y_train)
  # Returns the feature columns and the label.
  return feature_cols, label


def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
    """Train and evaluate the model."""
    # train_file_name, test_file_name = maybe_download(train_data, test_data)
    train = pd.read_csv('../train_2016_v2.csv')
    prop = pd.read_csv('../properties_2016.csv')
    sample = pd.read_csv('../sample_submission.csv')

    print('Creating training set ...')

    for c, dtype in zip(prop.columns, prop.dtypes):
        if dtype == np.float64:
            prop[c] = prop[c].astype(np.float32)

    df_train = train.merge(prop, how='left', on='parcelid')

    x_train = df_train.drop(
        ['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
    y_train = df_train['logerror'].values
    print(x_train.shape, y_train.shape)

    for c in CATEGORICAL_COLUMNS:
        x_train[c] = x_train[c].fillna('False').astype(str)


    split = 80000
    x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]


    model_dir = tempfile.mkdtemp() if not model_dir else model_dir
    print("model directory = %s" % model_dir)

    m = build_estimator(model_dir, model_type)
    m.fit(input_fn=lambda: input_fn(x_train, y_train), steps=train_steps)
    results = m.evaluate(input_fn=lambda: input_fn(x_valid, y_valid), steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))


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
  default="wdRegress",
  help="Valid model types: {'wide', 'deep', 'wdRegress', 'wide_n_deep',}."
)
parser.add_argument(
  "--train_steps",
  type=int,
  default=1000,
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


