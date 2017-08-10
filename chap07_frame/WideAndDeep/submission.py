import pandas as pd

sub = pd.read_csv('../sample_submission.csv')



for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = '1'