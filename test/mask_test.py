import pandas as pd
from environs import Env
from timer import timer

env = Env()
env.read_env()
FINAL_DATA = env.str("FINAL_DATA")

preprocessed_data = pd.read_csv(FINAL_DATA, engine="pyarrow")
preprocessed_data.tags = preprocessed_data.tags.apply(lambda x: x.split('|'))



tags = ['linux', 'apache', 'virtualhost']
tags = set(tags)
with timer("initial mask time", 5):
    mask = [True if len(tags.intersection(set(preprocessed_data.iloc[i].tags))) >= 1 else False
            for i in range(preprocessed_data.shape[0])]
    data_new = preprocessed_data[mask]
data_new.reset_index(inplace=True, drop=True)

with timer("new mask time", 5):
    preprocessed_data['new'] = True
    mask = preprocessed_data.tags
    mask = [True if len(tags.intersection(set(preprocessed_data.iloc[i].tags))) >= 1 else False
            for i in range(preprocessed_data.shape[0])]
    data_new = preprocessed_data[mask]
data_new.reset_index(inplace=True, drop=True)

