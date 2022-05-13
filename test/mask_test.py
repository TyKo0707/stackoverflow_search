import pandas as pd
from environs import Env
from timer import timer

env = Env()
env.read_env()
FINAL_DATA = env.str("FINAL_DATA")

preprocessed_data = pd.read_csv(FINAL_DATA, engine="pyarrow")
preprocessed_data1 = preprocessed_data[:500]

tags = ['linux', 'apache', 'virtualhost']

with timer("new mask time", 5):
    mask1 = preprocessed_data1['tags'].str.contains('|'.join(tags))
    data_new1 = preprocessed_data1[mask1]
data_new1.reset_index(inplace=True, drop=True)
print(data_new1.shape[0])

with timer("initial mask time", 5):
    preprocessed_data1.tags = preprocessed_data1.tags.apply(lambda x: x.split('|'))
    mask2 = [True if len(set(tags).intersection(set(preprocessed_data1.iloc[i].tags))) >= 1 else False
             for i in range(preprocessed_data1.shape[0])]
    data_new2 = preprocessed_data1[mask2]
data_new2.reset_index(inplace=True, drop=True)
print(data_new2.shape[0])
