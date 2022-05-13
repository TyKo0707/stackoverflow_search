import pandas as pd
from environs import Env
from timer import timer

env = Env()
env.read_env()
FINAL_DATA = env.str("FINAL_DATA")

with timer('open csv with 1 method', 0):
    preprocessed_data1 = pd.read_csv(FINAL_DATA)
with timer('open csv with 2 method', 0):
    preprocessed_data2 = pd.read_csv(FINAL_DATA, engine="pyarrow")

# [open csv with 1 method] done in 4 s
# [open csv with 2 method] done in 1 s
