import pandas as pd
from environs import Env
from timer import timer
import swifter

env = Env()
env.read_env()
FINAL_DATA = env.str("FINAL_DATA")

preprocessed_data = pd.read_csv(FINAL_DATA, engine="pyarrow")
test1 = preprocessed_data.tags
test2 = preprocessed_data.tags

with timer('apply function with 1 method', 5):
    test1 = test1.apply(lambda x: x.split('|'))

with timer('apply function with 2 method', 5):
    test2 = test2.swifter.apply(lambda x: x.split('|'))

# [apply function with 1 method] done in 0.29714 s
# [apply function with 2 method] done in 0.79506 s
