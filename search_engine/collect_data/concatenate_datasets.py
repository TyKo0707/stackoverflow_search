import pandas as pd
from environs import Env

env = Env()
env.read_env()
RAW_DATA_PATH = env.str("RAW_DATA_PATH")

preprocessed_data1 = pd.read_parquet(f'{RAW_DATA_PATH}out1.gzip')
preprocessed_data2 = pd.read_parquet(f'{RAW_DATA_PATH}out2.gzip')
preprocessed_data3 = pd.read_parquet(f'{RAW_DATA_PATH}out3.gzip')
preprocessed_data4 = pd.read_parquet(f'{RAW_DATA_PATH}out4.gzip')
preprocessed_data5 = pd.read_parquet(f'{RAW_DATA_PATH}out5.gzip')

main_data1 = pd.concat([preprocessed_data1, preprocessed_data2])
main_data2 = pd.concat([preprocessed_data3, preprocessed_data4, preprocessed_data5])

main_data1.to_parquet(f"{RAW_DATA_PATH}raw_data1.gzip", compression='gzip', index=False)
main_data2.to_parquet(f"{RAW_DATA_PATH}raw_data2.gzip", compression='gzip', index=False)
