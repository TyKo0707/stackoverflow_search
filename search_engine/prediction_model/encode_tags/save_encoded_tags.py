from encode_tags import Encoder, Decoder
import pandas as pd
from environs import Env

env = Env()
env.read_env()

DATA_PATH = env.str("DATA_PATH")

dataframe = pd.read_csv(DATA_PATH + 'categories_data.csv', engine='pyarrow')

category = dataframe['category']
title = dataframe['title']

encoder = Encoder(dataframe, 2000)

encoded_df = encoder.encode_tags()
encoded_df.to_csv(DATA_PATH + 'enc_dataset.csv', index=False)
print('File enc_dataset.csv was saved')

keys = encoder.keys_from_tags()
keys.to_csv(DATA_PATH + 'tags_keys.csv', index=False)
print('File tags_keys.csv was saved')

decoder = Decoder(encoded_df, keys)
decoded_df = decoder.decode_tags()
decoded_df.to_csv(DATA_PATH + 'dec_dataset.csv', index=False)
print('File dec_dataset.csv was saved')
