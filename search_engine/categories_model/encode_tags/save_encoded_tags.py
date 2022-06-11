from encode_tags import Encoder, Decoder
import pandas as pd
from environs import Env
from logger import get_logger

env = Env()
env.read_env()
logger = get_logger(handle_errors=False)

DATA_PATH = env.str("DATA_PATH")

dataframe = pd.read_csv(DATA_PATH + 'categories_data.csv', engine='pyarrow')

encoder = Encoder(dataframe, 2000)

encoded_df = encoder.encode_tags()
encoded_df.to_csv(DATA_PATH + 'enc_dataset.csv', index=False)
logger.info('File enc_dataset.csv was saved')

df_tags_keys = encoder.keys_from_tags()
df_tags_keys.to_csv(DATA_PATH + 'tags_keys.csv', index=False)
logger.info('File tags_keys.csv was saved')

decoder = Decoder(encoded_df, df_tags_keys)
decoded_df = decoder.decode_tags()
decoded_df.to_csv(DATA_PATH + 'dec_dataset.csv', index=False)
logger.info('File dec_dataset.csv was saved')
