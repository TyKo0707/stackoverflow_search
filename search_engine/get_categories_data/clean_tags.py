import pandas as pd
from environs import Env

env = Env()
env.read_env()

DATA_PATH = env.str("DATA_PATH")


def code_from_key(keys_data, key):
    if key in keys_data.tag.values:
        return keys_data[keys_data.tag == key].code.values[0]


def delete_elem_from_tags(df, condition, tag_to_delete, mode='num'):
    ind = 30939
    if mode == 'num':
        cond_key = code_from_key(df_keys, condition)
        ttd_key = code_from_key(df_keys, tag_to_delete)
        for i in df.tags.values:
            if df.category[ind] == condition:
                l_buff = i[::]
                if cond_key in l_buff and ttd_key in l_buff:
                    del l_buff[l_buff.index(ttd_key)]
                    df.at[ind, "tags"] = l_buff
                ind += 1
    elif mode == 'str':
        for i in df.tags.values:
            if df.category[ind] == condition:
                l_buff = i[::]
                if condition in l_buff and tag_to_delete in l_buff:
                    del l_buff[l_buff.index(tag_to_delete)]
                    df.at[ind, "tags"] = l_buff
                ind += 1


df_dec = pd.read_csv(DATA_PATH + 'dec_dataset.csv', engine='pyarrow')[30939:30943]
df_enc = pd.read_csv(DATA_PATH + 'enc_dataset.csv', engine='pyarrow')[30939:30943]
df_keys = pd.read_csv(DATA_PATH + 'tags_keys.csv', engine='pyarrow')


def split_tags(string):
    if string:
        try:
            return [int(i) for i in string.split('|')]
        except:
            return [str(i) for i in string.split('|')]


df_dec.tags = df_dec.tags.apply(split_tags)
df_dec.dropna(inplace=True, axis=0)
df_enc.tags = df_enc.tags.apply(split_tags)
df_enc.dropna(inplace=True, axis=0)

delete_elem_from_tags(df_enc, 'asp.net', 'c#')
delete_elem_from_tags(df_dec, 'asp.net', 'c#', 'str')

print(2)
