import pandas as pd
from environs import Env

env = Env()
env.read_env()

DATA_PATH = env.str("DATA_PATH")


def code_from_key(keys_data, key):
    if key in keys_data.tag.values:
        return keys_data[keys_data.tag == key].code.values[0]


def delete_elem_from_tags(df, condition, tag_to_delete, mode='num'):
    ind = 0

    def delete_tag(index, cond, ttd):
        for i in df.tags.values:
            if not i == '':
                if mode == 'num':
                    i = [int(j) for j in i.split('|')]
                elif mode == 'str':
                    i = [str(j) for j in i.split('|')]
                if df.category[index] == condition:
                    if cond in i and ttd in i:
                        l_buff = i[::]
                        del l_buff[l_buff.index(ttd)]
                        df.at[index, "tags"] = '|'.join([str(j) for j in l_buff])
            index += 1

    if mode == 'num':
        cond_key = code_from_key(df_keys, condition)
        ttd_key = code_from_key(df_keys, tag_to_delete)
        delete_tag(ind, cond_key, ttd_key)
    elif mode == 'str':
        delete_tag(ind, condition, tag_to_delete)


df_dec = pd.read_csv(DATA_PATH + 'dec_dataset.csv', engine='pyarrow')
df_enc = pd.read_csv(DATA_PATH + 'enc_dataset.csv', engine='pyarrow')
df_keys = pd.read_csv(DATA_PATH + 'tags_keys.csv', engine='pyarrow')

delete_elem_from_tags(df_enc, 'asp.net', 'c#')
delete_elem_from_tags(df_enc, '.net', 'c#')
delete_elem_from_tags(df_enc, 'c#', 'asp.net')
delete_elem_from_tags(df_enc, 'c#', '.net')
delete_elem_from_tags(df_enc, 'с++', 'с')
delete_elem_from_tags(df_enc, 'с', 'c++')
df_enc.dropna(inplace=True, axis=0)
df_enc.to_csv(DATA_PATH + 'enc_dataset.csv', index=False)
delete_elem_from_tags(df_dec, 'asp.net', 'c#', 'str')
delete_elem_from_tags(df_dec, 'asp.net', 'c#', 'str')
delete_elem_from_tags(df_dec, 'asp.net', 'c#', 'str')
delete_elem_from_tags(df_dec, 'asp.net', 'c#', 'str')
df_dec.dropna(inplace=True, axis=0)
df_dec.to_csv(DATA_PATH + 'dec_dataset.csv', index=False)
