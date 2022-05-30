import pickle
import pandas as pd
from environs import Env

env = Env()
env.read_env()
DATA_PATH = env.str("DATA_PATH")
FINAL_DATA = env.str("FINAL_DATA")
MODELS = env.str("MODELS")

df_keys = pd.read_csv(DATA_PATH + 'tags_keys.csv', engine='pyarrow')
main_data = pd.read_parquet(DATA_PATH + 'final_data.gzip', engine='pyarrow')


def encode_tags(list_of_tags: list, keys: pd.DataFrame):
    new_l = []
    for k in list_of_tags:
        if k in keys.tag.values:
            new_l.append(keys[keys.tag == k].code.values[0])

    return new_l


model = pickle.load(open(MODELS + 'model_tags.pkl', 'rb'))
mlb = pickle.load(open(MODELS + 'mlb.pkl', 'rb'))


def predict_category(list_of_tags):
    t = encode_tags(list_of_tags, df_keys)
    tags = mlb.transform([t, []])
    full_res = model.predict_proba(tags)[0]
    res = model.predict(tags)[0]
    return list(full_res), res


def get_df(data):
    dict_of_df = {}
    for i in data.iterrows():
        spl_tags = i[1].tags.split('|')
        full, category = predict_category(spl_tags)
        if max(full) > 0.95:
            try:
                dict_of_df[category] = pd.concat([dict_of_df[category], i[1].to_frame().T])
            except:
                dict_of_df[category] = i[1].to_frame().T

        else:
            for j in [model.classes_[full.index(c)] for c in sorted(full)[-3:]]:
                try:
                    dict_of_df[j] = pd.concat([dict_of_df[j], i[1].to_frame().T])
                except:
                    dict_of_df[j] = i[1].to_frame().T

    return dict_of_df


def save_dataset(key_cat, dataframe, header=False, mode='a'):
    if "html" in key_cat:
        key_cat = 'html_css'
    dataframe.to_csv(f'{DATA_PATH}dbc/{key_cat}.csv', index=False, mode=mode, header=header)


if __name__ == '__main__':
    for i in range((len(main_data) // 10000) + 1):
        df = main_data[i * 10000: (i + 1) * 10000]
        df_dict = get_df(df)
        for key, df in df_dict.items():
            if i == 0:
                save_dataset(key, df, True, 'w')
            else:
                save_dataset(key, df, False, 'a')
        print(f'From {i * 10000} to {(i + 1) * 10000} was processed!')
