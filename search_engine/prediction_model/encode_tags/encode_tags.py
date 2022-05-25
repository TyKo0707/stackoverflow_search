import pandas as pd


class Encoder:

    def __init__(self, df, max_freq: int):
        self.df = df
        self.max_freq = max_freq

    def keys_from_tags(self):
        df = self.df.copy(deep=True)
        df.tags = [i.split('|') for i in df['tags'].to_list()]
        tags_dict = {}
        tags_freq = {}
        ind = 0
        for i in range(len(df.tags)):
            for j in range(len(df.tags[i])):
                if not df.tags[i][j] in list(tags_dict.keys()):
                    tags_dict[df.tags[i][j]] = ind
                    tags_freq[df.tags[i][j]] = 0
                    ind += 1
                else:
                    tags_freq[df.tags[i][j]] += 1

        tags_freq = dict(sorted(tags_freq.items(), key=lambda x: x[1], reverse=True))
        keys = list(tags_freq.keys())[:self.max_freq]
        values = [tags_dict[i] for i in keys]
        df_tags = pd.DataFrame()
        df_tags['tag'] = keys
        df_tags['code'] = values
        return df_tags

    def encode_tags(self):
        df = self.df.copy(deep=True)
        new_tags = []
        keys = self.keys_from_tags()
        for i in df.iterrows():
            t = []
            for j in i[1].tags.split('|'):
                try:
                    code = keys.loc[keys['tag'] == j].code.values[0]
                    t.append(code)
                except:
                    continue
            new_tags.append(t)
        df.tags = ['|'.join([str(j) for j in i]) for i in new_tags]

        return df


class Decoder:
    def __init__(self, df, keys):
        self.df = df
        self.keys = keys

    def decode_tags(self):
        df = self.df.copy(deep=True)
        new_tags = []
        for i in df.iterrows():
            t = []
            for j in i[1].tags.split('|'):
                try:
                    tag = self.keys.loc[self.keys['code'] == int(j)].tag.values[0]
                    t.append(tag)
                except:
                    continue
            new_tags.append(t)
        df.tags = ['|'.join([str(j) for j in i]) for i in new_tags]

        return df
