import requests
import pandas as pd
from bs4 import BeautifulSoup
from search_engine.processing_data.normalize_functions import preprocess_text
from stackapi import StackAPI
from environs import Env

env = Env()
env.read_env()
RAW_DATA_PATH = env.str("RAW_DATA_PATH")
SITE = StackAPI(name='stackoverflow')
categories_dict = {'c%23': 'c#', 'c%2b%2b': 'c++'}


class CategoryDataset:
    def __init__(self, categories):
        self.df = pd.DataFrame()
        self.df['article_index'] = None
        self.df['title'] = None
        self.df['tags'] = None
        self.df['category'] = None
        self.categories = categories

    def get_ids(self, search_text, c_type, search_size):
        search_query = '+'.join(search_text.split(' '))
        id_list = []
        categories_list = []
        for i in range(1, int((search_size / 50)) + 1):
            if c_type == 'tag':
                result_html = requests.get(
                    f'https://stackoverflow.com/questions/tagged/{search_text}?tab=votes&page={i}&pagesize=50') \
                    .content
            else:
                result_html = requests.get(
                    f'https://stackoverflow.com/search?page={i}&tab=Relevance&pagesize=50&q={search_query}').content
            soup = BeautifulSoup(result_html, "html.parser")
            ids = soup.find_all('div', class_='s-post-summary js-post-summary')
            category = categories_dict[search_text] if search_text in categories_dict.keys() else search_text
            if ids:
                for row in ids:
                    id_list.append(int(row.attrs['data-post-id']))
                    categories_list.append(category)
            else:
                raise TimeoutError
        self.df = pd.concat([self.df, pd.DataFrame(pd.DataFrame.from_dict(
            {"article_index": id_list, "title": None, "tags": None, "category": categories_list}))], ignore_index=True)
        return self.df.shape[0] - len(id_list)

    def filter_values(self, search_size, start_index):
        for i in range(int(search_size / 20)):
            try:
                qs = SITE.fetch('questions',
                                ids=self.df.article_index.values[start_index + i * 20: start_index + (i + 1) * 20])
            except:
                input("Change location in VPN (then enter ok): ")
                qs = SITE.fetch('questions',
                                ids=self.df.article_index.values[start_index + i * 20: start_index + (i + 1) * 20])
            for row in qs['items']:
                s = row['title']
                s = s.replace('&#39;', '')
                self.df.loc[self.df['article_index'] == row['question_id'], 'tags'] = '|'.join(row['tags'])
                self.df.loc[self.df['article_index'] == row['question_id'], 'title'] = preprocess_text(s)

    def create_and_save_dataset(self):
        i = 0
        while i < len(self.categories):
            c_type = 'tag' if ' ' not in self.categories[i] else 'query'
            size = 2500 if c_type == 'tag' else 500
            try:
                start_index = self.get_ids(self.categories[i], c_type, size)
                self.filter_values(size, start_index)
                self.df.to_csv(f'{RAW_DATA_PATH}data_b_c/categories_data.csv', index=False)
                print(self.df.tail(10))
            except TimeoutError:
                print('Please enter captcha for ', self.categories[i])
                input('Input "ok": ')
                continue
            i += 1
        df = self.df.dropna()
        df.drop(columns='article_index', inplace=True)
        df.to_csv(f'{RAW_DATA_PATH}data_b_c/categories_data.csv', index=False)
