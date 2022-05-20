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


class CategoryDataset:
    def __init__(self, search_text, search_size, c_type):
        self.search_text = search_text
        self.search_size = search_size
        self.idx = []
        self.c_type = c_type
        self.df = pd.DataFrame()
        self.df['article_index'] = None
        self.df['titles'] = None
        self.df['tags'] = None

    def get_ids(self):
        search_query = '+'.join(self.search_text.split(' '))
        for i in range(1, int((self.search_size / 50)) + 1):
            if self.c_type == 'tag':
                result_html = requests.get(
                    f'https://stackoverflow.com/questions/tagged/{self.search_text}?tab=votes&page={i}&pagesize=50') \
                    .content
            else:
                result_html = requests.get(
                    f'https://stackoverflow.com/search?page={i}&tab=Relevance&pagesize=50&q={search_query}').content
            soup = BeautifulSoup(result_html, "html.parser")
            ids = soup.find_all('div', class_='s-post-summary js-post-summary')
            if ids:
                for row in ids:
                    self.idx.append(int(row.attrs['data-post-id']))
            else:
                raise TypeError
        self.df['article_index'] = self.idx

    def filter_values(self):
        for i in range(int(self.search_size / 20)):
            qs = SITE.fetch('questions', ids=self.df.article_index.values[i * 20:(i + 1) * 20])

            for row in qs['items']:
                s = row['title']
                s = s.replace('&#39;', '')
                self.df.loc[self.df['article_index'] == row['question_id'], 'tags'] = '|'.join(row['tags'])
                self.df.loc[self.df['article_index'] == row['question_id'], 'titles'] = preprocess_text(s)

    def create_and_save_dataset(self):
        self.get_ids()
        self.filter_values()
        df = self.df.dropna()
        df.drop(columns='article_index', inplace=True)
        df.to_csv(f'{RAW_DATA_PATH}data_b_c/{"_".join(self.search_text.split())}.csv', index=False)
