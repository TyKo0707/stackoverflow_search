import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from textblob import TextBlob
from preprocess_data import de_duplicated_data

title_list = []
content_list = []
url_list = []
comment_list = []
sentiment_polarity_list = []
sentiment_subjectivity_list = []
vote_list = []
tag_list = []
corpus_list = []


def lxml_to_text(body):
    content = body
    soup = BeautifulSoup(content, 'lxml')
    if soup.code:
        soup.code.decompose()  # Remove the code section
    tag_p = soup.p
    tag_pre = soup.pre
    text = ''
    if tag_p:
        text = text + tag_p.get_text()
    if tag_pre:
        text = text + tag_pre.get_text()

    return text


def content_to_tokens(dataframe):
    for i, row in tqdm(dataframe.iterrows()):
        title_list.append(row.title)  # Get question title
        tag_list.append(row.tags)  # Get question tags

        # Questions
        text_body = lxml_to_text(row.body)

        content_list.append(
            str(row.title) + ' ' + str(text_body))  # Append title and question body data to the updated question body

        url_list.append('https://stackoverflow.com/questions/' + str(row.id))

        # Answers
        text_answers = lxml_to_text(row.combined_answers)
        comment_list.append(text_answers)

        vote_list.append(row.combined_score)  # Append votes

        corpus_list.append(
            content_list[-1] + ' ' + comment_list[-1])  # Combine the updated body and answers to make the corpus

        sentiment = TextBlob(row.combined_answers).sentiment
        sentiment_polarity_list.append(sentiment.polarity)
        sentiment_subjectivity_list.append(sentiment.subjectivity)

    content_token_df = pd.DataFrame({'original_title': title_list, 'post_corpus': corpus_list,
                                     'question_content': content_list, 'question_url': url_list,
                                     'tags': tag_list, 'overall_scores': vote_list,
                                     'answers_content': comment_list,
                                     'sentiment_polarity': sentiment_polarity_list,
                                     'sentiment_subjectivity': sentiment_subjectivity_list})

    return content_token_df
