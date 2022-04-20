import pandas as pd
import spacy
import warnings

EN = spacy.load('en_core_web_sm')
warnings.filterwarnings('ignore')

preprocessed_data = pd.read_csv('../../final_data.csv')
print(preprocessed_data.shape)
preprocessed_data.head()

preprocessed_data.tags = preprocessed_data.tags.apply(lambda x: x.split('|'))  # Making the list of tags
tag_freq_dict = {}
for tags in preprocessed_data.tags:
    for tag in tags:
        if tag not in tag_freq_dict:
            tag_freq_dict[tag] = 0
        else:
            tag_freq_dict[tag] += 1

# Get most common tags
tags_to_use = 50
tag_freq_dict_sorted = dict(sorted(tag_freq_dict.items(), key=lambda x: x[1], reverse=True))
final_tags = list(tag_freq_dict_sorted.keys())[:tags_to_use]
len(final_tags)

# Change tag data to only for final_tags
final_tag_data = []
for tags in preprocessed_data.tags:
    temp = []
    for tag in tags:
        if tag in final_tags:
            temp.append(tag)
    final_tag_data.append(temp)
