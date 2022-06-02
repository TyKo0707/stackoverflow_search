import pickle
import warnings
import pandas as pd
from environs import Env
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from logger import get_logger

env = Env()
env.read_env()
logger = get_logger()

warnings.filterwarnings('ignore')

DATA_PATH = env.str("DATA_PATH")
MODELS = env.str("MODELS")


def split_tags(string):
    if string:
        return [int(i) for i in string.split('|')]


df_tags = pd.read_csv(DATA_PATH + 'enc_dataset.csv', engine='pyarrow').sample(frac=1)
df_tags.tags = df_tags.tags.apply(split_tags)
df_tags.dropna(inplace=True, axis=0)
tags = df_tags.tags

multilabel_binarizer = MultiLabelBinarizer()
y_bin = multilabel_binarizer.fit_transform(df_tags.tags)
pickle.dump(multilabel_binarizer, open(MODELS + 'mlb.pkl', 'wb'))

X_train_tags, X_test_tags, y_train_tags, y_test_tags = train_test_split(y_bin, df_tags['category'], test_size=0.2,
                                                                        random_state=0)

# region tags model
logreg_tags = LogisticRegression(n_jobs=1, C=1e5)
logreg_tags = logreg_tags.fit(X_train_tags, y_train_tags)
y_pred = logreg_tags.predict(X_test_tags)
pickle.dump(logreg_tags, open(MODELS + 'model_tags.pkl', 'wb'))
print('accuracy %s' % accuracy_score(y_pred, y_test_tags))

print(classification_report(y_test_tags, y_pred))
# endregion
