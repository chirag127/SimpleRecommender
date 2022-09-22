import re
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_desc(s):
    s = str(s)
    s = s.lower()
    s = re.sub(r'[^a-zA-Z]', ' ', s)
    s = word_tokenize(s)
    p = [word for word in s if word not in set(stopwords.words('english'))]
    return ' '.join(p)


def clean_struct_table_implicit(dataset, descCol, itemCol):
    tfidf = TfidfVectorizer(min_df=2, max_df=0.7)
    tfidf_vectors = tfidf.fit_transform(dataset[descCol])
    tfidf_df = pd.DataFrame(tfidf_vectors.toarray(), columns=tfidf.get_feature_names_out(), index=dataset[itemCol])

    return tfidf_df, tfidf
