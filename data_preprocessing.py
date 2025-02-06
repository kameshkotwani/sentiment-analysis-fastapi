# data preprocessing

import numpy as np
import re
import nltk
import string

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer




nltk.download('wordnet')
nltk.download('stopwords')


def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)


def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)


def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text


def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)


def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text


def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan


def normalize_text(user_input):
    """Normalize the text data."""
    try:
        df = pd.DataFrame({"content":[user_input]})

        df['content'] = df['content'].apply(lower_case) \
        .apply(remove_stop_words)\
        .apply(removing_numbers)\
        .apply(removing_punctuations)\
        .apply(removing_urls)\
        .apply(lemmatization)
        return df
    except Exception as e:
        print(e)


print(normalize_text("This is a stopword hi hello and testing a test to check if the fucntion is working or not!!!@@@@."))
