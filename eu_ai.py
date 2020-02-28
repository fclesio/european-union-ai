# This is a anti-pattern to disable warnings
# I'm using just for a simplification
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pyLDAvis.sklearn
import re
import seaborn as sns
import spacy
import string
from collections import Counter
from io import StringIO
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.en.stop_words import STOP_WORDS
from unicodedata import normalize
from wordcloud import WordCloud

# Local directory
ROOT_DIR = os.getcwd()

# NLP
nlp = spacy.load("en_core_web_sm")
stoplist = list(STOP_WORDS)
punctuations = string.punctuation


def load_text_file(filepath):
    f = open(filepath, "r")
    file = []
    for x in f:
        file.append(x)

    raw_text = " ".join([str(x) for x in file])
    df = pd.read_csv(StringIO(raw_text), delimiter="\n")
    df.columns = ["text"]

    return df


df = load_text_file(
    filepath="commission-white-paper-artificial-intelligence-feb2020_en.txt"
)

# Pre-Processing
special_by_space = re.compile('[/(){}\[\]"\|@,;]')


def clean_text(text):
    text = str(text)
    text = text.lower()
    text = text.replace("\n", " ")
    text = special_by_space.sub(" ", text)
    text = " ".join(word for word in text.split() if word not in stoplist)
    return text


def remove_punctuation(text):
    """
     This function remove the replacement_patterns from input string.
     Parameters
     ----------
     text : String
         Input string to the function.
     Returns
     -------
     text : String
         Output string after replacement.
     """
    rem = string.punctuation
    pattern = r"[{}]".format(rem)
    text = re.sub(r"[-()\"#/@;:&<>{}`+=~|.!?,[\]©_*]", " ", text)
    text = text.replace(pattern, "")
    return text


def replace_ptbr_char_by_word(word):
    word = str(word)
    word = normalize("NFKD", word).encode("ASCII", "ignore").decode("ASCII")
    return word


def remove_pt_br_char_by_text(text):
    text = str(text)
    text = " ".join(
        replace_ptbr_char_by_word(word) for word in text.split() if word not in stoplist
    )
    return text


def get_word_frequency(df):
    # Word Frequency per Category
    def cleanup_text(docs, logging=False):
        texts = []
        counter = 1
        for doc in docs:
            if counter % 1000 == 0 and logging:
                print("Processed %d out of %d documents." % (counter, len(docs)))
            counter += 1
            doc = nlp(doc, disable=["parser", "ner"])
            tokens = [str(tok).lower().strip() for tok in doc if tok.lemma_ != "-PRON-"]
            tokens = [
                tok for tok in tokens if tok not in stoplist and tok not in punctuations
            ]
            tokens = " ".join(tokens)
            texts.append(tokens)
        return pd.Series(texts)

    df_text = [str(text) for text in df["text"]]
    df_text_clean = cleanup_text(df_text)
    df_text_clean = " ".join(df_text_clean).split()
    df_text_clean_counts = Counter(df_text_clean)
    df_common_words = [word[0] for word in df_text_clean_counts.most_common(41)]
    df_common_counts = [word[1] for word in df_text_clean_counts.most_common(41)]
    df_common_words.pop(0)
    df_common_counts.pop(0)

    fig = plt.figure(figsize=(18, 6))
    sns.barplot(x=df_common_words, y=df_common_counts)
    plt.title(f"Most Common Words used in European Commission White Paper on AI")
    plt.xticks(rotation=45)
    plt.show()


def show_wordcloud(text):
    # Create and generate a word cloud image:
    wordcloud = WordCloud(stopwords=stoplist, background_color="white").generate(text)

    # Display the generated image:
    fig = plt.figure(figsize=(25, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(f"Word Cloud for European Commission White Paper on AI", fontsize=20)
    plt.axis("off")
    plt.show()


def get_wordcloud(df):
    # Get all texts and generate a cloud
    text = " ".join(str(review) for review in df.text)
    show_wordcloud(text)


def get_tfidf_df(df):
    # This one came from Analytics Vidhya
    # Ref: https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/

    # Generate the TF
    tf1 = (
        (df["text"][:])
        .apply(lambda x: pd.value_counts(x.split(" ")))
        .sum(axis=0)
        .reset_index()
    )

    tf1.columns = ["word", "tf"]

    # Remove some instances with NaN
    tf1 = tf1.dropna()
    df = df.dropna()

    # Calculate the log of the terms according to the TF
    for i, word in enumerate(tf1["word"]):
        tf1.loc[i, "idf"] = np.log(
            df.shape[0] / (len(df[df["text"].str.contains(word)]))
        )

    # Full calculation of TF-IDF
    tf1["tfidf"] = tf1["tf"] * tf1["idf"]
    return (
        tf1.head(300).sort_values(by=["tfidf"], ascending=False).reset_index(drop=True)
    )


def get_word_ngrams_list(df, word_ngram):
    def get_top_word_n_bigram(corpus, n=None):
        vec = CountVectorizer(ngram_range=(word_ngram, word_ngram)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [
            (word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()
        ]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]

    common_words = get_top_word_n_bigram(df["text"], 20)
    df3 = pd.DataFrame(common_words, columns=["ngram", "qty"])

    return df3


def get_topics(df, n_components, number_words):

    # Convert to list
    data = df.text.values.tolist()

    # Remove special characters
    data = [re.sub("\S*@\S*\s?", "", sent) for sent in data]

    # Remove new line characters
    data = [re.sub("\s+", " ", sent) for sent in data]

    # Remove distracting single quotes
    data = [re.sub("'", "", sent) for sent in data]

    vectorizer = CountVectorizer(analyzer="word", stop_words=stoplist, lowercase=True)

    data_vectorized = vectorizer.fit_transform(data)

    # Materialize the sparse data
    data_dense = data_vectorized.todense()

    # Compute Sparsicity = Percentage of Non-Zero cells
    print("Sparsicity: ", ((data_dense > 0).sum() / data_dense.size) * 100, "%")

    # Build LDA Model
    lda_model = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=10,
        learning_method="online",
        random_state=42,
        batch_size=10,
        evaluate_every=-1,
        n_jobs=-1,
    )
    lda_output = lda_model.fit_transform(data_vectorized)

    # Helper function
    def print_topics(model, count_vectorizer, n_top_words):
        words = count_vectorizer.get_feature_names()
        for topic_idx, topic in enumerate(model.components_):
            print("\nTopic #%d:" % topic_idx)
            print(
                " ".join([words[i] for i in topic.argsort()[: -n_top_words - 1 : -1]])
            )

    # Print the topics found by the LDA model
    print("Topics found via LDA:")
    lda_model.fit(data_vectorized)

    print_topics(lda_model, vectorizer, number_words)

    return lda_model, data_vectorized, data, lda_output, vectorizer


def get_lda_plot(lda_model, data_vectorized, vectorizer):
    pyLDAvis.enable_notebook()
    panel = pyLDAvis.sklearn.prepare(lda_model, data_vectorized, vectorizer, mds="tsne")
    return panel


df["text"] = df["text"].apply(
    lambda x: " ".join([word for word in x.split() if word not in (stoplist)])
)
df["text"] = df["text"].apply(remove_pt_br_char_by_text)
df["text"] = df["text"].apply(clean_text)
df["text"] = df["text"].str.replace("[^\w\s]", "")
df["text"] = df["text"].apply(remove_punctuation)
df["text"] = df["text"].str.strip()
df["text"] = df["text"].str.replace("\d+", "")

get_word_frequency(df)

get_wordcloud(df)

df_tfidf = get_tfidf_df(df)
df_tfidf.head(30)

get_word_ngrams_list(df, 2)

get_word_ngrams_list(df, 3)

lda_model, data_vectorized, data, lda_output, vectorizer = get_topics(
    df, n_components=7, number_words=7
)

get_lda_plot(lda_model, data_vectorized, vectorizer)
