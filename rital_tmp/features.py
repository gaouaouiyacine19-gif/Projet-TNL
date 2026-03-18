from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

def vectorize_and_split(texts,
                        labels,
                        test_size=0.2,
                        random_state=42,
                        ngram_range=(1, 1),
                        max_features=None):
    """
    Transforme les textes en vecteurs TF-IDF
    puis sépare les données en train / test.
    """
    #split train / test
    txt_train, txt_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    #vectorization TF-IDF
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features
    )

    #apprentissage du vocabulaire et vectorisation
    X_train = vectorizer.fit_transform(txt_train)
    X_test = vectorizer.transform(txt_test)

    return vectorizer, X_train, X_test, y_train, y_test


# Taille du vocabulaire
def get_vocab_size(vectorizer):
    """
    Retourne la taille du vocabulaire.
    """
    return len(vectorizer.vocabulary_)


# Top mots fréquents
def get_top_words(vectorizer, X, n=20):
    """
    Retourne les mots les plus fréquents
    """
    sums = np.array(X.sum(axis=0)).flatten()
    words = vectorizer.get_feature_names_out()

    top_idx = np.argsort(sums)[::-1][:n]
    return [(words[i], sums[i]) for i in top_idx]

########VECTORISERS PROPERS#########

#BoW classique
def vectorize_bow(texts):
    """
    compte le nombre d'occurrences des mots.
    """
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

#BoW binaire
def vectorize_bow_binary(texts):
    """
    indique seulement la présence ou absence des mots.
    """
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

#TF-IDF
def vectorize_tfidf(texts):
    """
    pondère les mots selon leur importance dans le corpus.
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

#n-grams
def vectorize_ngrams(texts):
    """
    prend en compte les suites de mots (ex: bigrammes).
    """
    vectorizer = CountVectorizer(ngram_range=(1,2))
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


#réduction vocabulaire
def vectorize_limited(texts):
    """
    Réduction du vocabulaire avec filtrage des mots rares/fréquents.
    """
    vectorizer = CountVectorizer(max_features=1000, min_df=2, max_df=0.9)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer