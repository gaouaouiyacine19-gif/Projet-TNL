from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def vectorize_and_split(texts,
                        labels,
                        test_size=0.2,
                        random_state=42,
                        ngram_range=(1, 1),
                        max_features=None):
    """
    Transforme les textes en vecteurs TF-IDF
    puis sépare les données en train / test.

    Parameters
    ----------
    texts : list[str]
        Liste des textes nettoyés
    labels : list[int]
        Liste des labels
    test_size : float
        Proportion du jeu de test
    random_state : int
        Pour rendre le split reproductible
    ngram_range : tuple
        Taille des n-grammes, ex: (1,1) ou (1,2)
    max_features : int or None
        Nombre maximum de features TF-IDF

    Returns
    -------
    vectorizer : TfidfVectorizer
        Le vectorizer appris
    X_train : sparse matrix
        Données d'entraînement vectorisées
    X_test : sparse matrix
        Données de test vectorisées
    y_train : list
        Labels d'entraînement
    y_test : list
        Labels de test
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