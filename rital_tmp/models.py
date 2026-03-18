from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# EREGRESSION LOGISTIC
def train_logistic_regression(X_train, y_train, balanced=False):
    """
    Régression logistique : modèle linéaire de classification.
    Peut gérer les classes déséquilibrées.
    """
    #création du modèle
    if balanced:
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
    else:
        model = LogisticRegression(max_iter=1000)

    #entraînement du modèle
    model.fit(X_train, y_train)

    return model

# NAIVE BAYES
def train_naive_bayes(X_train, y_train):
    """
    Naive Bayes : modèle probabiliste adapté au texte (TF-IDF).
    Suppose l'indépendance des mots.
    """
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

# SVM
def train_svm(X_train, y_train):
    """
    SVM : modèle puissant pour classification texte.
    Cherche la meilleure frontière entre les classes.
    """
    model = LinearSVC()
    model.fit(X_train, y_train)
    return model