from sklearn.linear_model import LogisticRegression


def train_logistic_regression(X_train, y_train, balanced=False):
    """
    Entraîne un modèle de Logistic Regression.

    Parameters
    ----------
    X_train : sparse matrix
        Données d'entraînement
    y_train : list
        Labels d'entraînement
    balanced : bool
        Si True → corrige le déséquilibre des classes

    Returns
    -------
    model : LogisticRegression
        Modèle entraîné
    """

    #création du modèle
    if balanced:
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
    else:
        model = LogisticRegression(max_iter=1000)

    #entraînement du modèle
    model.fit(X_train, y_train)

    return model
