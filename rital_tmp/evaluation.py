from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(model, X_test, y_test):
    """
    Évalue un modèle avec plusieurs métriques.

    Returns
    -------
    dict contenant :
        accuracy, precision, recall, f1
    """

    ##prediction
    y_pred = model.predict(X_test)

    ##Metriques
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Affichage des résultats
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")

    # Retour des métriques dans un dict
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }