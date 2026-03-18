import os
import re


# chargement données président

def load_pres(file_path):
    """
    Charge le dataset président (Chirac vs Mitterrand)

    Format ligne :
    <id:id:label> texte

    label :
    - M → Mitterrand → -1
    - sinon → Chirac → 1
    """

    alltxts = []
    alllabs = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(line.strip()) < 5:
                continue

            # extraire le label (M ou C)
            match = re.search(r"<[0-9]+:[0-9]+:(.)>", line)
            if match:
                label = match.group(1)
            else:
                continue

            # extraire le texte
            text = re.sub(r".*<[0-9]+:[0-9]+:.[ ]*(.*)", r"\1", line).strip()

            # assigner label
            if label == "M":
                alllabs.append(-1)
            else:
                alllabs.append(1)

            alltxts.append(text)

    return alltxts, alllabs


#chargement données movies

def load_movies(path):
    """
    Charge dataset movies1000

    Structure :
    - pos/ → positif (label = 1)
    - neg/ → négatif (label = 0)
    """

    alltxts = []
    alllabs = []

    #positif
    pos_path = os.path.join(path, "pos")

    for file in os.listdir(pos_path):
        file_path = os.path.join(pos_path, file)

        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                alltxts.append(text)
                alllabs.append(1)  # positif

    #négatif
    neg_path = os.path.join(path, "neg")

    for file in os.listdir(neg_path):
        file_path = os.path.join(neg_path, file)

        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                alltxts.append(text)
                alllabs.append(0)  # négatif

    return alltxts, alllabs