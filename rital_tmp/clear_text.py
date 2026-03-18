import re
import nltk
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Initialisation du stemmer
stemmer = PorterStemmer()

# Stopwords français + anglais
stop_words = set(stopwords.words('english') + stopwords.words('french'))


def clean_text(text,
               remove_accents=True,
               lowercase=True,
               remove_punctuation=True,
               remove_digits=True,
               remove_stopwords=True,
               do_stemming=True):
    """
    Nettoie un texte brut.

    Étapes :
    - suppression des accents
    - mise en minuscules
    - suppression ponctuation
    - suppression chiffres
    - suppression stopwords
    - stemming

    Parameters
    ----------
    text : str
        Texte à nettoyer

    Returns
    -------
    text : str
        Texte nettoyé
    """

   
    # supression accents 
    if remove_accents:
        text = unicodedata.normalize('NFD', text)\
                          .encode('ascii', 'ignore')\
                          .decode('utf-8')

    #minuscule
    if lowercase:
        text = text.lower()

    # suppression ponctuation
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', ' ', text)

    # suppression chiffres
    if remove_digits:
        text = re.sub(r'\d+', ' ', text)

    # tokenization
    words = text.split()

    # suppression stopwords
    if remove_stopwords:
        words = [w for w in words if w not in stop_words]

    # stemming
    if do_stemming:
        words = [stemmer.stem(w) for w in words]

    # reconstruction du texte
    text = " ".join(words)

    return text