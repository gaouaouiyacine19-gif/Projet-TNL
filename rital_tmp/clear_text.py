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
               do_stemming=True,
               mark_uppercase=False,
               keep_part=None):
    """
    Paramètres :
    - remove_accents : bool, supprime les accents
    - lowercase : bool, met le texte en minuscules
    - remove_punctuation : bool, supprime la ponctuation
    - remove_digits : bool, supprime les chiffres
    - remove_stopwords : bool, supprime les stopwords
    - do_stemming : bool, applique le stemming
    - mark_uppercase : bool, marque les mots entièrement en majuscules
    (ex : "USA" -> "USA_UPPER")
    - keep_part : str ou None, si "first" garde uniquement la première ligne du texte,
    si "last" garde uniquement la dernière ligne, sinon conserve tout le texte
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

    # garder une partie du texte
    if keep_part == "first":
        text = text.split("\n")[0]
    elif keep_part == "last":
        text = text.split("\n")[-1]

    # marquer les mots en majuscules
    if mark_uppercase:
        words_tmp = text.split()
        words_tmp = [w + "_UPPER" if w.isupper() else w for w in words_tmp]
        text = " ".join(words_tmp)


    return text