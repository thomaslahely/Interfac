
import math
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict
### Énoncé de la fonction 1 : vectoriser_phrase (phrase, vocabulaire_direct,methode="tfidf", **params)

# **Titre de la fonction :**
# > vectoriser_phrase (phrase, vocabulaire_direct,methode="tfidf", **params) 

# **Consigne :**
# * de mots) en un vecteur numérique selon une méthode 
# * de pondération choisie parmi les méthodes vues à la séance 4 (par défaut : TF-IDF).
# * Cette fonction va aplatir toutes les phrases en une seule liste de mots 
#  * et fera appel à la fonction vectoriser_phrase (phrase,vocabulaire_direct, methode="tfidf", **params)
#  * pour calculer le vecteur.
# *     methode : "tf", "tfidf", "binaire"


def vectoriser_phrase(phrase, vocabulaire_direct: Dict[str, int], methode: str = "tfidf", **params) -> List[float]:
    # Initialiser le vecteur avec des zeros
    vecteur = [0.0] * len(vocabulaire_direct)
    
    # Tokeniser la phrase en mots si c'est une chaine
    if isinstance(phrase, str):
        mots = phrase.split()
    else:
        mots = phrase

    # Compter la fréquence des mots dans la phrase
    compteur_mots = Counter(mots)

    for mot, freq in compteur_mots.items():
        if mot in vocabulaire_direct:
            index = vocabulaire_direct[mot]
            if methode == "tf":
                vecteur[index] = freq
            elif methode == "binaire":
                vecteur[index] = 1
            elif methode == "tfidf":
                idf = params.get("idf", {}).get(mot, 1.0)
                vecteur[index] = freq * idf
            else:
                raise ValueError(f"méthode inconnue :{methode}")
    return vecteur


def vectoriser_document_mots(document:str , vocabulaire_direct: Dict[str, int], methode="tfidf", **params) -> List[float]:
    """
    Vectorise un document en utilisant une méthode de pondération (par défaut TF-IDF).

    document : liste de listes de mots
        Exemple : [["je", "mange"], ["une", "pomme"]]

    vocabulaire_direct : dict {mot: index}

    methode : str
        Méthode de pondération ("tf", "tfidf", "binaire", etc.)

    **params :
        Paramètres optionnels transmis à vectoriser_phrase
        (par exemple idf=..., normalisation=..., etc.)
    """

    # 1) Aplatir toutes les phrases du document → une seule liste de mots
    mots_document = [mot for phrase in document for mot in phrase]

    # 2) Appeler vectoriser_phrase comme demandé
    vecteur = vectoriser_phrase(
        mots_document,
        vocabulaire_direct,
        methode=methode,
        **params
    )

    return vecteur


def vectoriser_document_agrege(document, vocabulaire_direct, methode="tfidf", strategie="moyenne", **params):
    """
    Vectorise un document en agrégeant les vecteurs de ses phrases.
    
    document : liste de listes de mots
    strategie : "moyenne", "max", "somme"
    """
    vecteurs_phrases = []
    for phrase in document:
        # phrase est une liste de mots
        v = vectoriser_phrase(phrase, vocabulaire_direct, methode=methode, **params)
        vecteurs_phrases.append(v)
    
    if not vecteurs_phrases:
        return [0.0] * len(vocabulaire_direct)
        
    # Agrégation
    nb_dims = len(vocabulaire_direct)
    vecteur_agrege = [0.0] * nb_dims
    
    if strategie == "moyenne":
        for v in vecteurs_phrases:
            for i in range(nb_dims):
                vecteur_agrege[i] += v[i]
        vecteur_agrege = [x / len(vecteurs_phrases) for x in vecteur_agrege]
        
    elif strategie == "max":
        if not vecteurs_phrases:
            return vecteur_agrege
        vecteur_agrege = list(vecteurs_phrases[0])
        for v in vecteurs_phrases[1:]:
            for i in range(nb_dims):
                if v[i] > vecteur_agrege[i]:
                    vecteur_agrege[i] = v[i]
                    
    elif strategie == "somme":
        for v in vecteurs_phrases:
            for i in range(nb_dims):
                vecteur_agrege[i] += v[i]
                
    return vecteur_agrege
