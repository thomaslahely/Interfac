from gensim.models import Word2Vec, FastText
import numpy as np
from typing import List, Union, Optional
import numpy as np
from gensim.models import KeyedVectors, Word2Vec, FastText
from typing import List, Union, Optional
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from typing import List, Union, Optional
import numpy as np
from typing import List, Union
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def entrainer_modele_word2vec(
    corpus: List[List[str]],
    taille_vecteur: int = 100,
    fenetre: int = 5,
    min_count: int = 1,
    epochs: int = 10,
    sg: int = 1
) -> Word2Vec:
    """
    Entraîne un modèle Word2Vec sur un corpus tokenisé.
    
    Args:
        corpus : liste de phrases, chaque phrase est une liste de tokens
        taille_vecteur : dimension des vecteurs
        fenetre : taille de la fenêtre de contexte
        min_count : fréquence minimale des mots
        epochs : nombre d'itérations
        sg : 1 pour skip-gram, 0 pour CBOW
        
    Returns:
        modèle Word2Vec entraîné
    """
    modele = Word2Vec(
        sentences=corpus,
        vector_size=taille_vecteur,
        window=fenetre,
        min_count=min_count,
        sg=sg
    )
    modele.train(corpus, total_examples=len(corpus), epochs=epochs)
    return modele

def entrainer_modele_fasttext(
    corpus: List[List[str]],
    taille_vecteur: int = 100,
    fenetre: int = 5,
    min_count: int = 1,
    epochs: int = 10,
    sg: int = 1
) -> FastText:
    """
    Entraîne un modèle FastText sur un corpus tokenisé.
    FastText permet de générer des vecteurs pour des mots hors vocabulaire (OOV).
    """
    modele = FastText(
        sentences=corpus,
        vector_size=taille_vecteur,
        window=fenetre,
        min_count=min_count,
        sg=sg
    )
    modele.train(corpus, total_examples=len(corpus), epochs=epochs)
    return modele

def vectoriser_embeddings(
    tokens: List[str],
    modele_embeddings: Union[Word2Vec, FastText],
    strategie_agregation: str = "moyenne",
    strategie_oov: str = "ignore"
) -> Optional[np.ndarray]:
    """
    Vectorise une liste de tokens en un vecteur embeddings.
    
    Args:
        tokens : liste de mots à vectoriser
        modele_embeddings : modèle Word2Vec ou FastText
        strategie_agregation : "moyenne" (default), "somme", ou "moyenne_ponderee"
        strategie_oov : "ignore" (ignore les mots OOV), "zero" (vecteur nul), "fasttext" (si modèle FT), "error"
        
    Returns:
        vecteur numpy représentant la phrase ou document, ou None si tous les mots sont ignorés
    """
    vecteurs = []
    for token in tokens:
        if token in modele_embeddings.wv:
            vecteurs.append(modele_embeddings.wv[token])
        else:
            if strategie_oov == "ignore":
                continue
            elif strategie_oov == "zero":
                vecteurs.append(np.zeros(modele_embeddings.vector_size))
            elif strategie_oov == "fasttext" and isinstance(modele_embeddings, FastText):
                vecteurs.append(modele_embeddings.wv[token])
            elif strategie_oov == "error":
                raise ValueError(f"Mot hors vocabulaire : {token}")
            else:
                continue  # par défaut on ignore

    if not vecteurs:
        return None if strategie_oov == "ignore" else np.zeros(modele_embeddings.vector_size)
    
    vecteurs = np.array(vecteurs)
    
    if strategie_agregation == "moyenne":
        return np.mean(vecteurs, axis=0)
    elif strategie_agregation == "somme":
        return np.sum(vecteurs, axis=0)
    else:
        # Par défaut on renvoie la moyenne
        return np.mean(vecteurs, axis=0)

# --------------------------
# Chargement d'un modèle pré-entraîné
# --------------------------
def charger_modele_preentraine(chemin: str, type_modele: str = 'word2vec') -> Union[KeyedVectors, FastText, Word2Vec]:
    """
    Charge un modèle de plongements pré-entraîné.
    
    Args:
        chemin : chemin vers le fichier du modèle
        type_modele : 'word2vec' ou 'fasttext'
    
    Returns:
        objet modele_embeddings
    """
    if type_modele.lower() == 'word2vec':
        modele_embeddings = KeyedVectors.load_word2vec_format(chemin, binary=True)
    elif type_modele.lower() == 'fasttext':
        modele_embeddings = FastText.load(chemin)
    else:
        raise ValueError("type_modele doit être 'word2vec' ou 'fasttext'")
    
    return modele_embeddings

# --------------------------
# Récupération du vecteur d'un mot
# --------------------------
def get_vecteur_mot(token: str, modele_embeddings, strategie_oov: str = 'ignore') -> Optional[np.ndarray]:
    """
    Retourne le vecteur d'un mot selon le modèle embeddings et la stratégie OOV.
    
    Args:
        token : mot à vectoriser
        modele_embeddings : Word2Vec, FastText ou KeyedVectors
        strategie_oov : 'ignore', 'zero', 'fasttext', 'error'
    
    Returns:
        np.ndarray du vecteur du mot, ou None si ignoré
    """
    vector_size = modele_embeddings.vector_size
    
    if token in modele_embeddings.wv:
        return modele_embeddings.wv[token]
    else:
        if strategie_oov == 'ignore':
            return None
        elif strategie_oov == 'zero':
            return np.zeros(vector_size)
        elif strategie_oov == 'fasttext' and isinstance(modele_embeddings, FastText):
            return modele_embeddings.wv[token]  # FastText construit à partir des sous-mots
        elif strategie_oov == 'error':
            raise ValueError(f"Mot hors vocabulaire : {token}")
        else:
            return None

# --------------------------
# Plongement de phrase à partir des vecteurs de mots
# --------------------------
def plongement_phrase_par_mots(
    phrase: List[str],
    modele_embeddings,
    strategie_agregation: str = 'moyenne',
    strategie_oov: str = 'ignore'
) -> np.ndarray:
    """
    Agrège les vecteurs de mots pour former un vecteur de phrase.
    
    Args:
        phrase : liste de tokens
        modele_embeddings : modèle embeddings
        strategie_agregation : 'moyenne', 'somme'
        strategie_oov : gestion des mots OOV
    
    Returns:
        np.ndarray du vecteur de la phrase
    """
    vecteurs = []
    for token in phrase:
        v = get_vecteur_mot(token, modele_embeddings, strategie_oov)
        if v is not None:
            vecteurs.append(v)
    
    if not vecteurs:
        return np.zeros(modele_embeddings.vector_size)  # Tous OOV

    vecteurs = np.array(vecteurs)
    
    if strategie_agregation == 'moyenne':
        return np.mean(vecteurs, axis=0)
    elif strategie_agregation == 'somme':
        return np.sum(vecteurs, axis=0)
    else:
        return np.mean(vecteurs, axis=0)

# --------------------------
# Similarité entre deux phrases
# --------------------------
def similarite_phrases(
    phrase1: List[str],
    phrase2: List[str],
    modele_embeddings,
    strategie_agregation: str = 'moyenne',
    strategie_oov: str = 'ignore',
    mesure: str = 'cosinus'
) -> float:
    """
    Calcule la similarité entre deux phrases.
    
    Args:
        phrase1, phrase2 : listes de tokens
        modele_embeddings : modèle embeddings
        strategie_agregation : méthode d'agrégation
        strategie_oov : gestion des OOV
        mesure : 'cosinus' (pour l'instant uniquement)
    
    Returns:
        similarité float entre 0 et 1
    """
    v1 = plongement_phrase_par_mots(phrase1, modele_embeddings, strategie_agregation, strategie_oov).reshape(1, -1)
    v2 = plongement_phrase_par_mots(phrase2, modele_embeddings, strategie_agregation, strategie_oov).reshape(1, -1)
    
    if mesure == 'cosinus':
        sim = cosine_similarity(v1, v2)[0][0]
        return float(sim)
    else:
        raise NotImplementedError("Seule la mesure 'cosinus' est implémentée")

# --------------------------
# Top-k phrases les plus similaires
# --------------------------
def top_k_phrases_similaires(
    phrase: List[str],
    corpus_phrases: List[List[str]],
    modele_embeddings,
    strategie_agregation: str = 'moyenne',
    strategie_oov: str = 'ignore',
    k: int = 5
) -> List[tuple]:
    """
    Retourne les k phrases les plus proches sémantiquement dans un corpus.
    
    Args:
        phrase : liste de tokens
        corpus_phrases : liste de phrases (liste de tokens)
        modele_embeddings : modèle embeddings
        k : nombre de phrases retournées
    
    Returns:
        liste de tuples (indice_phrase, similarité)
    """
    sim_list = []
    for idx, p in enumerate(corpus_phrases):
        sim = similarite_phrases(phrase, p, modele_embeddings, strategie_agregation, strategie_oov)
        sim_list.append((idx, sim))
    
    # Tri décroissant par similarité
    sim_list.sort(key=lambda x: x[1], reverse=True)
    
    return sim_list[:k]

# --------------------------
# Plongement de document par mots
# --------------------------
def plongement_document_par_mots(
    document: List[str],
    modele_embeddings,
    strategie_agregation: str = 'moyenne',
    strategie_oov: str = 'ignore'
) -> np.ndarray:
    """
    Agrège les vecteurs de mots pour obtenir un vecteur document.
    
    Args:
        document : liste de tokens du document
        modele_embeddings : modèle Word2Vec / FastText
        strategie_agregation : 'moyenne' ou 'somme'
        strategie_oov : gestion des OOV
    
    Returns:
        np.ndarray du vecteur document
    """
    from copy import deepcopy
    return plongement_phrase_par_mots(document, modele_embeddings, strategie_agregation, strategie_oov)

# --------------------------
# Plongement de document par phrases
# --------------------------
def plongement_document_par_phrases(
    document: List[List[str]],
    modele_embeddings,
    strategie_agregation_phrase: str = 'moyenne',
    strategie_agregation_doc: str = 'moyenne',
    strategie_oov: str = 'ignore'
) -> np.ndarray:
    """
    Agrège les vecteurs de phrases pour obtenir un vecteur document.
    
    Args:
        document : liste de phrases, chaque phrase est une liste de tokens
        modele_embeddings : modèle Word2Vec / FastText
        strategie_agregation_phrase : agrégation au niveau phrase
        strategie_agregation_doc : agrégation des vecteurs de phrases
        strategie_oov : gestion des OOV
    
    Returns:
        np.ndarray du vecteur document
    """
    vecteurs_phrases = []
    for phrase in document:
        v_phrase = plongement_phrase_par_mots(phrase, modele_embeddings, strategie_agregation_phrase, strategie_oov)
        if v_phrase is not None:
            vecteurs_phrases.append(v_phrase)
    
    if not vecteurs_phrases:
        return np.zeros(modele_embeddings.vector_size)
    
    vecteurs_phrases = np.array(vecteurs_phrases)
    
    if strategie_agregation_doc == 'moyenne':
        return np.mean(vecteurs_phrases, axis=0)
    elif strategie_agregation_doc == 'somme':
        return np.sum(vecteurs_phrases, axis=0)
    else:
        return np.mean(vecteurs_phrases, axis=0)

# --------------------------
# Entraînement Doc2Vec
# --------------------------
def entrainer_modele_doc2vec(
    corpus: List[List[str]],
    taille_vecteur: int = 100,
    fenetre: int = 5,
    min_count: int = 1,
    epochs: int = 10,
    dm: int = 1
) -> Doc2Vec:
    """
    Entraîne un modèle Doc2Vec sur un corpus.
    
    Args:
        corpus : liste de documents (liste de tokens)
        taille_vecteur : dimension des vecteurs
        fenetre : fenêtre de contexte
        min_count : fréquence minimale
        epochs : nombre d'itérations
        dm : 1 pour PV-DM, 0 pour PV-DBOW
    
    Returns:
        modele_doc2vec entraîné
    """
    tagged_docs = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(corpus)]
    
    modele_doc2vec = Doc2Vec(vector_size=taille_vecteur,
                             window=fenetre,
                             min_count=min_count,
                             epochs=epochs,
                             dm=dm)
    modele_doc2vec.build_vocab(tagged_docs)
    modele_doc2vec.train(tagged_docs, total_examples=len(tagged_docs), epochs=epochs)
    return modele_doc2vec

# --------------------------
# Inférence du vecteur document Doc2Vec
# --------------------------
def plongement_document_doc2vec(document: List[str], modele_doc2vec: Doc2Vec) -> np.ndarray:
    """
    Infère le vecteur d'un document ou requête à partir d'un modèle Doc2Vec.
    
    Args:
        document : liste de tokens
        modele_doc2vec : modèle Doc2Vec entraîné
    
    Returns:
        np.ndarray du vecteur document
    """
    return modele_doc2vec.infer_vector(document)

# --------------------------
# Similarité entre deux documents
# --------------------------
def similarite_documents(
    doc1: Union[List[str], np.ndarray],
    doc2: Union[List[str], np.ndarray],
    modele_embeddings=None,
    methode: str = 'embeddings',  # 'embeddings' ou 'doc2vec' ou 'vecteurs_precalcules'
    strategie_agregation: str = 'moyenne',
    strategie_oov: str = 'ignore',
    mesure: str = 'cosinus'
) -> float:
    """
    Calcule la similarité entre deux documents.
    
    Args:
        doc1, doc2 : documents sous forme de liste de tokens ou vecteurs pré-calculés
        modele_embeddings : Word2Vec, FastText ou Doc2Vec
        methode : 'embeddings' (agg mots/phrases), 'doc2vec', 'vecteurs_precalcules'
        strategie_agregation : agrégation pour embeddings
        strategie_oov : gestion OOV
        mesure : 'cosinus' (autres non implémentées)
    
    Returns:
        float similarité
    """
    # Cas vecteurs pré-calculés
    if methode == 'vecteurs_precalcules':
        v1 = np.array(doc1).reshape(1, -1)
        v2 = np.array(doc2).reshape(1, -1)
    elif methode == 'doc2vec':
        v1 = modele_embeddings.infer_vector(doc1).reshape(1, -1)
        v2 = modele_embeddings.infer_vector(doc2).reshape(1, -1)
    elif methode == 'embeddings':
        v1 = plongement_document_par_mots(doc1, modele_embeddings, strategie_agregation, strategie_oov).reshape(1, -1)
        v2 = plongement_document_par_mots(doc2, modele_embeddings, strategie_agregation, strategie_oov).reshape(1, -1)
    else:
        raise ValueError("methode doit être 'embeddings', 'doc2vec' ou 'vecteurs_precalcules'")
    
    if mesure == 'cosinus':
        sim = cosine_similarity(v1, v2)[0][0]
        return float(sim)
    else:
        raise NotImplementedError("Seule la mesure 'cosinus' est implémentée")

# --------------------------
# Top-k documents les plus similaires
# --------------------------
def top_k_documents_similaires(
    document: List[str],
    corpus: List[List[str]],
    modele_embeddings=None,
    methode: str = 'embeddings',
    k: int = 5,
    strategie_agregation: str = 'moyenne',
    strategie_oov: str = 'ignore'
) -> List[tuple]:
    """
    Retourne les k documents les plus proches sémantiquement dans un corpus.
    
    Returns:
        Liste de tuples (indice_document, similarité)
    """
    sim_list = []
    for idx, doc in enumerate(corpus):
        sim = similarite_documents(document, doc, modele_embeddings, methode, strategie_agregation, strategie_oov)
        sim_list.append((idx, sim))
    
    sim_list.sort(key=lambda x: x[1], reverse=True)
    return sim_list[:k]

# --------------------------
# Nuage de mots à partir d'un texte brut
# --------------------------
def generer_nuage_mots_texte(
    texte: str,
    stopwords: set = None,
    largeur: int = 600,
    hauteur: int = 300,
    couleur_fond: str = 'white'
):
    """
    Génère et affiche un nuage de mots à partir d'un texte.
    """
    wc = WordCloud(width=largeur, height=hauteur, background_color=couleur_fond, stopwords=stopwords)
    wc.generate(texte)
    plt.figure(figsize=(largeur/100, hauteur/100))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# --------------------------
# Nuage de mots pour un document
# --------------------------
def nuage_mots_document(document: str, pretraitement: bool = True, stopwords: set = None):
    """
    Génère un nuage de mots pour un document.
    """
    texte = document
    if pretraitement:
        # Simple tokenisation et nettoyage
        tokens = document.lower().split()
        if stopwords:
            tokens = [t for t in tokens if t not in stopwords]
        texte = " ".join(tokens)
    
    generer_nuage_mots_texte(texte, stopwords=stopwords)

# --------------------------
# Nuage de mots pour un sous-corpus
# --------------------------
def nuage_mots_sous_corpus(liste_documents: List[str], pretraitement: bool = True, stopwords: set = None):
    """
    Concatène les textes et génère un nuage de mots global.
    """
    textes = []
    for doc in liste_documents:
        if pretraitement:
            tokens = doc.lower().split()
            if stopwords:
                tokens = [t for t in tokens if t not in stopwords]
            textes.append(" ".join(tokens))
        else:
            textes.append(doc)
    
    texte_concatene = " ".join(textes)
    generer_nuage_mots_texte(texte_concatene, stopwords=stopwords)

# --------------------------
# Nuage de mots pondéré par des scores
# --------------------------
def nuage_mots_pondere(poids_par_mot: dict, largeur=600, hauteur=300, couleur_fond='white'):
    """
    Génère un nuage de mots à partir d'un dictionnaire {mot: poids}.
    """
    wc = WordCloud(width=largeur, height=hauteur, background_color=couleur_fond)
    wc.generate_from_frequencies(poids_par_mot)
    plt.figure(figsize=(largeur/100, hauteur/100))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# --------------------------
# Comparer côte à côte deux nuages de mots
# --------------------------
def comparer_nuages_documents(doc1, doc2, scores1: dict, scores2: dict, largeur=600, hauteur=300, couleur_fond='white'):
    """
    Affiche côte à côte deux nuages de mots pour comparer visuellement deux documents.
    """
    fig, axes = plt.subplots(1, 2, figsize=(largeur/50, hauteur/100))
    
    wc1 = WordCloud(width=largeur, height=hauteur, background_color=couleur_fond)
    wc1.generate_from_frequencies(scores1)
    axes[0].imshow(wc1, interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title("Document 1")
    
    wc2 = WordCloud(width=largeur, height=hauteur, background_color=couleur_fond)
    wc2.generate_from_frequencies(scores2)
    axes[1].imshow(wc2, interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title("Document 2")
    
    plt.show()

# --------------------------
# Nuage pour un document retourné par le moteur
# --------------------------
def nuage_document_resultat(document, poids: dict):
    """
    Affiche un nuage de mots pour un document avec des poids associés.
    """
    nuage_mots_pondere(poids)

# --------------------------
# Nuage global à partir d'une liste de documents et scores
# --------------------------
def nuage_mots_resultats_recherche(documents: list, scores_list: list = None):
    """
    Génère un nuage de mots à partir des documents retournés par une requête.
    Si scores_list est fourni, pondère par les scores correspondants.
    """
    freqs = {}
    for idx, doc in enumerate(documents):
        tokens = doc.lower().split()
        if scores_list is not None:
            poids_doc = scores_list[idx]
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + poids_doc.get(t, 1.0)
        else:
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1.0
    nuage_mots_pondere(freqs)

# --------------------------
# Nuage pour les top-k documents proches d'une requête
# --------------------------
def nuage_mots_top_k_documents(
    requete: List[str],
    corpus: List[List[str]],
    modele_embeddings,
    k: int = 5,
    methode: str = 'embeddings',
    strategie_agregation: str = 'moyenne',
    strategie_oov: str = 'ignore'
):
    """
    Affiche un nuage de mots représentant les k documents les plus proches sémantiquement d'une requête.
    """
    top_docs = top_k_documents_similaires(requete, corpus, modele_embeddings, methode, k, strategie_agregation, strategie_oov)
    documents_top = [corpus[idx] for idx, _ in top_docs]
    freqs = {}
    for doc in documents_top:
        for token in doc:
            freqs[token] = freqs.get(token, 0) + 1
    nuage_mots_pondere(freqs)

# --------------------------
# Nuage comparatif entre requête et document
# --------------------------
def nuage_requete_vs_document(
    requete: List[str],
    document: List[str],
    poids_doc: dict
):
    """
    Visualise conjointement les termes de la requête et du document pour analyser le recouvrement lexical.
    Les mots de la requête sont en gras ou surlignés par un poids plus fort.
    """
    freqs = {}
    for token in document:
        freqs[token] = poids_doc.get(token, 1.0)
    for token in requete:
        freqs[token] = freqs.get(token, 1.0) + 2  # mettre un poids plus fort pour la requête
    
    nuage_mots_pondere(freqs)

import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --------------------------
# 1️⃣ Termes d'expansion pour une requête
# --------------------------
def termes_expansion(requete: list, modele_embeddings, k: int = 5):
    """
    Pour chaque mot de la requête, retourne les k mots les plus proches dans l'espace vectoriel.
    
    Args:
        requete : liste de tokens (ex: ["chat", "dort"])
        modele_embeddings : Word2Vec ou FastText
        k : nombre de mots proches à récupérer
    
    Returns:
        dict {mot_requete: [(mot_proche, score), ...]}
    """
    expansion = {}
    for mot in requete:
        if mot in modele_embeddings.wv:
            voisins = modele_embeddings.wv.most_similar(mot, topn=k)
            expansion[mot] = voisins
        else:
            expansion[mot] = []
    return expansion

# --------------------------
# 2️⃣ Construire une requête étendue avec pondérations
# --------------------------
def construire_requete_etendue(requete: list, termes_expanses: dict, alpha: float = 1.0, beta: float = 0.5):
    """
    Combine la requête initiale et les termes d'expansion.
    
    Args:
        requete : liste de tokens
        termes_expanses : dict {mot: [(mot_proche, score), ...]}
        alpha : poids des termes originaux
        beta : poids des termes d'expansion
    
    Returns:
        dict {mot: poids}
    """
    requete_etendue = {}
    # termes originaux
    for mot in requete:
        requete_etendue[mot] = alpha
    # termes d'expansion
    for mot_orig, voisins in termes_expanses.items():
        for mot_voisin, score in voisins:
            if mot_voisin in requete_etendue:
                requete_etendue[mot_voisin] += beta * score
            else:
                requete_etendue[mot_voisin] = beta * score
    return requete_etendue

# --------------------------
# 3️⃣ Recherche avec expansion de requête
# --------------------------
def recherche_avec_expansion(requete: list, corpus: list, modele_embeddings, methode='embeddings', k_expansion=5, top_k=5):
    """
    Recherche des documents en intégrant l'expansion de requête.
    
    Args:
        requete : liste de tokens
        corpus : liste de documents (listes de tokens)
        modele_embeddings : Word2Vec, FastText ou Doc2Vec
        methode : 'embeddings', 'doc2vec', 'vecteurs_precalcules'
        k_expansion : nombre de termes voisins pour expansion
        top_k : nombre de documents à retourner
    
    Returns:
        liste des tuples (indice_doc, score)
    """
    # 1. Obtenir les termes d'expansion
    termes_exp = termes_expansion(requete, modele_embeddings, k=k_expansion)
    # 2. Construire la requête étendue
    requete_etendue = construire_requete_etendue(requete, termes_exp, alpha=1.0, beta=0.5)
    
    # 3. Pondérer le corpus selon la requête étendue et calculer similarité
    # Ici on utilise top_k_documents_similaires avec un ajustement simple :
    # On "étend" chaque mot de la requête par sa pondération pour la similarité
    # Pour simplifier, on utilise uniquement plongement_document_par_mots
    scores = []
    for idx, doc in enumerate(corpus):
        v_doc = plongement_document_par_mots(doc, modele_embeddings)
        # pondération de la requête comme un vecteur pseudo-document
        v_requete = plongement_document_par_mots(list(requete_etendue.keys()), modele_embeddings)
        sim = cosine_similarity(v_doc.reshape(1, -1), v_requete.reshape(1, -1))[0][0]
        scores.append((idx, sim))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

# --------------------------
# 4️⃣ Visualisation des voisins sémantiques pour un mot
# --------------------------
def mots_plus_proches(mot: str, modele_embeddings, k: int = 20):
    """
    Retourne les k mots les plus proches d'un mot avec leur score de similarité.
    """
    if mot in modele_embeddings.wv:
        return modele_embeddings.wv.most_similar(mot, topn=k)
    else:
        return []

def nuage_expansion_requete(mot: str, modele_embeddings, k: int = 20):
    """
    Génère un nuage de mots représentant les termes utilisés pour l'expansion d'un mot.
    """
    voisins = mots_plus_proches(mot, modele_embeddings, k)
    poids = {mot_voisin: score for mot_voisin, score in voisins}
    nuage_mots_pondere(poids)

def comparer_expansion_requete(mot: str, modele_embeddings1, modele_embeddings2, k: int = 20):
    """
    Compare visuellement l'expansion produite par deux modèles de plongements différents.
    """
    voisins1 = mots_plus_proches(mot, modele_embeddings1, k)
    voisins2 = mots_plus_proches(mot, modele_embeddings2, k)
    poids1 = {m: s for m, s in voisins1}
    poids2 = {m: s for m, s in voisins2}
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    wc1 = WordCloud(width=600, height=300).generate_from_frequencies(poids1)
    axes[0].imshow(wc1, interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title("Modèle 1")
    
    wc2 = WordCloud(width=600, height=300).generate_from_frequencies(poids2)
    axes[1].imshow(wc2, interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title("Modèle 2")
    
    plt.show()


# --------------------------
# 1️⃣ Initialisation des fiches pour chaque unité (phrase ou document)
# --------------------------
def initialiser_fiche_unite(id_unite, type_unite, texte, config):
    """
    Crée une fiche vide pour une unité linguistique (phrase ou document).
    
    Args:
        id_unite : identifiant unique
        type_unite : 'phrase' ou 'document'
        texte : texte brut ou tokenisé
        config : dictionnaire de configuration de prétraitement
    
    Returns:
        dict représentant la fiche de l'unité
    """
    fiche = {
        "id": id_unite,
        "type": type_unite,
        "texte": texte,
        "config": config,
        "descripteurs": {},   # contiendra BOW, TF, TF-IDF, embeddings, etc.
        "parametres": {},     # normalisation, stratégie d'agrégation, pondération
        "metadonnees": {      # longueur, nombre de tokens, langue, sous-corpus
            "longueur": len(texte) if isinstance(texte, str) else len(texte),
            "tokens": len(texte) if isinstance(texte, list) else len(texte.split()),
            "langue": config.get("langue", None),
            "sous_corpus": config.get("sous_corpus", None)
        },
        "statut": "vide"      # 'vide', 'calculé', 'chargé'
    }
    return fiche

# --------------------------
# 2️⃣ Dictionnaire global des fiches
# --------------------------
def initialiser_dictionnaire_global():
    """
    Crée un dictionnaire global pour toutes les fiches du corpus.
    """
    return {}

def ajouter_fiche(dico_global, fiche):
    """
    Ajoute une fiche au dictionnaire global.
    
    Args:
        dico_global : dictionnaire global
        fiche : fiche à ajouter
    """
    dico_global[fiche["id"]] = fiche

def filtrer_fiches(dico_global, langue=None, config_id=None, type_unite=None):
    """
    Retourne un sous-ensemble de fiches selon les critères spécifiés.
    
    Args:
        dico_global : dictionnaire global
        langue : filtrage par langue
        config_id : filtrage par configuration de prétraitement
        type_unite : 'phrase' ou 'document'
    
    Returns:
        dict filtré
    """
    resultat = {}
    for id_u, fiche in dico_global.items():
        if langue is not None and fiche["metadonnees"].get("langue") != langue:
            continue
        if config_id is not None and fiche["config"].get("id") != config_id:
            continue
        if type_unite is not None and fiche["type"] != type_unite:
            continue
        resultat[id_u] = fiche
    return resultat

# --------------------------
# 3️⃣ Vocabulaire des embeddings pour consultation rapide
# --------------------------
def initialiser_vocabulaire_embeddings(modele_embeddings):
    """
    Crée un dictionnaire mot -> vecteur pour tous les mots du modèle.
    Utile pour l'agrégation rapide de phrases/documents et l'expansion de requête.
    
    Args:
        modele_embeddings : Word2Vec ou FastText
        
    Returns:
        dict {mot: vecteur}
    """
    vocab = {}
    for mot in modele_embeddings.wv.index_to_key:
        vocab[mot] = modele_embeddings.wv[mot]
    return vocab




# --------------------------
# 1️⃣ Descripteurs symboliques
# --------------------------

def calculer_bow_fiches(fiches, vocabulaire, type_bow="binaire", config_id=None, normalisation=None):
    """
    Calcule et stocke les vecteurs BOW pour les fiches sélectionnées.
    """
    for fiche in fiches.values():
        if config_id is not None and fiche["config"].get("id") != config_id:
            continue
        tokens = fiche["texte"] if isinstance(fiche["texte"], list) else fiche["texte"].split()
        vecteur = calculer_bow(tokens, vocabulaire, type_bow=type_bow)  # fonction atomique existante
        if normalisation:
            vecteur = normaliser_vecteur(vecteur, normalisation)
        fiche["descripteurs"]["BOW"] = vecteur
        fiche["parametres"]["BOW"] = {"type_bow": type_bow, "normalisation": normalisation, "config_id": fiche["config"].get("id")}
        fiche["statut"] = "calculé"

def calculer_tf_fiches(fiches, vocabulaire, config_id=None, normalisation=None):
    """
    Calcule et stocke les vecteurs TF.
    """
    for fiche in fiches.values():
        if config_id is not None and fiche["config"].get("id") != config_id:
            continue
        tokens = fiche["texte"] if isinstance(fiche["texte"], list) else fiche["texte"].split()
        vecteur = calculer_tf(tokens, vocabulaire)  # fonction atomique existante
        if normalisation:
            vecteur = normaliser_vecteur(vecteur, normalisation)
        fiche["descripteurs"]["TF"] = vecteur
        fiche["parametres"]["TF"] = {"normalisation": normalisation, "config_id": fiche["config"].get("id")}
        fiche["statut"] = "calculé"

def calculer_tfidf_fiches(fiches, idf_vecteur, config_id=None, normalisation=None):
    """
    Calcule et stocke les vecteurs TF-IDF à partir des vecteurs TF.
    """
    for fiche in fiches.values():
        if config_id is not None and fiche["config"].get("id") != config_id:
            continue
        tf_vecteur = fiche["descripteurs"].get("TF")
        if tf_vecteur is None:
            continue  # TF doit être calculé avant
        vecteur = calculer_tfidf(tf_vecteur, idf_vecteur)  # fonction atomique existante
        if normalisation:
            vecteur = normaliser_vecteur(vecteur, normalisation)
        fiche["descripteurs"]["TF-IDF"] = vecteur
        fiche["parametres"]["TF-IDF"] = {"normalisation": normalisation, "config_id": fiche["config"].get("id")}
        fiche["statut"] = "calculé"

def calculer_bm25_fiches(fiches, vocabulaire, stats_corpus, parametres_bm25, config_id=None, normalisation=None):
    """
    Calcule et stocke les vecteurs BM25.
    """
    for fiche in fiches.values():
        if config_id is not None and fiche["config"].get("id") != config_id:
            continue
        tokens = fiche["texte"] if isinstance(fiche["texte"], list) else fiche["texte"].split()
        vecteur = calculer_bm25(tokens, vocabulaire, stats_corpus, parametres_bm25)  # fonction atomique existante
        if normalisation:
            vecteur = normaliser_vecteur(vecteur, normalisation)
        fiche["descripteurs"]["BM25"] = vecteur
        fiche["parametres"]["BM25"] = {"parametres": parametres_bm25, "normalisation": normalisation, "config_id": fiche["config"].get("id")}
        fiche["statut"] = "calculé"

# --------------------------
# 2️⃣ Descripteurs vectoriels (embeddings)
# --------------------------

def calculer_embeddings_phrases(fiches, modele_embeddings, strategie_agregation="moyenne", strategie_oov="ignore"):
    """
    Calcule les embeddings pour les phrases et stocke les vecteurs dans les fiches.
    """
    for fiche in fiches.values():
        if fiche["type"] != "phrase":
            continue
        tokens = fiche["texte"] if isinstance(fiche["texte"], list) else fiche["texte"].split()
        vecteur = plongement_phrase_par_mots(tokens, modele_embeddings, strategie_agregation, strategie_oov)
        fiche["descripteurs"]["embedding_phrase"] = vecteur
        fiche["parametres"]["embedding_phrase"] = {"strategie_agregation": strategie_agregation, "strategie_oov": strategie_oov}
        fiche["statut"] = "calculé"

def calculer_embeddings_documents_mots(fiches, modele_embeddings, strategie_agregation="moyenne", strategie_oov="ignore"):
    """
    Calcule les embeddings des documents par agrégation des embeddings de mots.
    """
    for fiche in fiches.values():
        if fiche["type"] != "document":
            continue
        tokens = fiche["texte"] if isinstance(fiche["texte"], list) else fiche["texte"].split()
        vecteur = plongement_document_par_mots(tokens, modele_embeddings, strategie_agregation, strategie_oov)
        fiche["descripteurs"]["embedding_doc_mots"] = vecteur
        fiche["parametres"]["embedding_doc_mots"] = {"strategie_agregation": strategie_agregation, "strategie_oov": strategie_oov}
        fiche["statut"] = "calculé"

def calculer_embeddings_documents_phrases(fiches, modele_embeddings, strategie_agregation_phrase="moyenne", strategie_agregation_doc="moyenne", strategie_oov="ignore"):
    """
    Calcule les embeddings des documents par agrégation des embeddings de phrases déjà disponibles.
    """
    for fiche in fiches.values():
        if fiche["type"] != "document":
            continue
        # récupère toutes les phrases associées
        phrases_vecteurs = fiche.get("phrases_vecteurs", [])
        if not phrases_vecteurs:
            continue
        vecteurs_valides = [v for v in phrases_vecteurs if v is not None]
        if not vecteurs_valides:
            vecteur_doc = None
        else:
            vecteur_doc = np.mean(vecteurs_valides, axis=0) if strategie_agregation_doc=="moyenne" else np.sum(vecteurs_valides, axis=0)
        fiche["descripteurs"]["embedding_doc_phrases"] = vecteur_doc
        fiche["parametres"]["embedding_doc_phrases"] = {"strategie_agregation_phrase": strategie_agregation_phrase, "strategie_agregation_doc": strategie_agregation_doc, "strategie_oov": strategie_oov}
        fiche["statut"] = "calculé"

def calculer_embeddings_documents_doc2vec(fiches, modele_doc2vec):
    """
    Utilise un modèle Doc2Vec pour inférer directement les embeddings des documents.
    """
    for fiche in fiches.values():
        if fiche["type"] != "document":
            continue
        texte_tokens = fiche["texte"] if isinstance(fiche["texte"], list) else fiche["texte"].split()
        vecteur = modele_doc2vec.infer_vector(texte_tokens)
        fiche["descripteurs"]["embedding_doc_doc2vec"] = vecteur
        fiche["parametres"]["embedding_doc_doc2vec"] = {"modele_doc2vec": True}
        fiche["statut"] = "calculé"


import numpy as np
from scipy.sparse import vstack, csr_matrix

# --------------------------
# 1️⃣ Index symbolique
# --------------------------
def construire_index_symbolique(fiches, type_descripteur, config_id=None):
    """
    Construit un index symbolique (matrice document-terme ou phrase-terme)
    à partir des fiches du corpus.
    """
    vecteurs = []
    id_lignes = []

    for fiche in fiches.values():
        if config_id is not None and fiche["config"].get("id") != config_id:
            continue
        vecteur = fiche["descripteurs"].get(type_descripteur)
        if vecteur is not None:
            if not isinstance(vecteur, csr_matrix):
                vecteur = csr_matrix(vecteur)  # convertir en sparse si nécessaire
            vecteurs.append(vecteur)
            id_lignes.append(fiche["id"])

    if vecteurs:
        matrice = vstack(vecteurs)
    else:
        matrice = csr_matrix((0, 0))

    mapping_id_ligne = {uid: idx for idx, uid in enumerate(id_lignes)}

    index_symbolique = {
        "matrice": matrice,
        "mapping_id_ligne": mapping_id_ligne,
        "type_descripteur": type_descripteur,
        "config_id": config_id
    }
    return index_symbolique

def mettre_a_jour_index_symbolique(index_symbolique, fiche):
    """
    Ajoute dynamiquement les descripteurs symboliques d'une fiche à un index existant.
    """
    vecteur = fiche["descripteurs"].get(index_symbolique["type_descripteur"])
    if vecteur is None:
        return  # rien à ajouter

    if not isinstance(vecteur, csr_matrix):
        vecteur = csr_matrix(vecteur)
    
    index_symbolique["matrice"] = vstack([index_symbolique["matrice"], vecteur])
    nouvelle_ligne = index_symbolique["matrice"].shape[0] - 1
    index_symbolique["mapping_id_ligne"][fiche["id"]] = nouvelle_ligne

# --------------------------
# 2️⃣ Index vectoriel (embeddings)
# --------------------------
def construire_index_embeddings(fiches, type_embeddings, config_id=None):
    """
    Construit un index vectoriel (matrice dense) à partir des embeddings des fiches.
    """
    vecteurs = []
    id_lignes = []

    for fiche in fiches.values():
        if config_id is not None and fiche["config"].get("id") != config_id:
            continue
        vecteur = fiche["descripteurs"].get(type_embeddings)
        if vecteur is not None:
            vecteurs.append(np.array(vecteur))
            id_lignes.append(fiche["id"])

    if vecteurs:
        matrice = np.vstack(vecteurs)
    else:
        matrice = np.zeros((0, 0))

    mapping_id_ligne = {uid: idx for idx, uid in enumerate(id_lignes)}

    index_embeddings = {
        "matrice": matrice,
        "mapping_id_ligne": mapping_id_ligne,
        "type_embeddings": type_embeddings,
        "config_id": config_id
    }
    return index_embeddings

def mettre_a_jour_index_embeddings(index_embeddings, fiche):
    """
    Ajoute dynamiquement les embeddings d'une nouvelle fiche à un index vectoriel existant.
    """
    vecteur = fiche["descripteurs"].get(index_embeddings["type_embeddings"])
    if vecteur is None:
        return  # rien à ajouter

    vecteur = np.array(vecteur)
    if index_embeddings["matrice"].size == 0:
        index_embeddings["matrice"] = vecteur.reshape(1, -1)
    else:
        index_embeddings["matrice"] = np.vstack([index_embeddings["matrice"], vecteur])
    
    nouvelle_ligne = index_embeddings["matrice"].shape[0] - 1
    index_embeddings["mapping_id_ligne"][fiche["id"]] = nouvelle_ligne

import numpy as np
from collections import defaultdict

# --------------------------
# 1️⃣ Pré-calcul des normes
# --------------------------

def precalculer_normes_embeddings(index_embeddings):
    """
    Calcule la norme de chaque vecteur dans un index vectoriel pour accélérer
    la similarité cosinus.
    """
    matrice = index_embeddings["matrice"]
    if matrice.size == 0:
        index_embeddings["normes"] = np.array([])
    else:
        index_embeddings["normes"] = np.linalg.norm(matrice, axis=1)
    return index_embeddings

def precalculer_normes_symboliques(index_symbolique):
    """
    Calcule la norme de chaque vecteur symbolique (BOW, TF, TF-IDF, BM25)
    pour accélérer la similarité cosinus.
    """
    matrice = index_symbolique["matrice"]
    if matrice.shape[0] == 0:
        index_symbolique["normes"] = np.array([])
    else:
        # matrice sparse CSR
        index_symbolique["normes"] = np.sqrt(matrice.multiply(matrice).sum(axis=1)).A1
    return index_symbolique

# --------------------------
# 2️⃣ Index inversé pour modèles symboliques
# --------------------------

def construire_index_inverse(fiches, type_descripteur, config_id=None):
    """
    Construit un index inversé {terme: {doc_id: poids}} pour descripteurs symboliques.
    Pour BOW simple, le poids peut être 1 ; pour TF/TF-IDF/BM25, on stocke la valeur exacte.
    """
    index_inverse = defaultdict(dict)

    for fiche in fiches.values():
        if config_id is not None and fiche["config"].get("id") != config_id:
            continue
        vecteur = fiche["descripteurs"].get(type_descripteur)
        if vecteur is None:
            continue
        
        # Si vecteur sparse
        if hasattr(vecteur, "tocoo"):
            vecteur = vecteur.tocoo()
            for i, j, v in zip(vecteur.row, vecteur.col, vecteur.data):
                index_inverse[j][fiche["id"]] = v
        else:
            # vecteur dense ou liste
            for idx, val in enumerate(vecteur):
                if val != 0:
                    index_inverse[idx][fiche["id"]] = val

    return index_inverse

def mettre_a_jour_index_inverse(index_inverse, fiche, type_descripteur):
    """
    Met à jour un index inversé existant avec une nouvelle fiche.
    """
    vecteur = fiche["descripteurs"].get(type_descripteur)
    if vecteur is None:
        return

    if hasattr(vecteur, "tocoo"):
        vecteur = vecteur.tocoo()
        for i, j, v in zip(vecteur.row, vecteur.col, vecteur.data):
            index_inverse[j][fiche["id"]] = v
    else:
        for idx, val in enumerate(vecteur):
            if val != 0:
                index_inverse[idx][fiche["id"]] = val

import pickle
import os

# --------------------------
# 1️⃣ Sauvegarde des fiches
# --------------------------

def sauvegarder_fiches(fichier, dictionnaire_global):
    """
    Sauvegarde l'ensemble des fiches sur disque.
    """
    with open(fichier, "wb") as f:
        pickle.dump(dictionnaire_global, f)
    print(f"Fiches sauvegardées dans {fichier}.")

def sauvegarder_fiche_unitaire(fichier, fiche):
    """
    Sauvegarde une seule fiche si au moins un descripteur est calculé.
    """
    descripteurs_calcules = any(
        desc.get("statut") == "calculé" for desc in fiche.get("descripteurs", {}).values()
    )
    if descripteurs_calcules:
        with open(fichier, "wb") as f:
            pickle.dump(fiche, f)
        print(f"Fiche {fiche['id']} sauvegardée dans {fichier}.")

def sauvegarder_fiches_calculees(dico_global, dossier_sortie):
    """
    Sauvegarde toutes les fiches dont au moins un descripteur a le statut 'calculé'.
    Chaque fiche est sauvegardée individuellement.
    """
    if not os.path.exists(dossier_sortie):
        os.makedirs(dossier_sortie)
    
    for fiche_id, fiche in dico_global.items():
        fichier = os.path.join(dossier_sortie, f"fiche_{fiche_id}.pkl")
        sauvegarder_fiche_unitaire(fichier, fiche)

# --------------------------
# 2️⃣ Sauvegarde des index
# --------------------------

def sauvegarder_index_symbolique(fichier, index_symbolique):
    """
    Sauvegarde l'index symbolique (matrice + vocabulaire + mapping + métadonnées).
    """
    with open(fichier, "wb") as f:
        pickle.dump(index_symbolique, f)
    print(f"Index symbolique sauvegardé dans {fichier}.")

def sauvegarder_index_embeddings(fichier, index_embeddings):
    """
    Sauvegarde l'index vectoriel (matrice dense + mapping + métadonnées).
    """
    with open(fichier, "wb") as f:
        pickle.dump(index_embeddings, f)
    print(f"Index embeddings sauvegardé dans {fichier}.")

# --------------------------
# 3️⃣ Sauvegarde des normes
# --------------------------

def sauvegarder_normes_embeddings(fichier, index_embeddings):
    """
    Sauvegarde les normes pré-calculées des vecteurs d'embeddings.
    """
    normes = index_embeddings.get("normes", None)
    with open(fichier, "wb") as f:
        pickle.dump(normes, f)
    print(f"Normes embeddings sauvegardées dans {fichier}.")

def sauvegarder_normes_symboliques(fichier, index_symbolique):
    """
    Sauvegarde les normes pré-calculées des vecteurs symboliques.
    """
    normes = index_symbolique.get("normes", None)
    with open(fichier, "wb") as f:
        pickle.dump(normes, f)
    print(f"Normes symboliques sauvegardées dans {fichier}.")

# --------------------------
# 4️⃣ Sauvegarde et mise à jour de l'index inversé
# --------------------------

def sauvegarder_index_inverse(fichier, index_inverse):
    """
    Sauvegarde l'index inversé pour descripteurs symboliques.
    """
    with open(fichier, "wb") as f:
        pickle.dump(dict(index_inverse), f)
    print(f"Index inversé sauvegardé dans {fichier}.")

def mettre_a_jour_index_inverse_persistant(fichier, index_inverse):
    """
    Met à jour un index inversé existant sur disque avec de nouvelles fiches.
    """
    if os.path.exists(fichier):
        with open(fichier, "rb") as f:
            index_existant = pickle.load(f)
        # Fusionner l'ancien index avec le nouveau
        for terme, docs in index_inverse.items():
            if terme in index_existant:
                index_existant[terme].update(docs)
            else:
                index_existant[terme] = docs
        index_inverse = index_existant
    
    # Sauvegarde finale
    with open(fichier, "wb") as f:
        pickle.dump(index_inverse, f)
    print(f"Index inversé mis à jour et sauvegardé dans {fichier}.")


import pickle
import os

# --------------------------
# 1️⃣ Chargement des fiches
# --------------------------

def charger_fiches(fichier):
    """
    Charge toutes les fiches sauvegardées depuis un fichier et retourne le dictionnaire global.
    """
    with open(fichier, "rb") as f:
        dictionnaire_global = pickle.load(f)
    print(f"{len(dictionnaire_global)} fiches chargées depuis {fichier}.")
    return dictionnaire_global

def charger_fiche_unitaire(fichier):
    """
    Charge une fiche individuelle depuis le disque et la retourne.
    """
    with open(fichier, "rb") as f:
        fiche = pickle.load(f)
    return fiche

def charger_fiches_depuis_dossier(dossier):
    """
    Parcourt un dossier contenant des fichiers de fiches individuelles et
    reconstruit le dictionnaire global des fiches.
    """
    dictionnaire_global = {}
    for fichier in os.listdir(dossier):
        if fichier.endswith(".pkl"):
            fiche = charger_fiche_unitaire(os.path.join(dossier, fichier))
            integrer_fiche_chargee(dictionnaire_global, fiche)
    print(f"{len(dictionnaire_global)} fiches chargées depuis le dossier {dossier}.")
    return dictionnaire_global

def integrer_fiche_chargee(dico_global, fiche):
    """
    Ajoute une fiche chargée dans le dictionnaire global en vérifiant la cohérence.
    """
    id_fiche = fiche["id"]
    if id_fiche in dico_global:
        print(f"Avertissement : fiche {id_fiche} déjà présente. Fusion possible.")
        # Exemple simple : on remplace les descripteurs existants
        dico_global[id_fiche].update(fiche)
    else:
        dico_global[id_fiche] = fiche

# --------------------------
# 2️⃣ Chargement des index
# --------------------------

def charger_index_symbolique(fichier):
    """
    Recharge un index symbolique depuis le disque.
    """
    with open(fichier, "rb") as f:
        index_symbolique = pickle.load(f)
    return index_symbolique

def charger_index_embeddings(fichier):
    """
    Recharge un index vectoriel depuis le disque.
    """
    with open(fichier, "rb") as f:
        index_embeddings = pickle.load(f)
    return index_embeddings

# --------------------------
# 3️⃣ Vérification de cohérence
# --------------------------

def verifier_coherence(dictionnaire_global):
    """
    Vérifie l'unicité des identifiants et la cohérence globale des fiches.
    """
    ids = set()
    for id_fiche, fiche in dictionnaire_global.items():
        if id_fiche in ids:
            print(f"Alerte : doublon d'identifiant {id_fiche}.")
        ids.add(id_fiche)
        # Vérification basique des métadonnées
        if "type" not in fiche or "texte" not in fiche:
            print(f"Alerte : fiche {id_fiche} incomplète.")
    print(f"{len(ids)} fiches uniques vérifiées.")

def gestion_vecteurs_absents(fiche, strategie='ignore'):
    """
    Gère les vecteurs absents selon la stratégie choisie.
    """
    vecteurs = fiche.get("descripteurs", {})
    for nom_desc, desc in vecteurs.items():
        if desc.get("vecteur") is None:
            if strategie == 'alerte':
                print(f"Alerte : vecteur manquant pour {nom_desc} dans fiche {fiche['id']}")
            elif strategie == 'recalcul':
                print(f"Recalcul nécessaire pour {nom_desc} dans fiche {fiche['id']}")
                # Ici, on pourrait appeler la fonction de calcul
            # ignore : ne rien faire

# --------------------------
# 4️⃣ Chargement des normes
# --------------------------

def charger_normes_embeddings(fichier):
    """
    Recharge les normes pré-calculées des vecteurs d'embeddings.
    """
    with open(fichier, "rb") as f:
        normes = pickle.load(f)
    return normes

def charger_normes_symboliques(fichier):
    """
    Recharge les normes pré-calculées des vecteurs symboliques.
    """
    with open(fichier, "rb") as f:
        normes = pickle.load(f)
    return normes

# --------------------------
# 5️⃣ Chargement de l'index inversé
# --------------------------

def charger_index_inverse(fichier):
    """
    Recharge un index inversé depuis le disque.
    """
    with open(fichier, "rb") as f:
        index_inverse = pickle.load(f)
    return index_inverse

def charger_index_inverse_mise_a_jour(fichier):
    """
    Recharge un index inversé et fusionne les mises à jour avec l'index existant.
    """
    if os.path.exists(fichier):
        with open(fichier, "rb") as f:
            index_inverse = pickle.load(f)
        return dict(index_inverse)
    else:
        return {}
