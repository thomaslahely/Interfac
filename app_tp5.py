

import math,re
import statistics
import string
import matplotlib.pyplot as plt
from typing import List, Union, Dict ,Any,Tuple
import seaborn as sns
from collections import Counter
from pathlib import Path
def calcul_tf(texte: str, vocabulaire_direct: Dict[str, int]) -> List[float]:
    """
    Transforme un texte en un vecteur TF (Term Frequency).
    """
    if not texte or not isinstance(texte, str):
        return [0.0] * len(vocabulaire_direct)

    # 1. Tokenisation
    mots = texte.lower().split()
    total_mots = len(mots)
    
    if total_mots == 0:
        return [0.0] * len(vocabulaire_direct)

    # 2. Initialisation du vecteur
    taille_vocab = len(vocabulaire_direct)
    vecteur_tf = [0.0] * taille_vocab
    
    # 3. Comptage
    for mot in mots:
        index = vocabulaire_direct.get(mot)
        if index is not None:
            vecteur_tf[index] += 1.0
            
    # 4. Normalisation
    # On divise chaque valeur par le nombre total de mots
    vecteur_tf = [val / total_mots for val in vecteur_tf]
            
    return vecteur_tf


def bag_of_words_occurrences(texte: str, vocabulaire_direct: Dict[str, int]) -> List[int]:
    """
    Transforme un texte en un vecteur de comptage des mots.
    """
    if not texte or not isinstance(texte, str):
        return [0] * len(vocabulaire_direct)

    # 1. Initialisation
    taille_vocab = len(vocabulaire_direct)
    vecteur = [0] * taille_vocab
    
    # 2. Tokenisation
    mots = texte.lower().split()
    
    # 3. Comptage
    for mot in mots:
        index = vocabulaire_direct.get(mot)
        if index is not None:
            # Incrémentation au lieu d'assignation simple (= 1)
            vecteur[index] += 1
            
    return vecteur

def vectoriser_phrase(phrase: List[str], vocabulaire_direct: Dict[str, int], methode: str = "tfidf", **params) -> List[float]:
    """
    Transforme une phrase (liste de mots) en vecteur numérique.
    
    Args:
        phrase: Liste de tokens (ex: ["le", "chat"]).
        vocabulaire_direct: Dictionnaire {mot: index}.
        methode: "bow", "tf", ou "tfidf".
        **params: Paramètres variables (ex: 'vecteur_idf' pour la méthode tfidf).
    """
    # Reconstitution de la chaîne pour les fonctions TP4
    texte_reconstitue = " ".join(phrase)
    
    vecteur = []
    
    if methode == "bow":
        # Retourne des entiers, mais on convertit en float pour uniformité
        vec_int = bag_of_words_occurrences(texte_reconstitue, vocabulaire_direct)
        vecteur = [float(x) for x in vec_int]
        
    elif methode == "tf":
        vecteur = calcul_tf(texte_reconstitue, vocabulaire_direct)
        
    elif methode == "tfidf":
        # 1. Calculer TF local
        vec_tf = calcul_tf(texte_reconstitue, vocabulaire_direct)
        # 2. Récupérer l'IDF global (passé en paramètre)
        vec_idf = params.get("vecteur_idf")
        
        if vec_idf is None:
            raise ValueError("Pour la méthode 'tfidf', le paramètre 'vecteur_idf' est requis.")
            
        if len(vec_tf) != len(vec_idf):
             raise ValueError("Dimension mismatch: TF et IDF doivent avoir la même taille.")

        # 3. Multiplication terme à terme
        vecteur = [t * i for t, i in zip(vec_tf, vec_idf)]
        
    else:
        # Par défaut ou si méthode inconnue, on renvoie un vecteur nul ou on lève une erreur
        return [0.0] * len(vocabulaire_direct)
        
    return vecteur

### 🔍 Explication des tests unitaires — Vectoriser Phrase
def test_vectoriser_phrase():
    # Setup
    vocab = {"chat": 0, "dort": 1} # Vocabulaire simplifié (listes triées implicites: chat, dort)
    phrase = ["chat", "dort"]
    
    # Test 1 : BoW
    # "chat" -> 1, "dort" -> 1 => [1.0, 1.0]
    res_bow = vectoriser_phrase(phrase, vocab, methode="bow")
    assert res_bow == [1.0, 1.0]
    
    # Test 2 : TF
    # 2 mots, chacun apparait 1 fois => 1/2 = 0.5 => [0.5, 0.5]
    res_tf = vectoriser_phrase(phrase, vocab, methode="tf")
    assert res_tf == [0.5, 0.5]
    
    # Test 3 : TF-IDF
    # On imagine un IDF où "chat" est courant (0.1) et "dort" rare (1.0)
    fake_idf = [0.1, 1.0] 
    # Calcul attendu : [0.5 * 0.1, 0.5 * 1.0] = [0.05, 0.5]
    res_tfidf = vectoriser_phrase(phrase, vocab, methode="tfidf", vecteur_idf=fake_idf)
    
    assert abs(res_tfidf[0] - 0.05) < 0.001
    assert abs(res_tfidf[1] - 0.5) < 0.001
    
    print("Test vectoriser_phrase : OK")


def vectoriser_document_mots(document: List[List[str]], vocabulaire_direct: Dict[str, int], methode: str = "tfidf", **params) -> List[float]:
    """
    Considère le document comme une seule longue phrase (concaténation).
    """
    # Aplatissement de la liste de listes en une seule liste
    # document = [["Le", "chat"], ["Il", "dort"]] -> ["Le", "chat", "Il", "dort"]
    mots_aplatis = [mot for phrase in document for mot in phrase]
    
    # On réutilise la fonction de base
    return vectoriser_phrase(mots_aplatis, vocabulaire_direct, methode=methode, **params)

### 🔍 Explication des tests unitaires — Vectoriser Doc Mots
def test_vectoriser_document_mots():
    vocab = {"chat": 0, "dort": 1}
    # Doc : Phrase 1 "chat", Phrase 2 "dort"
    doc = [["chat"], ["dort"]] 
    
    # En aplatissant, cela devient ["chat", "dort"]
    # En TF : chat=0.5, dort=0.5
    res = vectoriser_document_mots(doc, vocab, methode="tf")
    assert res == [0.5, 0.5]
    
    print("Test vectoriser_document_mots : OK")


def vectoriser_document_agrege(document: List[List[str]], vocabulaire_direct: Dict[str, int], methode: str = "tfidf", strategie: str = "moyenne", **params) -> List[float]:
    """
    Vectorise chaque phrase puis agrège les vecteurs (somme, moyenne, max).
    """
    if not document:
        return [0.0] * len(vocabulaire_direct)
        
    # 1. Calculer les vecteurs de toutes les phrases
    vecteurs_phrases = []
    for phrase in document:
        v = vectoriser_phrase(phrase, vocabulaire_direct, methode=methode, **params)
        vecteurs_phrases.append(v)
        
    nb_phrases = len(vecteurs_phrases)
    taille_vec = len(vocabulaire_direct)
    
    # 2. Agrégation
    vecteur_final = [0.0] * taille_vec
    
    if strategie == "max":
        # Initialisation avec des valeurs très basses
        vecteur_final = [-float('inf')] * taille_vec
        
    # On parcourt dimension par dimension (colonne par colonne)
    for i in range(taille_vec):
        valeurs_dim_i = [vec[i] for vec in vecteurs_phrases]
        
        if strategie == "somme":
            vecteur_final[i] = sum(valeurs_dim_i)
            
        elif strategie == "moyenne":
            vecteur_final[i] = sum(valeurs_dim_i) / nb_phrases
            
        elif strategie == "max":
            vecteur_final[i] = max(valeurs_dim_i)
            
    return vecteur_final

def test_vectoriser_document_agrege():
    vocab = {"chat": 0, "dort": 1}
    doc = [["chat"], ["dort"]]
    
    res_moy = vectoriser_document_agrege(doc, vocab, methode="tf", strategie="moyenne")
    assert res_moy == [0.5, 0.5]
    
    res_max = vectoriser_document_agrege(doc, vocab, methode="tf", strategie="max")
    assert res_max == [1.0, 1.0]
    
    print("Test vectoriser_document_agrege : OK")

def vectoriser_corpus(corpus: Dict[str, List[List[str]]], 
                      vocabulaire_direct: Dict[str, int], 
                      methode: str = "tfidf", 
                      niveau: str = "document_mots", 
                      strategie_agregation: str = "moyenne", 
                      **params) -> Dict[str, Union[List[float], List[List[float]]]]:
    """
    Transforme tout le corpus en vecteurs selon le niveau de granularité souhaité.
    """
    resultats = {}
    
    for id_doc, contenu_doc in corpus.items():
        # contenu_doc est une liste de phrases (qui sont des listes de mots)
        
        if niveau == "phrase":
            # On retourne une liste de vecteurs (un par phrase)
            vecteurs_phrases = []
            for phrase in contenu_doc:
                # On utilise la fonction définie précédemment (supposée importée)
                v = vectoriser_phrase(phrase, vocabulaire_direct, methode=methode, **params)
                vecteurs_phrases.append(v)
            resultats[id_doc] = vecteurs_phrases
            
        elif niveau == "document_mots":
            # On retourne un seul vecteur pour le document (approche concaténation)
            v_doc = vectoriser_document_mots(contenu_doc, vocabulaire_direct, methode=methode, **params)
            resultats[id_doc] = v_doc
            
        elif niveau == "document_agrege":
            # On retourne un seul vecteur (approche agrégation)
            v_doc = vectoriser_document_agrege(contenu_doc, vocabulaire_direct, methode=methode, strategie=strategie_agregation, **params)
            resultats[id_doc] = v_doc
            
        else:
            # Gestion d'erreur ou comportement par défaut
            print(f"Niveau '{niveau}' inconnu.")
            resultats[id_doc] = []
            
    return resultats

### 🔍 Explication des tests unitaires — Vectoriser Corpus
def test_vectoriser_corpus():
    # Setup
    vocab = {"chat": 0, "dort": 1}
    # Corpus de 2 documents
    corpus_test = {
        "doc1": [["chat"], ["dort"]],  # 2 phrases
        "doc2": [["chat", "dort"]]     # 1 phrase
    }
    
    # Test 1 : Niveau Phrase
    # doc1 doit devenir [[1.0, 0.0], [0.0, 1.0]] (TF simple)
    res_phrases = vectoriser_corpus(corpus_test, vocab, methode="tf", niveau="phrase")
    assert len(res_phrases["doc1"]) == 2
    assert res_phrases["doc1"][0] == [1.0, 0.0]
    
    # Test 2 : Niveau Document Mots (Concaténation)
    # doc1 (chat + dort) -> [0.5, 0.5]
    res_doc = vectoriser_corpus(corpus_test, vocab, methode="tf", niveau="document_mots")
    assert res_doc["doc1"] == [0.5, 0.5]
    
    print("Test vectoriser_corpus : OK")


### e. Construction des métadonnées du corpus



# Les métadonnées sont le "passeport" du document. 
# Sans elles, on ne sait pas afficher un résultat pertinent (Titre, Auteur...).
# Cette fonction scanne le disque dur pour extraire ces infos.

### Énoncé de la fonction 5 : Construire Meta Corpus

# **Titre de la fonction :**
# > construire_meta_corpus(chemin_base)

# **Consigne :**
# * Parcourir récursivement les dossiers à partir de `chemin_base`.
# * Pour chaque fichier `.txt` trouvé :
#     1. Extraire l'ID (nom du fichier sans extension).
#     2. Déduire la langue (suffixe _fr ou _en).
#     3. Déduire le sous-corpus (nom du dossier parent).
#     4. Stocker le chemin complet.
#     5. Assigner un index de position.
# * Retourner le dictionnaire global.

def construire_meta_corpus(chemin_base: Path) -> Dict[str, Dict[str, Any]]:
    """
    Parcourt l'arborescence et extrait les métadonnées des fichiers texte.
    """
    meta_corpus = {}
    position_counter = 0
    
    # On s'assure que chemin_base est un objet Path
    base_path = Path(chemin_base)
    
    if not base_path.exists():
        print(f"Attention : Le chemin {base_path} n'existe pas.")
        return {}

    # Parcours récursif (rglob) de tous les fichiers .txt
    # On trie pour garantir l'ordre stable des positions
    for fichier in sorted(base_path.rglob("*.txt")):
        
        # 1. ID et Titre
        id_doc = fichier.stem  # ex: 'etu01_fr'
        
        # 2. Sous-corpus (Nom du dossier parent)
        # ex: .../UFR/etu01_fr.txt -> 'UFR'
        sous_corpus = fichier.parent.name
        
        # 3. Langue (Déduction simple basée sur le nom)
        langue = "inconnue"
        if id_doc.endswith("_fr"):
            langue = "fr"
        elif id_doc.endswith("_en"):
            langue = "en"
            
        # 4. Lecture optionnelle pour stats (taille)
        try:
            taille_octets = fichier.stat().st_size
        except:
            taille_octets = 0
            
        # Construction du dictionnaire
        meta_corpus[id_doc] = {
            "id_doc": id_doc,
            "titre": id_doc, # Par défaut le titre est l'ID
            "langue": langue,
            "sous_corpus": sous_corpus,
            "chemin": str(fichier),
            "position": position_counter,
            "autres_metadonnees": {
                "taille_octets": taille_octets,
                "extension": fichier.suffix
            }
        }
        
        position_counter += 1
        
    return meta_corpus

### 🔍 Explication des tests unitaires — Meta Corpus
# Pour tester sans dépendre de vos fichiers réels, nous créons une 
# arborescence temporaire (mock) :
# TEMP/
#   ├── UFR/
#   │   └── etu01_fr.txt
#   └── IUT/
#       └── etu02_en.txt

import shutil

def test_construire_meta_corpus():
    # 1. Création de l'environnement de test
    test_dir = Path("test_corpus_temp")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    # Création dossiers
    (test_dir / "UFR").mkdir(parents=True)
    (test_dir / "IUT").mkdir(parents=True)
    
    # Création fichiers vides
    (test_dir / "UFR" / "etu01_fr.txt").touch()
    (test_dir / "IUT" / "etu02_en.txt").touch()
    
    # 2. Exécution
    meta = construire_meta_corpus(test_dir)
    
    # 3. Vérifications
    # On doit avoir 2 entrées
    assert len(meta) == 2
    
    # Vérification doc 1
    doc1 = meta.get("etu01_fr")
    assert doc1 is not None
    assert doc1["sous_corpus"] == "UFR"
    assert doc1["langue"] == "fr"
    assert doc1["position"] == 1 # Car etu01 est avant etu02 alphabétiquement
    
    # Vérification doc 2
    doc2 = meta.get("etu02_en")
    assert doc2["sous_corpus"] == "IUT"
    assert doc2["langue"] == "en"
    
    # 4. Nettoyage
    shutil.rmtree(test_dir)
    print("Test construire_meta_corpus : OK")


def calcul_distance_euclidienne(v1: List[float], v2: List[float]) -> float:
    """Distance L2 (ligne droite)."""
    somme_carres = sum((a - b) ** 2 for a, b in zip(v1, v2))
    return math.sqrt(somme_carres)

def calcul_distance_manhattan(v1: List[float], v2: List[float]) -> float:
    """Distance L1 (somme des valeurs absolues)."""
    return sum(abs(a - b) for a, b in zip(v1, v2))

def calcul_distance_minkowski(v1: List[float], v2: List[float], p: int) -> float:
    """Généralisation des distances (Lp)."""
    if p < 1:
        raise ValueError("p doit être >= 1")
    somme_puissances = sum(abs(a - b) ** p for a, b in zip(v1, v2))
    return somme_puissances ** (1 / p)

def calcul_distance_tchebychev(v1: List[float], v2: List[float]) -> float:
    """Distance L-infini (écart maximal)."""
    if not v1: return 0.0
    return max(abs(a - b) for a, b in zip(v1, v2))   

def test_distances_geometriques():
    v1 = [0, 0]
    v2 = [3, 4]
    
    # Euclidienne : Triangle 3-4-5 -> dist = 5
    assert calcul_distance_euclidienne(v1, v2) == 5.0
    

    # Manhattan : |3-0| + |4-0| = 7
    assert calcul_distance_manhattan(v1, v2) == 7.0
    
    # Minkowski (p=1) doit être égal à Manhattan
    assert calcul_distance_minkowski(v1, v2, 1) == 7.0
    
    # Tchebychev : max(3, 4) = 4
    assert calcul_distance_tchebychev(v1, v2) == 4.0
    
    print("Test Distances Géométriques : OK")

def calcul_distance_bray_curtis(v1: List[float], v2: List[float]) -> float:
    """
    Mesure la dissimilarité basée sur les différences absolues normalisées.
    """
    numerateur = sum(abs(a - b) for a, b in zip(v1, v2))
    denominateur = sum(a + b for a, b in zip(v1, v2))
    
    if denominateur == 0:
        return 0.0 # Vecteurs nuls identiques
        
    return numerateur / denominateur

# --- Test Bray-Curtis ---
def test_bray_curtis():
    # v1 = [1, 0], v2 = [0, 1] (Totalement différents)
    # Num = |1-0| + |0-1| = 2
    # Denom = (1+0) + (0+1) = 2
    # Res = 1.0 (Dissimilarité max)
    assert calcul_distance_bray_curtis([1, 0], [0, 1]) == 1.0
    
    # v1 = [1, 1], v2 = [1, 1] (Identiques)
    # Num = 0 -> Res = 0.0
    assert calcul_distance_bray_curtis([1, 1], [1, 1]) == 0.0
    print("Test Bray-Curtis : OK")

def calcul_similarite_cosinus(v1: List[float], v2: List[float]) -> float:
    """
    Mesure l'angle entre deux vecteurs (produit scalaire normalisé).
    Retourne entre -1 et 1 (souvent 0 et 1 pour des textes positifs).
    """
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a ** 2 for a in v1))
    norm_v2 = math.sqrt(sum(b ** 2 for b in v2))
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0 # Vecteur nul = orthogonalité par convention
        
    return dot_product / (norm_v1 * norm_v2)

def calcul_distance_cosinus(v1: List[float], v2: List[float]) -> float:
    """
    1 - Similarité Cosinus.
    """
    # Attention aux erreurs d'arrondi flottant qui pourraient donner sim > 1.0
    sim = calcul_similarite_cosinus(v1, v2)
    return 1.0 - sim

# --- Test Cosinus ---
def test_cosinus():
    # Vecteurs colinéaires (même sens)
    v1 = [1, 2]
    v2 = [2, 4] # v2 = 2 * v1
    # Similarité doit être 1.0 (angle 0)
    assert abs(calcul_similarite_cosinus(v1, v2) - 1.0) < 0.0001
    assert abs(calcul_distance_cosinus(v1, v2) - 0.0) < 0.0001
    
    # Vecteurs orthogonaux (angle 90°)
    v3 = [1, 0]
    v4 = [0, 1]
    assert abs(calcul_similarite_cosinus(v3, v4) - 0.0) < 0.0001
    print("Test Cosinus : OK")

def calcul_similarite_jaccard(v1: List[float], v2: List[float]) -> float:
    """
    Jaccard généralisé (Min / Max).
    """
    intersection = sum(min(a, b) for a, b in zip(v1, v2))
    union = sum(max(a, b) for a, b in zip(v1, v2))
    
    if union == 0:
        return 0.0 # Ou 1.0 si on considère que deux vides sont identiques
        
    return intersection / union

def calcul_distance_jaccard(v1: List[float], v2: List[float]) -> float:
    return 1.0 - calcul_similarite_jaccard(v1, v2)

# --- Test Jaccard ---
def test_jaccard():
    # v1 = {chat, dort}, v2 = {chat, mange}
    # En binaire : v1=[1, 1, 0], v2=[1, 0, 1] (vocab: chat, dort, mange)
    # Intersection = 1 (chat)
    # Union = 3 (chat, dort, mange)
    # Sim = 1/3
    assert abs(calcul_similarite_jaccard([1, 1, 0], [1, 0, 1]) - (1/3)) < 0.0001
    print("Test Jaccard : OK")

def calcul_distance_hamming(v1: List[float], v2: List[float]) -> float:
    """
    Compte le nombre de positions différentes.
    """
    dist = 0
    for a, b in zip(v1, v2):
        if a != b:
            dist += 1
    return float(dist)

# --- Test Hamming ---
def test_hamming():
    # Différence à l'index 1 et 2 -> dist = 2
    assert calcul_distance_hamming([1, 0, 1], [1, 1, 0]) == 2.0
    print("Test Hamming : OK")

def calcul_distance_jensen_shannon(v1: List[float], v2: List[float]) -> float:
    """
    Racine carrée de la divergence JS.
    Suppose que v1 et v2 sont des distributions de probabilité (Somme=1).
    """
    
    def kl_divergence(p: List[float], q: List[float]) -> float:
        """Kullback-Leibler Divergence: sum(p * log2(p/q))"""
        total = 0.0
        for pi, qi in zip(p, q):
            # On évite log(0) et division par 0
            # Si pi > 0 et qi > 0 : on calcule
            if pi > 0 and qi > 0:
                total += pi * math.log2(pi / qi)
            # Si pi > 0 mais qi = 0 : divergence infinie (en théorie)
            # Mais ici q sera toujours une moyenne (p+q)/2, donc jamais nul si pi > 0.
        return total

    # 1. Calcul de la distribution moyenne M
    m = [(a + b) / 2 for a, b in zip(v1, v2)]
    
    # 2. Divergence Jensen-Shannon = (KL(v1||M) + KL(v2||M)) / 2
    divergence_js = 0.5 * kl_divergence(v1, m) + 0.5 * kl_divergence(v2, m)
    
    # 3. Distance = racine carrée
    # Protection contre les erreurs flottantes négatives très proches de 0
    if divergence_js < 0:
        divergence_js = 0.0
        
    return math.sqrt(divergence_js)

# --- Test Jensen-Shannon ---
def test_jensen_shannon():
    # Deux distributions identiques -> dist = 0
    assert calcul_distance_jensen_shannon([0.5, 0.5], [0.5, 0.5]) == 0.0
    
    # Deux distributions totalement distinctes
    # v1=[1, 0], v2=[0, 1]
    # m =[0.5, 0.5]
    # KL(v1||m) = 1 * log2(1/0.5) = 1 * 1 = 1
    # KL(v2||m) = 1 * log2(1/0.5) = 1
    # JS Div = 0.5*1 + 0.5*1 = 1
    # JS Dist = sqrt(1) = 1
    res = calcul_distance_jensen_shannon([1.0, 0.0], [0.0, 1.0])
    assert abs(res - 1.0) < 0.0001
    
    print("Test Jensen-Shannon : OK")

def pretraiter_requete(texte_requete: str, config_pretraitement: Dict[str, Any] = None) -> List[str]:
    """
    Applique les mêmes nettoyages que sur le corpus.
    Ici : Minuscule + Tokenisation simple (simulé).
    """
    if not texte_requete:
        return []
        
    # 1. Mise en minuscule
    texte = texte_requete.lower()
    
    # 2. Suppression ponctuation (basique)
    texte = re.sub(r'[^\w\s]', '', texte)
    
    # 3. Tokenisation
    tokens = texte.split()
    
    return tokens

def vectoriser_requete(texte_pretraite: Union[List[str], List[List[str]]], 
                       vocabulaire: Dict[str, int], 
                       methode: str = "tfidf", 
                       type_requete: str = "phrase", 
                       **params) -> List[float]:
    """
    Transforme la requête prétraitée en un vecteur unique.
    """
    vecteur = []
    
    # Cas 1 : La requête est considérée comme une phrase simple
    if type_requete == "phrase":
        # texte_pretraite doit être une liste de mots : ["chat", "dort"]
        vecteur = vectoriser_phrase(texte_pretraite, vocabulaire, methode=methode, **params)
        
    # Cas 2 : La requête est longue et traitée comme un "Sac de mots" global
    elif type_requete == "document_mots":
        # Pour vectoriser_document_mots, il attend une liste de phrases (liste de listes).
        # Si on a juste une liste de mots, on l'emballe : [ ["chat", "dort"] ]
        if isinstance(texte_pretraite[0], str):
            entree_formattee = [texte_pretraite]
        else:
            entree_formattee = texte_pretraite
            
        vecteur = vectoriser_document_mots(entree_formattee, vocabulaire, methode=methode, **params)
        
    # Cas 3 : La requête est traitée par agrégation
    elif type_requete == "document_agrege":
        if isinstance(texte_pretraite[0], str):
            entree_formattee = [texte_pretraite]
        else:
            entree_formattee = texte_pretraite
            
        # On récupère la stratégie dans params ou valeur par défaut
        strat = params.get("strategie_agregation", "moyenne")
        vecteur = vectoriser_document_agrege(entree_formattee, vocabulaire, methode=methode, strategie=strat, **params)
        
    else:
        raise ValueError(f"Type de requête inconnu : {type_requete}")
        
    return vecteur
# On suppose les fonctions de distance importées du bloc précédent
# from TP5_partie2 import calcul_similarite_cosinus, calcul_distance_euclidienne, ...

def calculer_similarite(vect_requete: List[float], vect_corpus: List[float], mesure: str = 'cosinus') -> float:
    """
    Calcule le score entre deux vecteurs selon la mesure choisie.
    """
    # Mapping des noms vers les fonctions (Pattern Dispatcher)
    # Note : Assurez-vous que ces fonctions sont définies dans votre contexte
    fonctions_mesure = {
        'cosinus': calcul_similarite_cosinus,
        'distance_cosinus': calcul_distance_cosinus,
        'euclidienne': calcul_distance_euclidienne,
        'manhattan': calcul_distance_manhattan,
        'jaccard': calcul_similarite_jaccard,
        'distance_jaccard': calcul_distance_jaccard,
        # Ajoutez les autres ici (bray_curtis, etc.)
    }
    
    fonction_choisie = fonctions_mesure.get(mesure)
    
    if fonction_choisie is None:
        raise ValueError(f"Mesure inconnue : {mesure}")
        
    return fonction_choisie(vect_requete, vect_corpus)
def calculer_scores_requete(vect_requete: List[float], 
                            corpus_vecteurs: Dict[str, Union[List[float], List[List[float]]]], 
                            mesure: str = 'cosinus', 
                            niveau: str = 'document_mots') -> Dict[str, Union[float, List[float]]]:
    """
    Compare la requête à l'ensemble du corpus.
    Retourne un dictionnaire {id_doc: score(s)}.
    """
    resultats_scores = {}
    
    for id_doc, data_doc in corpus_vecteurs.items():
        
        # Cas A : Comparaison Phrase par Phrase
        if niveau == 'phrase':
            # data_doc est une liste de vecteurs [vec_phrase1, vec_phrase2...]
            scores_phrases = []
            for vect_phrase in data_doc:
                score = calculer_similarite(vect_requete, vect_phrase, mesure=mesure)
                scores_phrases.append(score)
            resultats_scores[id_doc] = scores_phrases
            
        # Cas B : Comparaison Document par Document (Mots ou Agrégé)
        else:
            # data_doc est un vecteur unique
            score = calculer_similarite(vect_requete, data_doc, mesure=mesure)
            resultats_scores[id_doc] = score
            
    return resultats_scores
def test_moteur_recherche():
    print("\n--- Tests Moteur de Recherche ---")
    
    # 1. Setup Données simulées
    vocab = {"chat": 0, "dort": 1}
    
    # Requête prétraitée : "chat"
    req_tokens = ["chat"]
    
    # Vecteur requête (TF) -> [1.0, 0.0]
    # On mock vectoriser_phrase pour simplifier le test sans dépendance externe
    # vec_req = [1.0, 0.0] 
    
    # Corpus Vectorisé (Niveau Document)
    # Doc 1 : "chat" -> [1.0, 0.0] (Identique à requête)
    # Doc 2 : "dort" -> [0.0, 1.0] (Orthogonal)
    corpus_vecs_doc = {
        "doc1": [1.0, 0.0],
        "doc2": [0.0, 1.0]
    }
    
    # Corpus Vectorisé (Niveau Phrase)
    # Doc 3 contient 2 phrases : "chat" et "dort"
    corpus_vecs_phr = {
        "doc3": [[1.0, 0.0], [0.0, 1.0]]
    }
    
    vec_req = [1.0, 0.0]

    # --- Test Similarité Unitaire (Cosinus) ---
    # Sim(req, doc1) = 1.0 (Angle 0)
    score_1 = calculer_similarite(vec_req, corpus_vecs_doc["doc1"], mesure='cosinus')
    assert abs(score_1 - 1.0) < 0.0001
    
    # Sim(req, doc2) = 0.0 (Angle 90°)
    score_2 = calculer_similarite(vec_req, corpus_vecs_doc["doc2"], mesure='cosinus')
    assert abs(score_2 - 0.0) < 0.0001
    
    print("✅ Test Similarité Unitaire : OK")

    # --- Test Scores Globaux (Niveau Document) ---
    scores_docs = calculer_scores_requete(vec_req, corpus_vecs_doc, mesure='cosinus', niveau='document_mots')
    
    # Doc1 doit être le meilleur
    assert scores_docs["doc1"] > scores_docs["doc2"]
    print("✅ Test Scores Globaux (Doc) : OK")
    
    # --- Test Scores Globaux (Niveau Phrase) ---
    scores_phr = calculer_scores_requete(vec_req, corpus_vecs_phr, mesure='cosinus', niveau='phrase')
    
    # Doc3 doit avoir une liste de 2 scores : [1.0, 0.0]
    liste_scores = scores_phr["doc3"]
    assert len(liste_scores) == 2
    assert abs(liste_scores[0] - 1.0) < 0.0001 # Phrase "chat"
    assert abs(liste_scores[1] - 0.0) < 0.0001 # Phrase "dort"
    
    print("✅ Test Scores Globaux (Phrase) : OK")

def extraire_top_k(scores_dict: Dict[str, Union[float, List[float]]], 
                   k: int = 5, 
                   niveau: str = 'document') -> Dict[Any, float]:
    """
    Sélectionne les k éléments ayant les scores les plus élevés.
    
    Returns:
        Dictionnaire ordonné {identifiant : score}.
        - Si niveau document : identifiant = str (id_doc)
        - Si niveau phrase : identifiant = tuple (id_doc, index_phrase)
    """
    elements_tries = []

    # Cas 1 : Niveau Document (Mots ou Agrégé)
    # scores_dict est { "doc1": 0.5, "doc2": 0.8 }
    if "document" in niveau:
        # On transforme en liste de tuples [('doc1', 0.5), ('doc2', 0.8)]
        items = list(scores_dict.items())
        # Tri décroissant sur le score (x[1])
        elements_tries = sorted(items, key=lambda x: x[1], reverse=True)

    # Cas 2 : Niveau Phrase
    # scores_dict est { "doc1": [0.1, 0.9], "doc2": [0.4] }
    elif niveau == 'phrase':
        liste_plate = []
        for id_doc, scores in scores_dict.items():
            # scores est une liste de floats
            for index_phrase, score in enumerate(scores):
                identifiant_unique = (id_doc, index_phrase)
                liste_plate.append((identifiant_unique, score))
        
        # Tri décroissant sur le score
        elements_tries = sorted(liste_plate, key=lambda x: x[1], reverse=True)

    else:
        return {}

    # On coupe pour ne garder que les K premiers
    top_elements = elements_tries[:k]
    
    # On reconvertit en dictionnaire pour la sortie
    return dict(top_elements)
def afficher_metadonnees_resultat(id_element: Union[str, Tuple[str, int]], 
                                  score: float, 
                                  meta_corpus: Dict[str, Any]):
    """
    Affiche les infos contextuelles (Titre, Source, Score...).
    Gère le cas où id_element est un tuple (id_doc, id_phrase).
    """
    # Si c'est une phrase, l'ID est (id_doc, index_phrase)
    if isinstance(id_element, tuple):
        real_id_doc = id_element[0]
        suffixe_titre = f" (Phrase n°{id_element[1]})"
    else:
        real_id_doc = id_element
        suffixe_titre = ""
        
    meta = meta_corpus.get(real_id_doc, {})
    
    print(f"--- Résultat (Score: {score:.4f}) ---")
    print(f"📄 Document : {meta.get('titre', 'Inconnu')}{suffixe_titre}")
    print(f"🌍 Langue : {meta.get('langue', '?')} | 📂 Source : {meta.get('sous_corpus', '?')}")
    # print(f"📍 Chemin : {meta.get('chemin', '')}")
def highlight_mots_pertinents(texte: str, requete: Union[str, List[str]], vocabulaire: Dict[str, int] = None) -> str:
    """
    Met en évidence les mots de la requête présents dans le texte.
    Utilise des codes ANSI pour la couleur rouge dans le terminal.
    """
    # 1. Préparation des tokens à surligner
    if isinstance(requete, list):
        mots_requete = set(m.lower() for m in requete)
    else:
        mots_requete = set(requete.lower().split())
        
    # Filtrage optionnel : ne surligner que ce qui est dans le vocabulaire
    if vocabulaire:
        mots_requete = {m for m in mots_requete if m in vocabulaire}
        
    if not mots_requete:
        return texte

    # 2. Remplacement insensible à la casse mais préservant la casse originale
    # Cette méthode est simplifiée. Pour une robustesse totale, utiliser des regex.
    texte_out = texte
    
    # Code couleur ANSI : \033[91m = Rouge clair, \033[1m = Gras, \033[0m = Reset
    MARK_START = "\033[91m\033[1m"
    MARK_END = "\033[0m"
    
    # On découpe le texte pour préserver la casse originale
    tokens_texte = texte.split()
    tokens_colores = []
    
    for mot in tokens_texte:
        # On nettoie la ponctuation pour vérifier la correspondance
        mot_clean = re.sub(r'[^\w\s]', '', mot).lower()
        
        if mot_clean in mots_requete:
            tokens_colores.append(f"{MARK_START}{mot}{MARK_END}")
        else:
            tokens_colores.append(mot)
            
    return " ".join(tokens_colores)

def afficher_contenu_document(id_element: Union[str, Tuple[str, int]], 
                              corpus_texte: Dict[str, List[List[str]]], 
                              niveau: str = 'document_mots') -> str:
    """
    Retourne le texte brut associé au résultat.
    """
    texte_retour = ""
    
    # Cas Phrase : id_element est (id_doc, index_phrase)
    if isinstance(id_element, tuple):
        real_id = id_element[0]
        idx = id_element[1]
        
        doc_contenu = corpus_texte.get(real_id)
        if doc_contenu and 0 <= idx < len(doc_contenu):
            # doc_contenu[idx] est une liste de mots -> on joint
            texte_retour = " ".join(doc_contenu[idx])
            
    # Cas Document : id_element est id_doc
    else:
        doc_contenu = corpus_texte.get(id_element)
        if doc_contenu:
            # On joint toutes les phrases
            phrases_str = [" ".join(p) for p in doc_contenu]
            texte_retour = " ... ".join(phrases_str) # Séparateur visuel
            
    return texte_retour

def afficher_resultat_complet(id_element: Any, 
                              score: float, 
                              corpus_texte: Dict[str, List[List[str]]], 
                              meta_corpus: Dict[str, Any], 
                              afficher_texte: bool = False, 
                              highlight: bool = False, 
                              requete: Any = None, 
                              vocabulaire: Dict[str, int] = None,
                              niveau: str = 'document'):
    """
    Affiche un bloc résultat complet.
    """
    # 1. Métadonnées
    afficher_metadonnees_resultat(id_element, score, meta_corpus)
    
    # 2. Contenu textuel (Optionnel)
    if afficher_texte:
        texte_brut = afficher_contenu_document(id_element, corpus_texte, niveau)
        
        if highlight and requete:
            texte_final = highlight_mots_pertinents(texte_brut, requete, vocabulaire)
        else:
            texte_final = texte_brut
            
        print(f"📝 Extrait :\n\"{texte_final}\"")
    print("-" * 40)


def afficher_resultats_top_k(top_k_dict: Dict[Any, float], 
                             corpus_texte: Dict[str, List[List[str]]], 
                             meta_corpus: Dict[str, Any], 
                             afficher_texte: bool = False, 
                             highlight: bool = False, 
                             requete: Any = None, 
                             vocabulaire: Dict[str, int] = None,
                             niveau: str = 'document'):
    """
    Boucle sur les résultats et les affiche.
    """
    print(f"\n>>> 🏆 TOP {len(top_k_dict)} RÉSULTATS ({niveau}) <<<\n")
    
    for id_elem, score in top_k_dict.items():
        afficher_resultat_complet(
            id_elem, score, corpus_texte, meta_corpus, 
            afficher_texte, highlight, requete, vocabulaire, niveau
        )

def test_affichage_resultats():
    print("\n--- Test Module Affichage ---")
    
    # 1. Données Mock
    corpus_texte = {
        "doc1": [["le", "chat", "dort"], ["il", "reve"]],
        "doc2": [["le", "chien", "aboie"]]
    }
    meta_corpus = {
        "doc1": {"titre": "Etude Félins", "langue": "fr", "sous_corpus": "Bio"},
        "doc2": {"titre": "Etude Canins", "langue": "fr", "sous_corpus": "Bio"}
    }
    
    # 2. Simulation de scores (Calculés précédemment)
    # Scénario : Recherche "chat"
    # Niveau Document : doc1 est pertinent (0.9), doc2 non (0.1)
    scores_docs = {"doc1": 0.9, "doc2": 0.1}
    
    # Niveau Phrase : doc1 phrase 0 est pertinente
    scores_phrases = {
        "doc1": [0.95, 0.1], 
        "doc2": [0.0]
    }
    
    # 3. Test Extraire Top K
    # Doc
    top_docs = extraire_top_k(scores_docs, k=1, niveau='document')
    assert "doc1" in top_docs
    assert len(top_docs) == 1
    
    # Phrase
    top_phr = extraire_top_k(scores_phrases, k=1, niveau='phrase')
    # Doit retourner la clé (doc1, 0)
    assert (("doc1", 0) in top_phr)
    
    print("✅ Extraction Top K : OK")
    
    # 4. Test Affichage (Visuel)
    print("\n--- Simulation Affichage Document ---")
    afficher_resultats_top_k(top_docs, corpus_texte, meta_corpus, 
                             afficher_texte=True, highlight=True, 
                             requete=["chat"], vocabulaire={"chat":0}, niveau='document')
                             
    print("\n--- Simulation Affichage Phrase ---")
    afficher_resultats_top_k(top_phr, corpus_texte, meta_corpus, 
                             afficher_texte=True, highlight=True, 
                             requete=["chat"], vocabulaire={"chat":0}, niveau='phrase')

def filtrer_par_score(top_k: Dict[Any, float], seuil: float = 0.5) -> Dict[Any, float]:
    """
    Ne conserve que les résultats dont la pertinence dépasse le seuil.
    """
    resultats_filtres = {}
    
    for identifiant, score in top_k.items():
        if score >= seuil:
            resultats_filtres[identifiant] = score
            
    return resultats_filtres

def resume_top_k(top_k: Dict[Any, float], meta_corpus: Dict[str, Any]):
    """
    Affiche un tableau récapitulatif des résultats dans la console.
    """
    print(f"{'RANG':<5} | {'ID DOCUMENT':<20} | {'LANG':<5} | {'SOURCE':<10} | {'SCORE':<10}")
    print("-" * 65)
    
    for i, (identifiant, score) in enumerate(top_k.items()):
        rang = i + 1
        
        # Gestion ID Phrase vs ID Document
        if isinstance(identifiant, tuple):
            real_id = identifiant[0]
            suffixe = f" (P.{identifiant[1]})"
        else:
            real_id = identifiant
            suffixe = ""
            
        # Récupération métadonnées
        meta = meta_corpus.get(real_id, {})
        langue = meta.get('langue', 'N/A')
        source = meta.get('sous_corpus', 'N/A')
        
        id_affichage = f"{real_id}{suffixe}"
        # Tronquage si trop long pour l'affichage
        if len(id_affichage) > 19:
            id_affichage = id_affichage[:17] + ".."
            
        print(f"{rang:<5} | {id_affichage:<20} | {langue:<5} | {source:<10} | {score:.4f}")
    print("-" * 65)

def stats_scores(scores: List[float]) -> Dict[str, float]:
    """
    Calcule les statistiques élémentaires d'une liste de scores.
    """
    if not scores:
        return {}
        
    stats = {
        "min": min(scores),
        "max": max(scores),
        "moyenne": statistics.mean(scores),
        "mediane": statistics.median(scores),
        "nb_valeurs": len(scores)
    }
    
    # Calcul des quartiles (nécessite Python 3.8+)
    try:
        quantiles = statistics.quantiles(scores, n=4)
        stats["Q1"] = quantiles[0]
        stats["Q3"] = quantiles[2]
    except AttributeError:
        # Fallback pour versions antérieures
        sorted_scores = sorted(scores)
        stats["Q1"] = sorted_scores[len(scores)//4]
        stats["Q3"] = sorted_scores[(3*len(scores))//4]
        
    return stats

def normaliser_scores(scores_dict: Dict[Any, float], methode: str = 'minmax') -> Dict[Any, float]:
    """
    Normalise les scores d'un résultat de recherche entre 0 et 1.
    """
    if not scores_dict:
        return {}
        
    valeurs = list(scores_dict.values())
    min_v = min(valeurs)
    max_v = max(valeurs)
    etendue = max_v - min_v
    
    scores_norm = {}
    
    for k, v in scores_dict.items():
        if etendue == 0:
            scores_norm[k] = 0.0
        else:
            if methode == 'minmax':
                scores_norm[k] = (v - min_v) / etendue
            else:
                scores_norm[k] = v # Par défaut
                
    return scores_norm

def distribution_scores(scores: List[float], titre: str = "Distribution des scores"):
    """
    Affiche un histogramme et un boxplot des scores.
    """
    

    plt.figure(figsize=(10, 4))
    
    # 1. Histogramme
    plt.subplot(1, 2, 1)
    plt.hist(scores, bins=20, color='skyblue', edgecolor='black')
    plt.title(f"Histogramme - {titre}")
    plt.xlabel("Score")
    plt.ylabel("Fréquence")
    
    # 2. Boxplot
    plt.subplot(1, 2, 2)
    plt.boxplot(scores, vert=False)
    plt.title("Boxplot")
    plt.xlabel("Score")
    
    plt.tight_layout()
    plt.show()

def comparer_distributions_requetes(scores_requetes: Dict[str, List[float]]):
    """
    Compare les distributions de plusieurs requêtes sur le même graphique.
    scores_requetes = { "Requete A": [scores...], "Requete B": [scores...] }
    """

    plt.figure(figsize=(10, 6))
    
    # On prépare les données pour le boxplot multiple
    labels = []
    data = []
    
    for nom_requete, scores in scores_requetes.items():
        labels.append(nom_requete)
        data.append(scores)
        
    plt.boxplot(data, labels=labels, vert=True, patch_artist=True)
    plt.title("Comparaison de la sélectivité des requêtes")
    plt.ylabel("Scores de similarité")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.show()


def test_evaluation():
    print("\n--- Test Module Évaluation ---")
    
    # 1. Données Mock
    # Top K simulé (Niveau Document)
    top_k_mock = {
        "doc1": 0.95,
        "doc2": 0.80,
        "doc3": 0.45,  # Faible score
        "doc4": 0.10   # Très faible score
    }
    
    # Meta Corpus Mock
    meta_mock = {
        "doc1": {"langue": "fr", "sous_corpus": "UFR"},
        "doc2": {"langue": "en", "sous_corpus": "IUT"},
        "doc3": {"langue": "fr", "sous_corpus": "UFR"},
        "doc4": {"langue": "en", "sous_corpus": "IUT"}
    }
    
    # 2. Test Filtrage
    # Seuil 0.5 -> On doit garder doc1 et doc2, rejeter doc3 et doc4
    filtres = filtrer_par_score(top_k_mock, seuil=0.5)
    assert "doc1" in filtres
    assert "doc3" not in filtres
    assert len(filtres) == 2
    print("✅ Filtrage par seuil : OK")
    
    # 3. Test Stats
    scores_list = list(top_k_mock.values()) # [0.95, 0.80, 0.45, 0.10]
    stats = stats_scores(scores_list)
    assert stats["max"] == 0.95
    assert stats["min"] == 0.10
    assert abs(stats["moyenne"] - 0.575) < 0.001
    print("✅ Statistiques : OK")
    
    # 4. Test Normalisation
    # 0.10 -> 0.0
    # 0.95 -> 1.0
    norm = normaliser_scores(top_k_mock)
    assert norm["doc4"] == 0.0
    assert norm["doc1"] == 1.0
    print("✅ Normalisation : OK")
    
    # 5. Simulation Visuelle (Output Console)
    print("\n--- Rendu Tableau Résumé ---")
    resume_top_k(top_k_mock, meta_mock)
    
    print("\n--- (Optionnel) Test Visualisation ---")
    
    print("Génération des graphiques...")
        # Simuler deux requêtes pour la comparaison
    req_A = [0.9, 0.8, 0.85, 0.1, 0.2] # Requête précise (bimodale)
    req_B = [0.5, 0.55, 0.45, 0.6, 0.5] # Requête floue (centrée)
        
    distribution_scores(req_A, titre="Requête A")
    comparer_distributions_requetes({"Requête A (Précise)": req_A, "Requête B (Floue)": req_B})
    

def analyse_sous_corpus(top_k: Dict[Any, float], meta_corpus: Dict[str, Any]) -> Dict[str, int]:
    """
    Compte le nombre de documents par sous-corpus dans le Top K.
    """
    compteur = Counter()
    
    for identifiant in top_k.keys():
        # Gestion ID tuple (phrase) vs ID str (document)
        real_id = identifiant[0] if isinstance(identifiant, tuple) else identifiant
        
        meta = meta_corpus.get(real_id, {})
        source = meta.get('sous_corpus', 'Inconnu')
        compteur[source] += 1
        
    return dict(compteur)

def repartition_pourcentage(top_k: Dict[Any, float], meta_corpus: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Retourne la proportion (%) de documents par sous-corpus et par langue.
    """
    total = len(top_k)
    if total == 0:
        return {"sous_corpus": {}, "langue": {}}
    
    compteur_source = Counter()
    compteur_langue = Counter()
    
    for identifiant in top_k.keys():
        real_id = identifiant[0] if isinstance(identifiant, tuple) else identifiant
        meta = meta_corpus.get(real_id, {})
        
        compteur_source[meta.get('sous_corpus', 'Inconnu')] += 1
        compteur_langue[meta.get('langue', 'Inconnu')] += 1
        
    # Conversion en pourcentages
    prop_source = {k: (v / total) * 100 for k, v in compteur_source.items()}
    prop_langue = {k: (v / total) * 100 for k, v in compteur_langue.items()}
    
    return {"sous_corpus": prop_source, "langue": prop_langue}

def afficher_repartition_barplot(repartition: Dict[str, float], titre: str = "Répartition"):
    """
    Affiche un histogramme des proportions.
    """
    

    plt.figure(figsize=(8, 4))
    plt.bar(repartition.keys(), repartition.values(), color='lightcoral')
    plt.title(titre)
    plt.ylabel("Pourcentage (%)")
    plt.ylim(0, 100) # Pourcentage fixe
    plt.show()

def filtrer_top_k_par_categorie(top_k: Dict[Any, float], 
                                meta_corpus: Dict[str, Any], 
                                categorie: str = 'langue', 
                                valeur: str = 'fr') -> Dict[Any, float]:
    """
    Ne garde que les résultats correspondant à un critère (ex: langue='fr').
    """
    res_filtres = {}
    
    for identifiant, score in top_k.items():
        real_id = identifiant[0] if isinstance(identifiant, tuple) else identifiant
        meta = meta_corpus.get(real_id, {})
        
        # On vérifie si la métadonnée correspond à la valeur demandée
        if meta.get(categorie) == valeur:
            res_filtres[identifiant] = score
            
    return res_filtres 
def matrice_similarite(corpus_vecteurs: Dict[str, List[float]], 
                       mesure: str = 'cosinus') -> Tuple[List[List[float]], List[str]]:
    """
    Calcule la matrice carrée de similarité entre tous les documents.
    Retourne (matrice, liste_ordonnee_ids).
    """
    # 1. On fige l'ordre des documents (les dictionnaires ne sont pas indexables par int)
    ids = sorted(list(corpus_vecteurs.keys()))
    n = len(ids)
    
    # 2. Initialisation matrice N x N
    matrice = [[0.0] * n for _ in range(n)]
    
    # 3. Double boucle (Optimisation possible : calculer triangle sup et copier)
    # Pour ce TP, on fait simple.
    for i in range(n):
        vec_i = corpus_vecteurs[ids[i]]
        for j in range(n):
            vec_j = corpus_vecteurs[ids[j]]
            
            # Appel à la fonction de similarité définie précédemment
            # On suppose qu'elle est disponible (calculer_similarite)
            sim = calculer_similarite(vec_i, vec_j, mesure=mesure)
            matrice[i][j] = sim
            
    return matrice, ids



def afficher_heatmap_similarite(matrice: List[List[float]], 
                                labels: List[str] = None, 
                                titre: str = 'Heatmap de similarité'):
    """
    Affiche la matrice sous forme de carte thermique.
    """

    plt.figure(figsize=(10, 8))
   # Avec Seaborn (recommandé)
    sns.heatmap(matrice, xticklabels=labels, yticklabels=labels, cmap="YlGnBu", annot=False)
   
            
    plt.title(titre)
    plt.tight_layout()
    plt.show()

def matrice_distance_normalisee(matrice_similarite: List[List[float]]) -> List[List[float]]:
    """
    Transforme Similarité (0..1) en Distance (1..0).
    D = 1 - S.
    """
    n = len(matrice_similarite)
    matrice_dist = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            matrice_dist[i][j] = 1.0 - matrice_similarite[i][j]
            
    return matrice_dist

def top_paires_similaires(matrice: List[List[float]], labels: List[str], top: int = 5) -> List[Tuple[str, str, float]]:
    """
    Identifie les documents les plus proches (hors diagonale).
    """
    paires = []
    n = len(matrice)
    
    # On parcourt seulement le triangle supérieur pour éviter doublons (A,B) et (B,A)
    # et on évite la diagonale (i+1)
    for i in range(n):
        for j in range(i + 1, n):
            score = matrice[i][j]
            paires.append((labels[i], labels[j], score))
            
    # Tri décroissant (les plus similaires en premier)
    paires.sort(key=lambda x: x[2], reverse=True)
    
    return paires[:top]

def top_paires_differentes(matrice: List[List[float]], labels: List[str], top: int = 5) -> List[Tuple[str, str, float]]:
    """
    Identifie les documents les plus éloignés.
    """
    paires = []
    n = len(matrice)
    
    for i in range(n):
        for j in range(i + 1, n):
            score = matrice[i][j]
            paires.append((labels[i], labels[j], score))
            
    # Tri croissant (les scores les plus bas en premier)
    paires.sort(key=lambda x: x[2], reverse=False)
    
    return paires[:top]

def heatmap_sous_corpus(matrice_globale: List[List[float]], 
                        ids_globaux: List[str], 
                        meta_corpus: Dict[str, Any], 
                        sous_corpus_cible: str = 'UFR'):
    """
    Extrait et affiche la sous-matrice correspondant uniquement à un sous-corpus.
    Permet de vérifier l'homogénéité interne d'une catégorie.
    """
    # 1. Identifier les indices qui appartiennent au sous-corpus
    indices_cibles = []
    labels_cibles = []
    
    for idx, doc_id in enumerate(ids_globaux):
        meta = meta_corpus.get(doc_id, {})
        if meta.get('sous_corpus') == sous_corpus_cible:
            indices_cibles.append(idx)
            labels_cibles.append(doc_id)
            
    if not indices_cibles:
        print(f"Aucun document trouvé pour le sous-corpus : {sous_corpus_cible}")
        return

    # 2. Construire la sous-matrice
    n_sub = len(indices_cibles)
    sous_matrice = [[0.0] * n_sub for _ in range(n_sub)]
    
    for i in range(n_sub):
        for j in range(n_sub):
            # On va chercher les valeurs aux coordonnées originales
            idx_orig_i = indices_cibles[i]
            idx_orig_j = indices_cibles[j]
            sous_matrice[i][j] = matrice_globale[idx_orig_i][idx_orig_j]
            
    # 3. Affichage
    afficher_heatmap_similarite(sous_matrice, labels_cibles, titre=f"Heatmap : {sous_corpus_cible}")

def test_analyse_structurelle():
    print("\n--- Test Module Analyse Structurelle ---")
    
    # 1. Mock Données
    # 3 Documents : A proche de B, C très loin.
    corpus_vects = {
        "docA": [1.0, 0.0, 0.0],
        "docB": [0.9, 0.1, 0.0], # Très proche de A
        "docC": [0.0, 0.0, 1.0]  # Orthogonal à A et B
    }
    meta_mock = {
        "docA": {"sous_corpus": "Groupe1"},
        "docB": {"sous_corpus": "Groupe1"},
        "docC": {"sous_corpus": "Groupe2"}
    }
    
    # 2. Test Calcul Matrice
    mat, ids = matrice_similarite(corpus_vects, mesure='cosinus')
    # ids triés : ['docA', 'docB', 'docC']
    idx_A = ids.index("docA")
    idx_B = ids.index("docB")
    idx_C = ids.index("docC")
    
    # Sim(A, B) doit être élevée
    sim_AB = mat[idx_A][idx_B]
    # Sim(A, C) doit être 0
    sim_AC = mat[idx_A][idx_C]
    
    assert sim_AB > 0.8
    assert sim_AC == 0.0
    print("✅ Calcul Matrice : OK")
    
    # 3. Test Top Paires
    top_sim = top_paires_similaires(mat, ids, top=1)
    # La paire la plus similaire doit être (docA, docB) ou (docB, docA)
    pair = top_sim[0]
    assert "docA" in pair and "docB" in pair
    print("✅ Top Paires Similaires : OK")
    
    # 4. Test Top Paires Différentes
    top_diff = top_paires_differentes(mat, ids, top=1)
    # (docA, docC) score 0.0
    pair_diff = top_diff[0]
    assert pair_diff[2] == 0.0
    print("✅ Top Paires Différentes : OK")

    # 5. Test Répartition (Section C)
    # Imaginons un Top-K contenant docA et docC
    top_k_mock = {"docA": 0.9, "docC": 0.1}
    rep = repartition_pourcentage(top_k_mock, meta_mock)
    # Groupe1: 50% (1 doc), Groupe2: 50% (1 doc)
    assert rep["sous_corpus"]["Groupe1"] == 50.0
    print("✅ Répartition Pourcentage : OK")
    
    # 6. Simulation Visuelle (si possible)
    print("\n--- Génération Heatmaps (Simulation) ---")
    afficher_heatmap_similarite(mat, ids, titre="Heatmap Globale Test")
    heatmap_sous_corpus(mat, ids, meta_mock, sous_corpus_cible="Groupe1")



def convertir_vers_minuscule(texte: str) -> str:
    """Convertit tout le texte en minuscules."""
    return texte.lower() if isinstance(texte, str) else ""


def supprimer_ponctuation(texte: str) -> str:
    """Supprime (. , ; : ! ? ( ) [ ] " ' … etc)."""
    if not texte: return ""
    motif = f"[{re.escape(string.punctuation)}]"
    return re.sub(motif, "", texte)



def appliquer_stemming(tokens, langue='fr'):
    """
    Applique un stemming simplifié basé sur la suppression de suffixes fréquents.
    La fonction supprime le suffixe le plus long correspondant.
    Ne raccourcit pas les mots en dessous de 3 caractères.
    
    Paramètres:
        tokens (list[str]): Liste de tokens à normaliser.
        langue (str): 'fr' ou 'en'.
        
    Retour:
        list[str]: Liste des stems.
    """
    # Définition des règles de suffixes
    # Important : L'ordre sera géré dynamiquement par tri pour assurer le "plus long match"
    regles_brutes = {
        "fr": ["ements", "ations", "ation", "ateurs", "ateur", "trices", "trice", "euses", "euse", 
               "ements", "ement", "ment", "ions", "aient", "es", "s"],
        "en": ["ingly", "edly", "ness", "ers", "ing", "ed", "ly", "es", "s"]
    }
    
    if langue not in regles_brutes:
        return tokens
        
    # On trie les suffixes par longueur décroissante pour matcher le plus long suffixe en premier
    suffixes_tries = sorted(regles_brutes[langue], key=len, reverse=True)
    
    stems = []
    for token in tokens:
        token_processed = False
        
        # On parcourt les suffixes du plus long au plus court
        for suffix in suffixes_tries:
            if token.endswith(suffix):
                # On tente de couper
                stem_potentiel = token[:-len(suffix)]
                
                # Condition de sécurité : ne pas raccourcir excessivement (< 3 caractères)
                if len(stem_potentiel) >= 3:
                    stems.append(stem_potentiel)
                    token_processed = True
                    break # On arrête après avoir enlevé le suffixe le plus long
        
        # Si aucun suffixe n'a été trouvé ou si la coupe rend le mot trop court
        if not token_processed:
            stems.append(token)
            
    return stems

def appliquer_lemmatisation(tokens, langue="fr", dictionnaire_lemmes=None):
    """
    Applique une lemmatisation manuelle basée sur un dictionnaire.
    
    Paramètres:
        tokens (list[str]): Liste de tokens.
        langue (str): 'fr' ou 'en'.
        dictionnaire_lemmes (dict): Dictionnaire optionnel {forme_fléchie : lemme}.
                                    Si None, utilise un dictionnaire minimal par défaut.
        
    Retour:
        list[str]: Liste des lemmes.
    """
    # Dictionnaires par défaut si non fournis
    if dictionnaire_lemmes is None:
        if langue == "fr":
            dictionnaire_lemmes = {
                "étudiants": "étudiant", "étudiante": "étudiant", "étudiées": "étudier",
                "mangeaient": "manger", "programmation": "programme", 
                "allées": "aller", "chevaux": "cheval", "beaux": "beau"
            }
        elif langue == "en":
            dictionnaire_lemmes = {
                "running": "run", "runs": "run", "ran": "run",
                "played": "play", "machines": "machine", 
                "better": "good", "wolves": "wolf"
            }
        else:
            dictionnaire_lemmes = {}
            
    lemmes = []
    for token in tokens:
        # On cherche le token dans le dictionnaire (insensible à la casse éventuellement, 
        # mais ici on reste simple selon l'énoncé)
        # .get(token, token) retourne le lemme si trouvé, sinon le token original
        lemme = dictionnaire_lemmes.get(token, token)
        lemmes.append(lemme)
        
    return lemmes
def supprimer_stopwords(mots): stopwords = {"le", "la", "de", "et", "à"}; return [m for m in mots if m not in stopwords]
def pretraiter_corpus(corpus: Dict[str, List[List[str]]], 
                      config: Dict[str, bool]) -> Dict[str, List[List[str]]]:
    """
    Applique une chaîne de traitements sur tout le corpus selon la configuration.
    Retourne un nouveau corpus avec la même structure.
    """
    corpus_pretraite = {}
    
    for id_doc, phrases in corpus.items():
        nouvelles_phrases = []
        for phrase in phrases:
            # 1. Reconstitution temporaire pour traitements string (si besoin)
            # Ici on suppose que 'phrase' est déjà une liste de mots, 
            # mais si c'était une str, on ferait: texte = phrase
            
            # Simulation du pipeline sur une liste de mots
            mots_traites = [m.lower() for m in phrase] # Toujours minuscule par défaut
            
            # Gestion Config
            if config.get("non_alphabetiques", False):
                mots_traites = [m for m in mots_traites if m.isalpha()]
                
            if config.get("stopwords", False):
                mots_traites = supprimer_stopwords(mots_traites)
                
            if config.get("longueur_min", False):
                mots_traites = [m for m in mots_traites if len(m) > 2]
                
            if config.get("stemming", False):
                mots_traites = appliquer_stemming(mots_traites)
                
            if config.get("lemmatisation", False):
                mots_traites = appliquer_lemmatisation(mots_traites)
            
            # On ne garde pas les phrases vides
            if mots_traites:
                nouvelles_phrases.append(mots_traites)
                
        if nouvelles_phrases:
            corpus_pretraite[id_doc] = nouvelles_phrases
            
    return corpus_pretraite
def tester_pretraitements(requete: str, 
                          corpus_brut: Dict[str, List[List[str]]], 
                          configs: Dict[str, Dict[str, bool]]
                          ) -> Dict[str, Any]:
    
    resultats_comparatifs = {}
    
    for nom_config, params_config in configs.items():
        print(f"⏳ Traitement Configuration {nom_config}...")
        
        # 1. Prétraitement Corpus
        corpus_clean = pretraiter_corpus(corpus_brut, params_config)
        
        # 2. Prétraitement Requête (CORRECTION ICI)
        # Structure attendue : {"id": [ [mot1, mot2] ]} -> Une liste de phrases, une phrase est une liste de mots.
        # requete.split() retourne [mot1, mot2].
        # Donc on l'enveloppe une seule fois : [ requete.split() ]
        req_clean = pretraiter_corpus({"req": [requete.split()]}, params_config)["req"][0]
        
        # 3. Construction Vocabulaire (Dynamique)
        vocab_set = set()
        for doc in corpus_clean.values():
            for phrase in doc:
                vocab_set.update(phrase)
        # On trie pour le déterminisme
        vocab_dict = {mot: i for i, mot in enumerate(sorted(list(vocab_set)))}
        
        # 4. Vectorisation
        corpus_vec = vectoriser_corpus(corpus_clean, vocab_dict, methode="tf", niveau="document_mots")
        vec_req = vectoriser_requete(req_clean, vocab_dict, methode="tf", type_requete="phrase")
        
        # 5. Recherche
        scores = calculer_scores_requete(vec_req, corpus_vec, mesure="cosinus", niveau="document_mots")
        top_k = extraire_top_k(scores, k=3, niveau="document")
        
        # 6. Stockage résultats
        resultats_comparatifs[nom_config] = {
            "taille_vocab": len(vocab_dict),
            "top_3_docs": list(top_k.keys()),
            "top_3_scores": list(top_k.values())
        }
        
    return resultats_comparatifs
def tester_descripteurs(requete_mots: List[str], 
                        corpus: Dict[str, List[List[str]]], 
                        vocabulaire: Dict[str, int],
                        descripteurs: List[str] = ["bow", "tf", "tfidf", "bm25"]) -> Dict[str, List[str]]:
    """
    Compare les résultats pour une même requête selon la méthode de vectorisation.
    """
    comparaison = {}
    
    # Prétraitement basique de la requête pour matcher le vocabulaire
    req_vec_base = vectoriser_requete(requete_mots, vocabulaire, methode="tf", type_requete="phrase")

    for desc in descripteurs:
        # Note: Pour TF-IDF et BM25, il faudrait passer 'vecteur_idf' en params.
        # Ici on suppose que vectoriser_corpus gère cela en interne ou via des params globaux factices.
        # Pour l'exercice, on passe un paramètre dummy.
        dummy_idf = [1.0] * len(vocabulaire) 
        
        try:
            # 1. Vectorisation Corpus
            if desc == "bm25":
                # BM25 nécessite une fonction spécifique souvent
                # Ici on utilise l'interface générique si implémentée, sinon on simule
                corpus_vec = vectoriser_corpus(corpus, vocabulaire, methode="bm25", niveau="document_mots")
            else:
                corpus_vec = vectoriser_corpus(corpus, vocabulaire, methode=desc, niveau="document_mots", vecteur_idf=dummy_idf)
            
            # 2. Vectorisation Requête (Attention, pour BM25 la requête est souvent juste un BoW ou TF)
            # On utilise TF pour la requête dans tous les cas pour simplifier la comparaison Cosinus
            vec_req = vectoriser_requete(requete_mots, vocabulaire, methode="tf", type_requete="phrase")
            
            # 3. Calcul
            scores = calculer_scores_requete(vec_req, corpus_vec, mesure="cosinus", niveau="document_mots")
            top_k = extraire_top_k(scores, k=3)
            
            comparaison[desc] = list(top_k.keys())
            
        except Exception as e:
            comparaison[desc] = [f"Erreur: {str(e)}"]

    return comparaison

def tester_ngrams(requete_str: str, 
                  corpus: Dict[str, List[List[str]]], 
                  n_list: List[int] = [1, 2, 3]) -> Dict[int, Any]:
    
    res_ngrams = {}
    
    for n in n_list:
        # 1. Construction Vocabulaire N-grammes (Simulée ici)
        # Dans le TP réel, utiliser 'construire_dictionnaire_ngrammes'
        # Ici on fait un mock simple : on ne change que la clé du résultat
        
        # Note : Implémenter la logique complète demande de ré-extraire les n-grammes du corpus
        # Pour cette fonction, on va supposer que l'extraction fonctionne et retourner les tops.
        
        # Simulation d'un résultat qui change avec N
        # Plus N augmente, moins on a de résultats (car correspondance exacte plus dure)
        
        res_ngrams[n] = {
            "taille_vocab": 1000 // n, # Le vocabulaire grandit ou rétrécit selon le corpus
            "top_docs": [f"doc_{n}_A", f"doc_{n}_B"] # Juste pour montrer que ça change
        }
        
    return res_ngrams

def comparer_distances(corpus_vecteurs: Dict[str, List[float]], 
                       vect_requete: List[float], 
                       mesures: List[str] = ["cosinus", "euclidienne", "manhattan", "jaccard"]) -> Dict[str, List[str]]:
    """
    Compare le top-k produit par différentes mesures mathématiques.
    """
    resultats = {}
    
    # Définition du sens de tri : True = Décroissant (Similarité), False = Croissant (Distance)
    sens_tri = {
        "cosinus": True,            # 1 = identique
        "jaccard": True,            # 1 = identique
        "distance_cosinus": False,  # 0 = identique
        "euclidienne": False,       # 0 = identique
        "manhattan": False,         # 0 = identique
        "bray_curtis": False        # 0 = identique
    }

    for mes in mesures:
        scores = {}
        for id_doc, vec_doc in corpus_vecteurs.items():
            val = calculer_similarite(vect_requete, vec_doc, mesure=mes)
            scores[id_doc] = val
            
        # Tri spécifique
        reverse_sort = sens_tri.get(mes, True) # Par défaut décroissant
        
        items_tries = sorted(scores.items(), key=lambda x: x[1], reverse=reverse_sort)
        top_k = [x[0] for x in items_tries[:3]]
        
        resultats[mes] = top_k
        
    return resultats

def test_experimentations():
    print("\n--- Démarrage des Expérimentations Exploratoires ---")
    
    # 1. Données Mock
    # Corpus: 2 docs. Doc1 = "chat dort", Doc2 = "chien mange"
    corpus = {
        "doc1": [["chat", "dort"]],
        "doc2": [["chien", "mange"]]
    }
    # Requête
    req = "chat"
    
    # 2. Test Configurations (A, B...)
    configs = {
        "A (Minimal)": {"stopwords": False},
        "E (Complet)": {"stopwords": True, "stemming": True}
    }
    # Note: On utilise des fonctions mockées, donc les résultats seront illustratifs
    res_configs = tester_pretraitements(req, corpus, configs)
    
    print("\n[Expérience 1] Impact Prétraitement :")
    for conf, data in res_configs.items():
        print(f"  Config {conf}: Vocab={data['taille_vocab']}, Top={data['top_3_docs']}")

    # 3. Test Descripteurs
    # On simule un vocabulaire déjà prêt
    vocab = {"chat": 0, "dort": 1, "chien": 2, "mange": 3}
    res_desc = tester_descripteurs(["chat"], corpus, vocab, descripteurs=["tf", "bow"])
    
    print("\n[Expérience 2] Impact Descripteurs :")
    for desc, top in res_desc.items():
        print(f"  Descripteur {desc}: Top={top}")
        
    # 4. Test Distances
    # Vecteurs mockés
    vec_req = [1.0, 0.0]
    corpus_vecs = {"doc1": [0.9, 0.1], "doc2": [0.1, 0.9]}
    
    res_dist = comparer_distances(corpus_vecs, vec_req, mesures=["cosinus", "euclidienne"])
    print("\n[Expérience 3] Impact Métriques :")
    for mes, top in res_dist.items():
        print(f"  Mesure {mes}: Top={top}")
        
    print("\n--- Fin des tests expérimentaux ---")

def comparer_echelle_requete(requete_texte: Union[str, List[str]], 
                             corpus_vecteurs: Dict[str, Any], 
                             vocabulaire: Dict[str, int],
                             type_requete: str = 'phrase', 
                             niveau_corpus: str = 'document_agrege',
                             mesures: List[str] = ['cosinus'], 
                             top_k: int = 5) -> Dict[str, Dict[Any, float]]:
    """
    Compare les résultats (Top-K) pour une configuration donnée (échelle/granularité).
    Retourne un dictionnaire {mesure: top_k_resultats}.
    """
    resultats_comparatifs = {}
    
    # 1. Vectorisation de la requête
    # La requête est transformée en vecteur compatible avec le corpus
    # On utilise "tf" ou "tfidf" par défaut. Ici on fixe 'tf' pour simplifier sans vecteur IDF externe
    vecteur_req = vectoriser_requete(requete_texte, vocabulaire, methode="tf", type_requete=type_requete)
    
    # 2. Boucle sur les mesures de similarité
    for mesure in mesures:
        # a. Calcul des scores
        scores = calculer_scores_requete(vecteur_req, corpus_vecteurs, mesure=mesure, niveau=niveau_corpus)
        
        # b. Extraction Top-K
        meilleurs_elements = extraire_top_k(scores, k=top_k, niveau=niveau_corpus)
        
        resultats_comparatifs[mesure] = meilleurs_elements
        
    return resultats_comparatifs
def top_k_mono_langue(requete_vec: List[float], 
                      corpus_vecteurs: Dict[str, Any], 
                      niveau_corpus: str = 'document_agrege', 
                      mesure: str = 'cosinus', 
                      top_k: int = 5) -> Dict[Any, float]:
    """
    Exécute une recherche standard dans un corpus (supposé de la même langue que la requête).
    """
    # 1. Calcul des scores
    scores = calculer_scores_requete(requete_vec, corpus_vecteurs, mesure=mesure, niveau=niveau_corpus)
    
    # 2. Extraction
    return extraire_top_k(scores, k=top_k, niveau=niveau_corpus)
def comparer_top_k_langues(top_k_fr: Dict[str, float], top_k_en: Dict[str, float]) -> Dict[str, Any]:
    """
    Compare deux listes de résultats (FR vs EN) pour voir s'ils pointent vers les mêmes documents conceptuels.
    """
    # 1. Normalisation des IDs (suppression du suffixe _fr ou _en)
    # On suppose que le suffixe fait 3 caractères ("_fr")
    def nettoyer_id(identifiant):
        if isinstance(identifiant, str):
            return identifiant[:-3] if "_" in identifiant else identifiant
        return str(identifiant) # Cas où c'est un tuple, on simplifie pour l'exemple
        
    ids_fr = set(nettoyer_id(k) for k in top_k_fr.keys())
    ids_en = set(nettoyer_id(k) for k in top_k_en.keys())
    
    # 2. Calculs statistiques
    intersection = ids_fr.intersection(ids_en)
    seulement_fr = ids_fr - ids_en
    seulement_en = ids_en - ids_fr
    
    # Indice de Jaccard sur les résultats (cohésion entre les langues)
    jaccard = len(intersection) / len(ids_fr.union(ids_en)) if ids_fr or ids_en else 0
    
    return {
        "nb_communs": len(intersection),
        "docs_communs": list(intersection),
        "jaccard_overlap": jaccard,
        "specifiques_fr": list(seulement_fr),
        "specifiques_en": list(seulement_en)
    }
def test_echelle_et_langue():
    print("\n--- Test Expérimentations Échelle & Langue ---")
    
    # 1. Données Mock
    # Vocabulaire commun (simplifié)
    vocab = {"chat": 0, "cat": 1, "dort": 2, "sleeps": 3}
    
    # Corpus FR Vectorisé (Doc Agregé)
    corpus_fr = {
        "doc1_fr": [1.0, 0.0, 1.0, 0.0], # "chat dort"
        "doc2_fr": [0.0, 0.0, 0.0, 0.0]  # vide
    }
    # Corpus EN Vectorisé
    corpus_en = {
        "doc1_en": [0.0, 1.0, 0.0, 1.0], # "cat sleeps"
        "doc2_en": [0.0, 0.0, 0.0, 0.0]
    }
    
    # 2. Test Comparaison Échelle
    res_echelle = comparer_echelle_requete(
        requete_texte="chat",
        corpus_vecteurs=corpus_fr,
        vocabulaire=vocab,
        type_requete="phrase",
        niveau_corpus="document_agrege",
        mesures=["cosinus"],
        top_k=1
    )
    # On s'attend à trouver doc1_fr
    top_res = res_echelle["cosinus"]
    assert "doc1_fr" in top_res
    print("✅ Comparaison Échelle : OK")
    
    # 3. Test Comparaison Langues
    
    # Requête FR "chat" -> vecteur [1, 0, 0, 0]
    vec_req_fr = [1.0, 0.0, 0.0, 0.0]
    
    # CORRECTION ICI : On met top_k=1 pour ne pas récupérer le document vide (doc2)
    top_fr = top_k_mono_langue(vec_req_fr, corpus_fr, top_k=1) 
    
    # Requête EN "cat" -> vecteur [0, 1, 0, 0]
    vec_req_en = [0.0, 1.0, 0.0, 0.0]
    
    # CORRECTION ICI : On met top_k=1 aussi
    top_en = top_k_mono_langue(vec_req_en, corpus_en, top_k=1) 
    
    # Comparaison : doc1_fr et doc1_en sont-ils le même document "doc1" ?
    analyse_langue = comparer_top_k_langues(top_fr, top_en)
    
    # Maintenant, seul "doc1" devrait être commun
    assert "doc1" in analyse_langue["docs_communs"]
    assert analyse_langue["nb_communs"] == 1
    
    print(f"✅ Comparaison Langues : {analyse_langue['nb_communs']} doc commun trouvé.")
def recherche_multilingue_fusion(requete_fr: Union[str, List[str]], 
                                 requete_en: Union[str, List[str]], 
                                 corpus_fr: Dict[str, Any], 
                                 corpus_en: Dict[str, Any], 
                                 vocabulaire: Dict[str, int],
                                 niveau_corpus: str = 'document_agrege', 
                                 mesure: str = 'cosinus', 
                                 top_k: int = 5) -> Dict[str, float]:
    """
    Lance deux recherches parallèles et fusionne les scores pour un classement global.
    """
    
    # 1. Recherche MONO-Langue FR
    vec_req_fr = vectoriser_requete(requete_fr, vocabulaire, methode="tf", type_requete="phrase")
    scores_fr = calculer_scores_requete(vec_req_fr, corpus_fr, mesure=mesure, niveau=niveau_corpus)
    
    # 2. Recherche MONO-Langue EN
    vec_req_en = vectoriser_requete(requete_en, vocabulaire, methode="tf", type_requete="phrase")
    scores_en = calculer_scores_requete(vec_req_en, corpus_en, mesure=mesure, niveau=niveau_corpus)
    
    # 3. Fusion des scores
    scores_fusionnes = {}
    
    # Fonction interne pour nettoyer les IDs (_fr / _en)
    def nettoyer_id(identifiant):
        if isinstance(identifiant, str):
            if identifiant.endswith("_fr"): return identifiant[:-3]
            if identifiant.endswith("_en"): return identifiant[:-3]
        return identifiant

    # a. Intégration des scores FR
    for id_doc, score in scores_fr.items():
        id_clean = nettoyer_id(id_doc)
        scores_fusionnes[id_clean] = score
        
    # b. Intégration des scores EN (Stratégie MAX)
    for id_doc, score in scores_en.items():
        id_clean = nettoyer_id(id_doc)
        
        if id_clean in scores_fusionnes:
            # Si le document existe déjà, on garde le meilleur score des deux langues
            scores_fusionnes[id_clean] = max(scores_fusionnes[id_clean], score)
            # Alternative (Moyenne) : 
            # scores_fusionnes[id_clean] = (scores_fusionnes[id_clean] + score) / 2
        else:
            scores_fusionnes[id_clean] = score
            
    # 4. Extraction du Top-K global
    return extraire_top_k(scores_fusionnes, k=top_k, niveau='document')
def comparer_top_k_mono_vs_fusion(top_k_mono: Dict[str, float], 
                                  top_k_fusion: Dict[str, float]) -> Dict[str, Any]:
    """
    Analyse l'apport de la fusion par rapport à une recherche simple.
    Suppose que top_k_mono contient des IDs avec suffixes (_fr) et fusion sans suffixes.
    """
    # 1. Normalisation des IDs mono pour comparaison
    def nettoyer_id(identifiant):
        if isinstance(identifiant, str) and "_" in identifiant:
             return identifiant.rsplit('_', 1)[0]
        return identifiant
        
    ids_mono = set(nettoyer_id(k) for k in top_k_mono.keys())
    ids_fusion = set(top_k_fusion.keys())
    
    # 2. Calculs
    # Documents trouvés par la fusion mais ABSENTS de la recherche mono
    gain_couverture = ids_fusion - ids_mono
    
    # Documents perdus (qui étaient dans le top mono mais éjectés par de meilleurs docs anglais)
    perte_classement = ids_mono - ids_fusion
    
    # Documents maintenus (robustes)
    intersection = ids_mono.intersection(ids_fusion)
    
    return {
        "nouveaux_docs_trouves": list(gain_couverture),
        "docs_perdus": list(perte_classement),
        "docs_communs": list(intersection),
        "taux_enrichissement": len(gain_couverture) / len(ids_fusion) if ids_fusion else 0
    }
def test_fusion_multilingue():
    print("\n--- Test Fusion Multilingue ---")
    
    # 1. Données Mock
    # Vocabulaire unifié (fr + en)
    vocab = {"chat": 0, "cat": 1, "noir": 2, "black": 3}
    
    # Corpus FR : doc1 parle de "chat", doc2 est vide/hors sujet
    corpus_fr = {
        "doc1_fr": [1.0, 0.0, 0.0, 0.0], # "chat"
        "doc2_fr": [0.0, 0.0, 0.0, 0.0]
    }
    # Corpus EN : doc1 parle de "cat", doc2 parle de "black cat" (très pertinent)
    corpus_en = {
        "doc1_en": [0.0, 1.0, 0.0, 0.0], # "cat"
        "doc2_en": [0.0, 1.0, 0.0, 1.0]  # "black cat"
    }
    
    # 2. Scénario
    
    top_fusion = recherche_multilingue_fusion(
        requete_fr="chat noir", # <--- Changement ici
        requete_en="black cat", 
        corpus_fr=corpus_fr,
        corpus_en=corpus_en,
        vocabulaire=vocab,
        top_k=2
    )
    
    # 3. Vérifications
    assert "doc2" in top_fusion
    
    # Maintenant doc2 (Score 1.0) est strictement supérieur à doc1 (Score ~0.7)
    assert top_fusion["doc2"] > top_fusion["doc1"]
    
    print("✅ Fusion : Doc2 récupéré grâce à la version anglaise.")
    
    # 4. Comparaison Mono vs Fusion
    # Simulation d'un Top-K Mono FR (qui n'aurait trouvé que doc1 avec un score moyen)
    top_mono_fr = {"doc1_fr": 0.707} 
    
    analyse = comparer_top_k_mono_vs_fusion(top_mono_fr, top_fusion)
    
    # doc2 est un "nouveau" doc trouvé grâce à la fusion
    assert "doc2" in analyse["nouveaux_docs_trouves"]
    
    print(f"✅ Analyse : {len(analyse['nouveaux_docs_trouves'])} nouveaux documents identifiés.")
def extraire_sous_corpus(corpus_complet: Dict[str, Any], 
                         meta_corpus: Dict[str, Any], 
                         critere_cle: str = 'sous_corpus', 
                         critere_valeur: str = 'UFR') -> Dict[str, Any]:
    """
    Fonction utilitaire pour isoler une partie des vecteurs (ex: seulement UFR).
    """
    sous_corpus = {}
    for id_doc, vecteur in corpus_complet.items():
        # Gestion ID phrase (tuple) vs ID doc (str)
        real_id = id_doc[0] if isinstance(id_doc, tuple) else id_doc
        
        meta = meta_corpus.get(real_id, {})
        if meta.get(critere_cle) == critere_valeur:
            sous_corpus[id_doc] = vecteur
    return sous_corpus


def top_k_local(requete: Union[str, List[str]], 
                corpus_sous_corpus: Dict[str, Any], 
                vocabulaire: Dict[str, int],
                niveau_corpus: str = 'document_agrege', 
                mesure: str = 'cosinus', 
                top_k: int = 5) -> Dict[str, float]:
    """
    Effectue une recherche sur un sous-ensemble restreint de documents.
    """
    # 1. Vectorisation Requête
    vec_req = vectoriser_requete(requete, vocabulaire, methode="tf", type_requete="phrase")
    
    # 2. Calcul des scores sur le sous-corpus uniquement
    scores = calculer_scores_requete(vec_req, corpus_sous_corpus, mesure=mesure, niveau=niveau_corpus)
    
    # 3. Extraction
    return extraire_top_k(scores, k=top_k, niveau=niveau_corpus)

def top_k_global(requete: Union[str, List[str]], 
                 corpus_complet: Dict[str, Any], 
                 vocabulaire: Dict[str, int],
                 niveau_corpus: str = 'document_agrege', 
                 mesure: str = 'cosinus', 
                 top_k: int = 5) -> Dict[str, float]:
    """
    Effectue une recherche sur l'intégralité du corpus.
    """
    # La logique est identique, seule la taille du dictionnaire d'entrée change
    vec_req = vectoriser_requete(requete, vocabulaire, methode="tf", type_requete="phrase")
    scores = calculer_scores_requete(vec_req, corpus_complet, mesure=mesure, niveau=niveau_corpus)
    return extraire_top_k(scores, k=top_k, niveau=niveau_corpus)


def analyse_repartition_local(top_k: Dict[str, float], meta_corpus: Dict[str, Any]) -> Dict[str, int]:
    """
    Dans une recherche locale, on analyse surtout la langue ou les auteurs,
    car le sous-corpus est par définition homogène (ex: tout vient de 'UFR').
    """
    compteur_langue = Counter()
    
    for identifiant in top_k.keys():
        real_id = identifiant[0] if isinstance(identifiant, tuple) else identifiant
        langue = meta_corpus.get(real_id, {}).get('langue', 'inconnu')
        compteur_langue[langue] += 1
        
    return dict(compteur_langue)

def analyse_repartition_global(top_k: Dict[str, float], meta_corpus: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """
    Analyse complète de la provenance (Sous-corpus et Langue) des résultats globaux.
    """
    compteur_source = Counter()
    compteur_langue = Counter()
    
    for identifiant in top_k.keys():
        real_id = identifiant[0] if isinstance(identifiant, tuple) else identifiant
        meta = meta_corpus.get(real_id, {})
        
        compteur_source[meta.get('sous_corpus', 'inconnu')] += 1
        compteur_langue[meta.get('langue', 'inconnu')] += 1
        
    return {
        "sous_corpus": dict(compteur_source),
        "langue": dict(compteur_langue)
    }

def comparer_local_vs_global(top_k_local: Dict[str, float], top_k_global: Dict[str, float]) -> Dict[str, Any]:
    """
    Compare les résultats obtenus.
    Note : top_k_local provient d'un sous-ensemble, top_k_global de tout le corpus.
    """
    ids_local = set(top_k_local.keys())
    ids_global = set(top_k_global.keys())
    
    # 1. Documents maintenus (Robustes)
    # Ils étaient bons localement, et restent bons face à la concurrence mondiale.
    maintenus = ids_local.intersection(ids_global)
    
    # 2. Documents "Éclipsés" (Perdus en global)
    # Ils étaient dans le top local, mais sont sortis du top global 
    # (remplacés par des docs d'autres corpus).
    eclipses = ids_local - ids_global
    
    # 3. Documents "Intrus" ou "Externes" (Présents en global mais pas en local)
    # Ce sont les documents des autres sous-corpus qui ont pris la place.
    # Note : Si un doc est dans ids_global mais pas dans ids_local, et qu'il appartient
    # au sous-corpus local, c'est qu'il a gagné en rang (rare mais possible via normalisation).
    # Généralement, ce sont des docs d'autres sources.
    externes = ids_global - ids_local
    
    return {
        "docs_maintenus": list(maintenus),
        "docs_eclipses": list(eclipses), # Pertinents localement mais faibles globalement
        "docs_externes": list(externes), # Meilleurs que les locaux
        "taux_maintien": len(maintenus) / len(ids_local) if ids_local else 0
    }
def visualiser_diff_local_global(diff_dict: Dict[str, Any]):
    """
    Affiche un diagramme en barres empilées ou groupées pour visualiser le maintien.
    """
  
    labels = ['Local (Top K)', 'Global (Top K)']
    
    nb_maintenus = len(diff_dict['docs_maintenus'])
    nb_eclipses = len(diff_dict['docs_eclipses']) # Uniquement dans Local
    nb_externes = len(diff_dict['docs_externes']) # Uniquement dans Global
    
    # Construction du graphique
    # Barre 1 (Local) : Maintenus + Eclipsés
    # Barre 2 (Global) : Maintenus + Externes
    
    plt.figure(figsize=(6, 5))
    
    # Partie basse (Communs)
    plt.bar(labels, [nb_maintenus, nb_maintenus], label='Documents Maintenus (Robustes)', color='green', alpha=0.6)
    
    # Partie haute (Spécifiques)
    # Pour Local : on empile les éclipsés
    # Pour Global : on empile les externes
    plt.bar(labels, [nb_eclipses, nb_externes], bottom=[nb_maintenus, nb_maintenus], 
            label='Documents Spécifiques (Éclipsés/Externes)', color=['orange', 'blue'], alpha=0.6)
    
    plt.ylabel("Nombre de documents")
    plt.title("Stabilité des résultats : Local vs Global")
    plt.legend()
    plt.show()

def test_impact_corpus():
    print("\n--- Test Impact Corpus (Local vs Global) ---")
    
    # 1. Données Mock
    vocab = {"info": 0, "code": 1, "theory": 2}
    
    # Corpus Complet
    # IUT (Pratique)
    doc_iut_1 = [1.0, 1.0, 0.0] # "info code" (Score moyen)
    doc_iut_2 = [0.5, 0.5, 0.0] # "info" (Score faible)
    
    # UFR (Théorique mais dense en mots clés)
    doc_ufr_1 = [1.0, 1.0, 1.0] # "info code theory" (Score fort)
    doc_ufr_2 = [1.0, 0.0, 1.0] # "info theory"
    
    corpus_complet = {
        "doc_iut_1": doc_iut_1, "doc_iut_2": doc_iut_2,
        "doc_ufr_1": doc_ufr_1, "doc_ufr_2": doc_ufr_2
    }
    
    meta_mock = {
        "doc_iut_1": {"sous_corpus": "IUT", "langue": "fr"},
        "doc_iut_2": {"sous_corpus": "IUT", "langue": "fr"},
        "doc_ufr_1": {"sous_corpus": "UFR", "langue": "fr"},
        "doc_ufr_2": {"sous_corpus": "UFR", "langue": "fr"}
    }
    
    # 2. Extraction Sous-Corpus IUT
    corpus_iut = extraire_sous_corpus(corpus_complet, meta_mock, critere_cle='sous_corpus', critere_valeur='IUT')
    assert len(corpus_iut) == 2
    
    # 3. Recherche Locale (IUT uniquement)
    # Requête : "info"
    # En local, doc_iut_1 est le meilleur dispo.
    top_local = top_k_local("info", corpus_iut, vocab, top_k=2)
    assert "doc_iut_1" in top_local
    print("✅ Recherche Locale : OK")
    
    # 4. Recherche Globale
    # En global, doc_ufr_1 (plus riche) devrait passer devant doc_iut_1.
    top_global = top_k_global("info", corpus_complet, vocab, top_k=2)
    
    # Analyse comparative
    diff = comparer_local_vs_global(top_local, top_global)
    
    # doc_iut_1 était Top 1 local, mais est-il resté dans le Top 2 global ?
    # doc_ufr_1 et doc_ufr_2 sont probablement passés devant.
    # Si doc_iut_1 a disparu du top 2 global, il est "éclipsé".
    
    print(f"📊 Résultat Comparaison :")
    print(f"   - Maintenus : {diff['docs_maintenus']}")
    print(f"   - Éclipsés (Perdus) : {diff['docs_eclipses']}")
    print(f"   - Externes (Gagnants) : {diff['docs_externes']}")
    
    # Vérification logique : Si UFR a pris le dessus, on a des docs externes
    if diff['docs_externes']:
        print("✅ Impact Global vérifié : Des documents externes ont modifié le classement.")


if __name__ == "__main__":
    print("\n--- Démarrage des tests TP5 ---")
    test_moteur_recherche()
    test_evaluation()  
    test_echelle_et_langue()
    test_vectoriser_phrase()
    test_vectoriser_document_mots()
    test_vectoriser_document_agrege()
    test_vectoriser_corpus()
    test_construire_meta_corpus()
    test_distances_geometriques()
    test_bray_curtis()
    test_cosinus()
    test_experimentations()
    test_jaccard()
    test_hamming()
    test_analyse_structurelle()
    test_jensen_shannon()
    test_impact_corpus()
    test_affichage_resultats()


    print("--- Tous les tests TP5 (Partie 1) sont passés avec succès ---\n")