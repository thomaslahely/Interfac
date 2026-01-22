
import math
from typing import List, Tuple, Dict
### Énoncé de la fonction 1 : Encodage (Texte -> Indices)

# **Titre de la fonction :**
# > texte_vers_indices(texte, vocabulaire_direct)

# **Consigne :**
# * Transformer une phrase en une liste d'entiers basés sur un vocabulaire donné.
# * Les mots doivent être mis en minuscule avant la recherche.
# * Les mots absents du vocabulaire doivent être remplacés par -1.

def texte_vers_indices(texte: str, vocabulaire_direct: Dict[str, int]) -> List[int]:
    """
    Transforme une phrase en une liste d'indices selon le vocabulaire.
    """
    if not texte or not isinstance(texte, str):
        return []
    
    # Tokenisation simple (séparation par espace et minuscule)
    mots = texte.lower().split()
    indices = []
    
    for mot in mots:
        # On cherche le mot, s'il n'existe pas, on retourne -1
        index = vocabulaire_direct.get(mot, -1)
        indices.append(index)
        
    return indices

### 🔍 Explication des tests unitaires — Encodage
# 1. **Phrase connue :** "chat dort" doit donner [0, 2].
# 2. **Phrase mixte :** "chien mange maison" doit donner [1, 3, 4].
# 3. **Mots inconnus :** "souris mange fromage" doit gérer les absents -> [-1, 3, -1].

def test_texte_vers_indices():
    # Configuration du vocabulaire de l'exercice
    vocab = {'chat': 0, 'chien': 1, 'dort': 2, 'mange': 3, 'maison': 4}
    
    # Test 1 : Cas standard
    assert texte_vers_indices("chat dort", vocab) == [0, 2]
    
    # Test 2 : Cas standard plus long
    assert texte_vers_indices("chien mange maison", vocab) == [1, 3, 4]
    
    # Test 3 : Cas avec mots inconnus (OOV - Out of Vocabulary)
    # "souris" n'est pas dans le dict -> -1
    assert texte_vers_indices("souris mange fromage", vocab) == [-1, 3, -1]
    
    # Test 4 : Cas vide
    assert texte_vers_indices("", vocab) == []

    print("Test texte_vers_indices : OK")


### Énoncé de la fonction 2 : Décodage (Indices -> Texte)

# **Titre de la fonction :**
# > indices_vers_texte(indices, vocabulaire_inverse)

# **Consigne :**
# * Reconstruire une phrase lisible à partir d'une liste d'entiers.
# * Les indices invalides (non présents dans le dictionnaire) doivent être remplacés par "<UNK>".

def indices_vers_texte(indices: List[int], vocabulaire_inverse: Dict[int, str]) -> str:
    """
    Reconstruit un texte à partir d'une liste d'indices.
    """
    if not indices or not isinstance(indices, list):
        return ""

    mots_reconstruits = []
    
    for index in indices:
        # On cherche l'index, s'il n'existe pas, on met un token spécial
        mot = vocabulaire_inverse.get(index, "<UNK>")
        mots_reconstruits.append(mot)
            
    # On joint les mots avec des espaces pour refaire une phrase
    return " ".join(mots_reconstruits)

### 🔍 Explication des tests unitaires — Décodage
# 1. **Reconstruction valide :** [1, 2, 0, 3] -> "chien dort chat mange".
# 2. **Index inconnu :** [10, 2, 3] -> "<UNK> dort mange".

def test_indices_vers_texte():
    # Configuration du vocabulaire inverse de l'exercice
    vocab_inv = {0: 'chat', 1: 'chien', 2: 'dort', 3: 'mange', 4: 'maison'}
    
    # Test 1 : Reconstruction parfaite
    vec_1 = [1, 2, 0, 3]
    assert indices_vers_texte(vec_1, vocab_inv) == "chien dort chat mange"
    
    # Test 2 : Gestion d'un index hors limites (10)
    vec_2 = [10, 2, 3]
    assert indices_vers_texte(vec_2, vocab_inv) == "<UNK> dort mange"
    
    # Test 3 : Liste vide
    assert indices_vers_texte([], vocab_inv) == ""

    print("Test indices_vers_texte : OK")

### Énoncé de la fonction 3 : Encodage One-Hot

# **Titre de la fonction :**
# > one_hot_encode_mots(texte, vocabulaire_direct)

# **Consigne :**
# * Convertir un texte en une liste de vecteurs.
# * Chaque vecteur a la taille du vocabulaire.
# * La position de l'index du mot est à 1, le reste à 0.
# * Si un mot est inconnu, on retourne un vecteur rempli de zéros (vecteur nul).

def one_hot_encode_mots(texte: str, vocabulaire_direct: Dict[str, int]) -> List[List[int]]:
    """
    Transforme un texte en une liste de vecteurs One-Hot par mot.
    """
    if not texte or not isinstance(texte, str):
        return []

    taille_vocab = len(vocabulaire_direct)
    mots = texte.lower().split()
    liste_vecteurs = []

    for mot in mots:
        # Initialiser un vecteur de zéros de la taille du vocabulaire
        vecteur = [0] * taille_vocab
        
        # Récupérer l'index
        index = vocabulaire_direct.get(mot)
        
        # Si le mot existe, on met le 1 à la bonne position
        if index is not None:
            vecteur[index] = 1
        # Sinon (mot inconnu), le vecteur reste rempli de zéros
        
        liste_vecteurs.append(vecteur)

    return liste_vecteurs

### 🔍 Explication des tests unitaires — Encodage One-Hot
# 1. **Mot simple :** "chat" (index 0) doit donner [[1, 0, 0, 0, 0]].
# 2. **Séquence :** "chat maison" doit donner une liste de deux vecteurs.
# 3. **Inconnu :** "alien" doit donner un vecteur nul [[0, 0, 0, 0, 0]].

def test_one_hot_encode():
    # Vocabulaire: taille 5
    vocab = {'chat': 0, 'chien': 1, 'dort': 2, 'mange': 3, 'maison': 4}
    
    # Test 1 : Mot unique
    res_chat = one_hot_encode_mots("chat", vocab)
    assert res_chat == [[1, 0, 0, 0, 0]]
    
    # Test 2 : Séquence
    # chat (idx 0) -> 10000
    # maison (idx 4) -> 00001
    res_seq = one_hot_encode_mots("chat maison", vocab)
    assert res_seq == [[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]]
    
    # Test 3 : Mot hors vocabulaire
    res_unk = one_hot_encode_mots("alien", vocab)
    assert res_unk == [[0, 0, 0, 0, 0]]

    print("Test one_hot_encode_mots : OK")


### Énoncé de la fonction 4 : Décodage One-Hot

# **Titre de la fonction :**
# > one_hot_decode(vecteurs, vocabulaire_inverse)

# **Consigne :**
# * Convertir une liste de vecteurs One-Hot en chaîne de caractères.
# * Retrouver l'index où la valeur est 1 pour déduire le mot.
# * Si le vecteur ne contient que des zéros, ignorer ou marquer comme <UNK>.

def one_hot_decode(vecteurs: List[List[int]], vocabulaire_inverse: Dict[int, str]) -> str:
    """
    Transforme une liste de vecteurs One-Hot en texte reconstruit.
    """
    mots_reconstruits = []
    
    for vec in vecteurs:
        try:
            # On cherche la position du 1
            index = vec.index(1)
            mot = vocabulaire_inverse.get(index, "<UNK>")
            mots_reconstruits.append(mot)
        except ValueError:
            # vec.index(1) lève une erreur si 1 n'est pas dans la liste (cas du vecteur nul)
            mots_reconstruits.append("<UNK>")
            
    return " ".join(mots_reconstruits)

### 🔍 Explication des tests unitaires — Décodage One-Hot
# 1. **Reconstruction :** [[0,1,0,0,0]] doit rendre "chien".
# 2. **Vecteur nul :** [[0,0,0,0,0]] doit rendre "<UNK>".

def test_one_hot_decode():
    vocab_inv = {0: 'chat', 1: 'chien', 2: 'dort', 3: 'mange', 4: 'maison'}
    
    # Test 1 : Cas nominal
    # [0,1,0,0,0] -> index 1 -> chien
    # [0,0,0,1,0] -> index 3 -> mange
    vecs = [[0, 1, 0, 0, 0], [0, 0, 0, 1, 0]]
    assert one_hot_decode(vecs, vocab_inv) == "chien mange"
    
    # Test 2 : Vecteur nul (mot inconnu)
    vecs_unk = [[0, 0, 0, 0, 0]]
    assert one_hot_decode(vecs_unk, vocab_inv) == "<UNK>"

    print("Test one_hot_decode : OK")

### Énoncé de la fonction 5 : BoW Binaire (Encodage)

# **Titre de la fonction :**
# > bag_of_words_binaire(texte, vocabulaire_direct)

# **Consigne :**
# * Créer un vecteur de zéros de la taille du vocabulaire.
# * Pour chaque mot du texte présent dans le vocabulaire, mettre l'index correspondant à 1.
# * Les répétitions de mots n'augmentent pas la valeur (reste à 1).
# * L'ordre des mots dans le texte est perdu.

def bag_of_words_binaire(texte: str, vocabulaire_direct: Dict[str, int]) -> List[int]:
    """
    Transforme un texte en un vecteur binaire (Présence/Absence).
    """
    if not texte or not isinstance(texte, str):
        return [0] * len(vocabulaire_direct)

    # 1. Initialisation du vecteur nul de taille fixe
    taille_vocab = len(vocabulaire_direct)
    vecteur = [0] * taille_vocab
    
    # 2. Tokenisation
    mots = texte.lower().split()
    
    # 3. Remplissage
    for mot in mots:
        index = vocabulaire_direct.get(mot)
        if index is not None:
            # On marque la présence. Même si le mot apparaît 10 fois, on met 1.
            vecteur[index] = 1
            
    return vecteur

### 🔍 Explication des tests unitaires — BoW Binaire
# 1. **Présence simple :** "chat dort" -> indices 0 et 2 à 1 -> [1, 0, 1, 0, 0].
# 2. **Perte de fréquence :** "chat chat chat" -> index 0 à 1 -> [1, 0, 0, 0, 0] (et non 3).
# 3. **Perte d'ordre :** "chat dort" et "dort chat" produisent exactement le même vecteur.

def test_bag_of_words_binaire():
    vocab = {'chat': 0, 'chien': 1, 'dort': 2, 'mange': 3, 'maison': 4}
    
    # Test 1 : Cas standard
    # chat(0), dort(2) -> [1, 0, 1, 0, 0]
    assert bag_of_words_binaire("chat dort", vocab) == [1, 0, 1, 0, 0]
    
    # Test 2 : Répétition (reste binaire)
    # chat(0), chat(0) -> [1, 0, 0, 0, 0]
    assert bag_of_words_binaire("chat chat", vocab) == [1, 0, 0, 0, 0]
    
    # Test 3 : Indifférence à l'ordre
    vec1 = bag_of_words_binaire("chat dort", vocab)
    vec2 = bag_of_words_binaire("dort chat", vocab)
    assert vec1 == vec2
    
    # Test 4 : Mots hors vocabulaire (ne changent rien au vecteur)
    assert bag_of_words_binaire("souris", vocab) == [0, 0, 0, 0, 0]

    print("Test bag_of_words_binaire : OK")


### Énoncé de la fonction 6 : BoW Décodage

# **Titre de la fonction :**
# > bag_of_words_decode(vecteur, vocabulaire_inverse)

# **Consigne :**
# * Convertir un vecteur binaire en liste de mots.
# * Parcourir le vecteur : si une dimension vaut 1, récupérer le mot correspondant.
# * Concaténer les mots trouvés.
# * Note : Le texte reconstruit sera ordonné selon les indices du vocabulaire, 
#   PAS selon l'ordre de la phrase originale.

def bag_of_words_decode(vecteur: List[int], vocabulaire_inverse: Dict[int, str]) -> str:
    """
    Transforme un vecteur binaire en texte (mots présents).
    """
    mots_presents = []
    
    # On parcourt le vecteur avec enumerate pour avoir (index, valeur)
    for index, val in enumerate(vecteur):
        if val == 1:
            mot = vocabulaire_inverse.get(index)
            if mot:
                mots_presents.append(mot)
                
    return " ".join(mots_presents)

### 🔍 Explication des tests unitaires — BoW Décodage
# 1. **Reconstruction ordonnée :** [1, 0, 1, 0, 0] (correspondant à "dort chat") 
#    sera décodé en "chat dort" car index 0 < index 2.
# 2. **Vecteur vide :** [0, 0, 0, 0, 0] -> "".

def test_bag_of_words_decode():
    vocab_inv = {0: 'chat', 1: 'chien', 2: 'dort', 3: 'mange', 4: 'maison'}
    
    # Test 1 : Reconstruction
    # [1, 0, 1, 0, 0] -> indices 0 et 2
    # ATTENTION : L'ordre de sortie dépend des indices (0 avant 2), 
    # donc "chat" sortira avant "dort", même si on pensait à "dort chat".
    vec = [1, 0, 1, 0, 0]
    assert bag_of_words_decode(vec, vocab_inv) == "chat dort"
    
    # Test 2 : Vecteur vide
    assert bag_of_words_decode([0, 0, 0, 0, 0], vocab_inv) == ""

    print("Test bag_of_words_decode : OK")

### Énoncé de la fonction 7 : BoW Occurrences (Comptage)

# **Titre de la fonction :**
# > bag_of_words_occurrences(texte, vocabulaire_direct)

# **Consigne :**
# * Initialiser un vecteur de zéros de la taille du vocabulaire.
# * Parcourir les mots du texte.
# * Pour chaque mot présent dans le vocabulaire, incrémenter la valeur à l'index correspondant.
# * Retourner le vecteur d'entiers.

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

### 🔍 Explication des tests unitaires — BoW Occurrences
# 1. **Comptage simple :** "chat chat" (index 0) doit donner [2, 0, 0, 0, 0].
# 2. **Comptage mixte :** "chat mange chat" (indices 0 et 3) -> [2, 0, 0, 1, 0].
# 3. **Différence avec Binaire :** Là où le binaire plafonne à 1, ici on monte à N.

def test_bag_of_words_occurrences():
    vocab = {'chat': 0, 'chien': 1, 'dort': 2, 'mange': 3, 'maison': 4}
    
    # Test 1 : Répétition d'un mot
    # chat(0) apparaît 2 fois
    res_repet = bag_of_words_occurrences("chat chat", vocab)
    assert res_repet == [2, 0, 0, 0, 0]
    
    # Test 2 : Phrase complexe
    # chat(0) -> 2 fois, mange(3) -> 1 fois
    res_mixte = bag_of_words_occurrences("chat mange chat", vocab)
    assert res_mixte == [2, 0, 0, 1, 0]
    
    # Test 3 : Absence
    res_vide = bag_of_words_occurrences("souris", vocab)
    assert res_vide == [0, 0, 0, 0, 0]

    print("Test bag_of_words_occurrences : OK")


### Énoncé de la fonction 8 : BoW Occurrences Décodage

# **Titre de la fonction :**
# > bag_of_words_occurrences_decode(vecteur, vocabulaire_inverse)

# **Consigne :**
# * Convertir un vecteur d'entiers en texte.
# * Si une dimension vaut N (ex: 2), le mot doit être répété N fois dans le texte de sortie.
# * Comme pour le BoW binaire, l'ordre dépend du vocabulaire, pas du texte original.

def bag_of_words_occurrences_decode(vecteur: List[int], vocabulaire_inverse: Dict[int, str]) -> str:
    """
    Transforme un vecteur de comptage en texte (répétition selon fréquence).
    """
    mots_reconstruits = []
    
    for index, count in enumerate(vecteur):
        if count > 0:
            mot = vocabulaire_inverse.get(index)
            if mot:
                # On ajoute le mot 'count' fois à la liste
                # Ex: ["chat"] * 2 donne ["chat", "chat"]
                mots_reconstruits.extend([mot] * count)
                
    return " ".join(mots_reconstruits)

### 🔍 Explication des tests unitaires — BoW Occurrences Décodage
# 1. **Reconstruction multiple :** [2, 0, 0, 0, 0] doit donner "chat chat".
# 2. **Reconstruction mixte :** [2, 0, 0, 1, 0] doit donner "chat chat mange".
#    Note : "chat" vient avant "mange" car index 0 < index 3.

def test_bag_of_words_occurrences_decode():
    vocab_inv = {0: 'chat', 1: 'chien', 2: 'dort', 3: 'mange', 4: 'maison'}
    
    # Test 1 : Répétition simple
    vec_1 = [2, 0, 0, 0, 0]
    assert bag_of_words_occurrences_decode(vec_1, vocab_inv) == "chat chat"
    
    # Test 2 : Répétition mixte
    # [2, 0, 0, 1, 0] correspond à 2 chats et 1 mange
    vec_2 = [2, 0, 0, 1, 0]
    assert bag_of_words_occurrences_decode(vec_2, vocab_inv) == "chat chat mange"
    
    # Test 3 : Vecteur vide
    assert bag_of_words_occurrences_decode([0, 0, 0, 0, 0], vocab_inv) == ""

    print("Test bag_of_words_occurrences_decode : OK")

### Énoncé de la fonction 9 : Calcul TF

# **Titre de la fonction :**
# > calcul_tf(texte, vocabulaire_direct)

# **Consigne :**
# * Compter les occurrences de chaque mot du vocabulaire dans le texte.
# * Diviser chaque compte par le nombre total de mots (tokens) dans le texte.
# * Retourner une liste de floats.

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

### 🔍 Explication des tests unitaires — TF
# 1. **Normalisation :** "chat dort" (2 mots) -> chat=1/2, dort=1/2 -> [0.5, 0.5].
# 2. **Document long :** "chien mange maison" (3 mots) -> 1/3 chacun -> [0.33..., 0.33..., 0.33...].
# 3. **Répétition :** "chat chat" (2 mots) -> chat=2/2=1.0 -> [1.0, ...].

def test_calcul_tf():
    vocab = {'chat': 0, 'chien': 1, 'dort': 2, 'mange': 3, 'maison': 4}
    
    # Test 1 : Document à 2 mots
    # chat(0), dort(2) -> [0.5, 0, 0.5, 0, 0]
    res_1 = calcul_tf("chat dort", vocab)
    assert res_1 == [0.5, 0.0, 0.5, 0.0, 0.0]
    
    # Test 2 : Document à 3 mots
    # chien(1), mange(3), maison(4) -> 1/3 environ 0.333
    res_2 = calcul_tf("chien mange maison", vocab)
    # On vérifie avec une petite marge d'erreur pour les flottants
    assert abs(res_2[1] - 0.333333) < 0.0001
    assert res_2[0] == 0.0
    
    print("Test calcul_tf : OK")


### c. IDF (Inverse Document Frequency)

# L'IDF mesure la rareté d'un mot. Plus il est rare, plus son poids est élevé.
# Il existe plusieurs formules (classique, smooth, bm25...).
# On utilise ici le logarithme naturel (math.log).

### Énoncé de la fonction 10 : Calcul IDF

# **Titre de la fonction :**
# > calcul_idf(corpus, vocabulaire_direct, methode='smooth')

# **Consigne :**
# * Calculer df_t (nombre de documents contenant le mot t) pour chaque mot.
# * Appliquer la formule correspondant au paramètre 'methode'.
# * Retourner le vecteur IDF (liste de floats).

def calcul_idf(corpus: List[str], vocabulaire_direct: Dict[str, int], methode: str = 'smooth') -> List[float]:
    """
    Calcule l'IDF pour chaque mot du vocabulaire selon la méthode choisie.
    """
    N = len(corpus)
    taille_vocab = len(vocabulaire_direct)
    
    # 1. Calculer df (Document Frequency) pour chaque mot
    # df[index] = nombre de docs contenant le mot d'index 'index'
    df_counts = [0] * taille_vocab
    
    for document in corpus:
        # On utilise un set pour ne compter un mot qu'une seule fois par document
        mots_uniques_doc = set(document.lower().split())
        
        for mot in mots_uniques_doc:
            index = vocabulaire_direct.get(mot)
            if index is not None:
                df_counts[index] += 1
                
    # 2. Calcul de l'IDF selon la méthode
    idf_vector = []
    
    # Pré-calcul pour la méthode 'max'
    max_df = max(df_counts) if df_counts else 1

    for df_t in df_counts:
        valeur_idf = 0.0
        
        if methode == 'classique':
            # log(N / df_t)
            # Attention à la division par zéro si df_t = 0 (mot présent dans le vocab mais pas dans le corpus fourni)
            valeur_idf = math.log(N / df_t) if df_t > 0 else 0.0
            
        elif methode == 'smooth':
            # log(N / (df_t + 1))
            valeur_idf = math.log(N / (df_t + 1))
            
        elif methode == 'smooth_P1':
            # log(N / (df_t + 1)) + 1
            # Note: Votre formule énoncé dit: log( (N / (df_t + 1)) ) + 1 ? 
            # Ou log( (N+1) / (df+1) ) + 1 (sklearn) ?
            # Je respecte strictement votre formule : Ajouter +1 à l’extérieur du log
            valeur_idf = math.log(N / (df_t + 1)) + 1
            
        elif methode == 'logplus':
            # log(1 + N / df_t)
            valeur_idf = math.log(1 + (N / df_t)) if df_t > 0 else 0.0
            
        elif methode == 'max':
            # log( max(df) / (1 + df_t) )
            valeur_idf = math.log(max_df / (1 + df_t))
            
        elif methode == 'bm25':
            # log( (N - df_t) / df_t )
            if df_t > 0:
                argument = (N - df_t) / df_t
                # Protection log : argument doit être > 0
                valeur_idf = math.log(argument) if argument > 0 else 0.0
            else:
                valeur_idf = 0.0
                
        elif methode == 'bm25_smooth':
            # log( (N - df_t + 0.5) / (df_t + 0.5) )
            argument = (N - df_t + 0.5) / (df_t + 0.5)
            valeur_idf = math.log(argument) if argument > 0 else 0.0
            
        idf_vector.append(valeur_idf)
        
    return idf_vector

### 🔍 Explication des tests unitaires — IDF
# On reprend l'exemple du cours :
# N = 4.
# "chat", "chien", "dort", "mange" apparaissent dans 2 documents (df=2).
# "maison" apparaît dans 1 document (df=1).
# Test 'classique' : 
#   - maison : log(4/1) = 1.386
#   - chat : log(4/2) = 0.693

def test_calcul_idf():
    corpus_test = [
        "chat dort",
        "chien dort",
        "chat mange",
        "chien mange maison"
    ]
    vocab = {'chat': 0, 'chien': 1, 'dort': 2, 'mange': 3, 'maison': 4}
    
    # Cas 1 : Méthode Classique
    # Attendu pour maison (index 4, df=1) : log(4/1) = 1.38629...
    idf_classique = calcul_idf(corpus_test, vocab, methode='classique')
    assert abs(idf_classique[4] - 1.386) < 0.001
    # Attendu pour chat (index 0, df=2) : log(4/2) = 0.69314...
    assert abs(idf_classique[0] - 0.693) < 0.001
    
    # Cas 2 : Méthode Smooth
    # Attendu pour maison : log(4 / (1+1)) = log(2) = 0.693
    idf_smooth = calcul_idf(corpus_test, vocab, methode='smooth')
    assert abs(idf_smooth[4] - 0.693) < 0.001
    
    # Cas 3 : BM25 Smooth
    # Attendu pour chat (df=2, N=4) :
    # log( (4 - 2 + 0.5) / (2 + 0.5) ) = log( 2.5 / 2.5 ) = log(1) = 0
    idf_bm25 = calcul_idf(corpus_test, vocab, methode='bm25_smooth')
    assert abs(idf_bm25[0] - 0.0) < 0.001

    print("Test calcul_idf : OK")


### Énoncé de la fonction 11 : Calcul Matrice TF-IDF

# **Titre de la fonction :**
# > calcul_tf_idf(corpus, vocabulaire_direct, methode_idf='smooth')

# **Consigne :**
# * Calculer le vecteur IDF global à partir du corpus.
# * Pour chaque document du corpus :
#     1. Calculer son vecteur TF.
#     2. Multiplier chaque terme du TF par le terme correspondant de l'IDF.
# * Retourner la matrice (liste de listes).

def calcul_tf_idf(corpus: List[str], vocabulaire_direct: Dict[str, int], methode_idf: str = 'smooth') -> List[List[float]]:
    """
    Transforme le corpus en une matrice de vecteurs TF-IDF.
    """
    # 1. Calcul du vecteur IDF global (une seule fois pour tout le corpus)
    # On utilise la fonction définie à l'étape précédente
    vecteur_idf = calcul_idf(corpus, vocabulaire_direct, methode=methode_idf)
    
    matrice_resultat = []
    
    # 2. Traitement document par document
    for document in corpus:
        # Calcul du TF pour ce document
        vecteur_tf = calcul_tf(document, vocabulaire_direct)
        
        # Calcul du TF-IDF : TF * IDF (terme à terme)
        vecteur_tf_idf = []
        for i in range(len(vecteur_tf)):
            valeur = vecteur_tf[i] * vecteur_idf[i]
            vecteur_tf_idf.append(valeur)
            
        matrice_resultat.append(vecteur_tf_idf)
        
    return matrice_resultat

### 🔍 Explication des tests unitaires — TF-IDF
# Reprenons l'exemple manuel de l'énoncé.
# Corpus : "chat dort", "chien dort"...
# Vocab : chat(0), chien(1), dort(2)...
# Pour "chat dort" : 
#   - TF("chat") = 0.5
#   - IDF("chat") (classique) ≈ 0.693
#   - TF-IDF = 0.5 * 0.693 = 0.3465

def test_calcul_tf_idf():
    corpus_test = [
        "chat dort",
        "chien dort",
        "chat mange",
        "chien mange maison"
    ]
    vocab = {'chat': 0, 'chien': 1, 'dort': 2, 'mange': 3, 'maison': 4}
    
    # On utilise la méthode 'classique' pour coller aux chiffres de l'exemple de l'énoncé
    matrice = calcul_tf_idf(corpus_test, vocab, methode_idf='classique')
    
    # Vérification du premier document "chat dort" (index 0)
    # Index 0 (chat) : attendu ~0.347
    doc1 = matrice[0]
    assert abs(doc1[0] - 0.3465) < 0.001
    # Index 1 (chien) : attendu 0
    assert doc1[1] == 0.0
    # Index 2 (dort) : attendu ~0.347
    assert abs(doc1[2] - 0.3465) < 0.001
    
    print("Test calcul_tf_idf : OK")


### e. BM25 (Best Matching 25)

# BM25 améliore TF-IDF en gérant la saturation de la fréquence (k1)
# et la normalisation par la longueur moyenne des documents (b).
# Formule complexe impliquant IDF, TF (fréquence brute), longueur doc et longueur moyenne.

### Énoncé de la fonction 12 : Calcul BM25

# **Titre de la fonction :**
# > calcul_bm25(corpus, vocabulaire_direct, k1=1.5, b=0.75)

# **Consigne :**
# * Calculer la longueur moyenne des documents (avgdl).
# * Calculer l'IDF (variante BM25 smooth).
# * Pour chaque document :
#     1. Récupérer les fréquences brutes (pas le TF normalisé 0-1, mais le comptage).
#     2. Appliquer la formule BM25 pour chaque mot.

def calcul_bm25(corpus: List[str], vocabulaire_direct: Dict[str, int], k1: float = 1.5, b: float = 0.75) -> List[List[float]]:
    """
    Calcule la matrice de vecteurs BM25 pour le corpus.
    """
    # 1. Pré-calculs statistiques sur le corpus
    n_docs = len(corpus)
    # Calcul des longueurs de chaque document
    longueurs_docs = [len(doc.lower().split()) for doc in corpus]
    # Calcul de la longueur moyenne (avgdl)
    avgdl = sum(longueurs_docs) / n_docs if n_docs > 0 else 1
    
    # Calcul de l'IDF spécifique BM25 (smooth)
    vecteur_idf_bm25 = calcul_idf(corpus, vocabulaire_direct, methode='bm25_smooth')
    
    matrice_bm25 = []
    
    # 2. Calcul pour chaque document
    for i, document in enumerate(corpus):
        # On a besoin des fréquences brutes (comptage)
        # On réutilise bag_of_words_occurrences définie précédemment
        freqs_brutes = bag_of_words_occurrences(document, vocabulaire_direct)
        
        longueur_doc = longueurs_docs[i]
        vecteur_doc = []
        
        # 3. Application de la formule mot par mot
        for j in range(len(vocabulaire_direct)):
            tf_val = freqs_brutes[j] # Fréquence brute f(t,d)
            idf_val = vecteur_idf_bm25[j]
            
            if tf_val > 0:
                numerateur = tf_val * (k1 + 1)
                denominateur = tf_val + k1 * (1 - b + b * (longueur_doc / avgdl))
                
                score = idf_val * (numerateur / denominateur)
            else:
                score = 0.0
                
            vecteur_doc.append(score)
            
        matrice_bm25.append(vecteur_doc)
        
    return matrice_bm25

### 🔍 Explication des tests unitaires — BM25
# Le calcul manuel est complexe, nous allons vérifier la logique :
# 1. Un mot absent doit avoir un score de 0.
# 2. Un mot présent doit avoir un score positif (si IDF > 0).
# 3. Test de saturation : avec k1, augmenter la fréquence augmente le score 
#    mais de moins en moins vite (contrairement au TF simple qui est linéaire).
def test_calcul_bm25():
    # On ajoute des phrases vides/neutres pour que N soit plus grand (N=5).
    # Ainsi, "chat" (présent dans 2 docs) aura un IDF positif car df < N/2.
    corpus_test = [
        "chat dort",            # Doc 0
        "chat chat chat dort",  # Doc 1
        "rien",                 # Doc 2 (remplissage)
        "rien du tout",         # Doc 3 (remplissage)
        "vide"                  # Doc 4 (remplissage)
    ]
    
    # Le vocabulaire reste le même, les mots des docs de remplissage seront ignorés
    vocab = {'chat': 0, 'dort': 1}
    
    # Calcul
    matrice = calcul_bm25(corpus_test, vocab, k1=1.5, b=0.75)
    
    # Test 1 : Zéros pour mots absents
    # (Par exemple "chat" n'est pas dans le doc "rien", son score doit être 0)
    score_chat_doc_vide = matrice[2][0]
    assert score_chat_doc_vide == 0.0
    
    # Test 2 : Comparaison de fréquence ("chat")
    # Maintenant que l'IDF est positif, plus il y a de "chat", plus le score monte.
    score_chat_doc1 = matrice[0][0] # 1 occurrence
    score_chat_doc2 = matrice[1][0] # 3 occurrences
    
    # On vérifie que le document avec plus de "chat" a un score plus élevé
    assert score_chat_doc2 > score_chat_doc1
    
    # Test 3 : Normalisation de longueur ("dort")
    # "dort" apparaît 1 fois dans Doc 0 (court) et 1 fois dans Doc 1 (long).
    # BM25 pénalise le document long.
    score_dort_doc1 = matrice[0][1] # Doc court
    score_dort_doc2 = matrice[1][1] # Doc long
    
    assert score_dort_doc1 > score_dort_doc2

    print("Test calcul_bm25 : OK")
### Énoncé de la fonction 13 : Normalisation L1

# **Titre de la fonction :**
# > normaliser_L1(v)

# **Consigne :**
# * Calculer la somme totale des éléments du vecteur.
# * Diviser chaque élément par cette somme.
# * Gérer le cas du vecteur nul (division par zéro).

def normaliser_L1(v: List[float]) -> List[float]:
    """
    Normalise un vecteur selon la norme L1 (somme des éléments = 1).
    """
    if not v:
        return []
        
    somme_totale = sum(v)
    
    # Protection contre la division par zéro (vecteur nul)
    if somme_totale == 0:
        return v[:] # Retourne une copie à l'identique (ou des zéros)
        
    return [x / somme_totale for x in v]

### 🔍 Explication des tests unitaires — Normalisation L1
# 1. **Somme unitaire :** [2, 2] -> Somme=4 -> [0.5, 0.5].
# 2. **Proportions :** [1, 3] -> Somme=4 -> [0.25, 0.75].

def test_normaliser_L1():
    # Cas 1 : Vecteur simple
    vec = [2.0, 2.0]
    res = normaliser_L1(vec)
    assert res == [0.5, 0.5]
    assert sum(res) == 1.0
    
    # Cas 2 : Proportions
    vec2 = [1.0, 3.0]
    res2 = normaliser_L1(vec2)
    assert res2 == [0.25, 0.75]
    
    # Cas 3 : Vecteur nul
    assert normaliser_L1([0, 0]) == [0, 0]

    print("Test normaliser_L1 : OK")


### b. Normalisation L2 (Norme Euclidienne = 1)

# Indispensable pour la similarité Cosinus.
# Le vecteur est projeté sur le cercle unitaire (hypersphère de rayon 1).
# Formule : v_i = v_i / sqrt(sum(v_j^2))

### Énoncé de la fonction 14 : Normalisation L2

# **Titre de la fonction :**
# > normaliser_L2(v)

# **Consigne :**
# * Calculer la somme des carrés des éléments.
# * Prendre la racine carrée de cette somme (la norme).
# * Diviser chaque élément par cette norme.

def normaliser_L2(v: List[float]) -> List[float]:
    """
    Normalise un vecteur selon la norme L2 (longueur du vecteur = 1).
    """
    if not v:
        return []
        
    somme_carres = sum(x**2 for x in v)
    norme = math.sqrt(somme_carres)
    
    if norme == 0:
        return v[:]
        
    return [x / norme for x in v]

### 🔍 Explication des tests unitaires — Normalisation L2
# 1. **Triangle rectangle (3-4-5) :** [3, 4] -> Norme=5 -> [0.6, 0.8].
#    Vérification : 0.6² + 0.8² = 0.36 + 0.64 = 1.
# 2. **Unitaire :** [1, 0] -> [1, 0].

def test_normaliser_L2():
    # Cas 1 : 3-4-5
    vec = [3.0, 4.0]
    res = normaliser_L2(vec)
    assert res == [0.6, 0.8]
    
    # Vérification que la norme du résultat est bien 1
    norme_res = math.sqrt(sum(x**2 for x in res))
    assert abs(norme_res - 1.0) < 0.0001
    
    # Cas 2 : Vecteur nul
    assert normaliser_L2([0, 0]) == [0, 0]

    print("Test normaliser_L2 : OK")


### c. Normalisation Min–Max (Mise à l'échelle 0-1)

# Ramène toutes les valeurs dans l'intervalle [0, 1].
# Formule : (x - min) / (max - min)

### Énoncé de la fonction 15 : Normalisation Min-Max

# **Titre de la fonction :**
# > normaliser_minmax(v)

# **Consigne :**
# * Trouver le min et le max du vecteur.
# * Calculer l'étendue (max - min).
# * Appliquer la formule. Attention si max == min (division par zéro).

def normaliser_minmax(v: List[float]) -> List[float]:
    """
    Normalise un vecteur pour que ses valeurs soient entre 0 et 1.
    """
    if not v:
        return []
        
    val_min = min(v)
    val_max = max(v)
    etendue = val_max - val_min
    
    if etendue == 0:
        # Si tous les nombres sont identiques, on retourne souvent un vecteur de 0
        # ou un vecteur de 1, ou on laisse tel quel. Convention ici : 0.
        return [0.0] * len(v)
        
    return [(x - val_min) / etendue for x in v]

### 🔍 Explication des tests unitaires — Min-Max
# 1. **Échelle standard :** [10, 20, 30]. Min=10, Max=30, Étendue=20.
#    - 10 -> (10-10)/20 = 0.0
#    - 20 -> (20-10)/20 = 0.5
#    - 30 -> (30-10)/20 = 1.0

def test_normaliser_minmax():
    vec = [10.0, 20.0, 30.0]
    res = normaliser_minmax(vec)
    assert res == [0.0, 0.5, 1.0]
    
    # Cas valeurs identiques
    assert normaliser_minmax([5, 5, 5]) == [0.0, 0.0, 0.0]

    print("Test normaliser_minmax : OK")


### d. Standardisation (Z-score)

# Centre sur la moyenne (0) et réduit l'écart-type (1).
# Formule : (x - moy) / ecart_type

### Énoncé de la fonction 16 : Standardisation Z-score

# **Titre de la fonction :**
# > standardiser_zscore(v)

# **Consigne :**
# * Calculer la moyenne.
# * Calculer la variance puis l'écart-type (sqrt(variance)).
# * Appliquer la formule.

def standardiser_zscore(v: List[float]) -> List[float]:
    """
    Standardise le vecteur (moyenne=0, écart-type=1).
    """
    if not v:
        return []
    
    n = len(v)
    moyenne = sum(v) / n
    
    # Variance (population) : somme((x - moy)^2) / n
    variance = sum((x - moyenne)**2 for x in v) / n
    ecart_type = math.sqrt(variance)
    
    if ecart_type == 0:
        return [0.0] * n
        
    return [(x - moyenne) / ecart_type for x in v]

### 🔍 Explication des tests unitaires — Z-score
# 1. **Symétrie :** [1, 2, 3]. Moyenne=2. 
#    - Variance = ((1-2)² + (2-2)² + (3-2)²) / 3 = (1+0+1)/3 = 2/3 ≈ 0.666
#    - Ecart-type = sqrt(0.666) ≈ 0.816
#    - 2 devient 0. 1 devient négatif, 3 devient positif.

def test_standardiser_zscore():
    vec = [1.0, 2.0, 3.0]
    res = standardiser_zscore(vec)
    
    # La valeur centrale (moyenne) doit être 0
    assert abs(res[1] - 0.0) < 0.0001
    # La première valeur doit être négative
    assert res[0] < 0
    # La dernière valeur doit être positive
    assert res[2] > 0
    
    # Vérification post-standardisation : moyenne doit être ~0 et std_dev ~1
    moy_res = sum(res) / len(res)
    var_res = sum((x - moy_res)**2 for x in res) / len(res)
    
    assert abs(moy_res) < 0.0001
    assert abs(var_res - 1.0) < 0.0001 # variance de 1 => écart-type de 1

    print("Test standardiser_zscore : OK")

### Énoncé de la fonction 17 : Indexation N-grammes

# **Titre de la fonction :**
# > construire_dictionnaire_ngrammes(vocabulaire_ngrammes)

# **Consigne :**
# * Recevoir une liste (ou un ensemble) de tuples représentant les n-grammes.
# * Trier cette liste pour garantir que les indices sont déterministes (toujours les mêmes à chaque exécution).
# * Construire le dictionnaire direct {tuple : index}.
# * Construire le dictionnaire inverse {index : tuple}.
# * Retourner les deux dictionnaires.

def construire_dictionnaire_ngrammes(vocabulaire_ngrammes: List[Tuple[str, ...]]) -> Tuple[List[Tuple[str, ...]], Dict[Tuple[str, ...], int], Dict[int, Tuple[str, ...]]]:
    """
    Construit les dictionnaires d'indexation pour une liste de n-grammes.
    
    Args:
        vocabulaire_ngrammes: Liste de tuples (ex: [('chat', 'dort'), ...])
        
    Returns:
        Un tuple contenant (liste_triee, dictionnaire_direct, dictionnaire_inverse)
    """
    # 1. Tri pour le déterminisme (important pour reproduire les résultats)
    # On convertit en liste si ce n'est pas le cas, puis on trie
    vocab_trie = sorted(list(vocabulaire_ngrammes))
    
    dict_direct = {}
    dict_inverse = {}
    
    # 2. Assignation des indices
    for index, ngramme in enumerate(vocab_trie):
        dict_direct[ngramme] = index
        dict_inverse[index] = ngramme
        
    return vocab_trie, dict_direct, dict_inverse

### 🔍 Explication des tests unitaires — Indexation N-grammes
# 1. **Indexation correcte :** Vérifier que chaque n-gramme a un ID unique.
# 2. **Correspondance :** Vérifier que l'ID 0 pointe bien vers le premier n-gramme alphabétique.
# 3. **Réversibilité :** direct[ngram] -> id -> inverse[id] doit redonner ngram.

def test_construire_dictionnaire_ngrammes():
    # Données simulées (issues d'une étape précédente imaginaire)
    # Vocabulaire de bi-grammes
    input_ngrammes = [
        ("pomme", "rouge"),
        ("chat", "dort"),
        ("mange", "souris")
    ]
    
    # Exécution
    vocab_trie, d_direct, d_inverse = construire_dictionnaire_ngrammes(input_ngrammes)
    
    # Test 1 : Vérification de la taille
    assert len(d_direct) == 3
    assert len(d_inverse) == 3
    
    # Test 2 : Vérification du tri et de l'index 0
    # "chat" arrive avant "mange" et "pomme" dans l'ordre alphabétique
    assert d_direct[("chat", "dort")] == 0
    assert d_inverse[0] == ("chat", "dort")
    
    # Test 3 : Vérification de l'intégrité
    ngramme_test = ("pomme", "rouge")
    idx = d_direct[ngramme_test]
    assert d_inverse[idx] == ngramme_test
    
    # Test 4 : Input vide
    l1, d1, d2 = construire_dictionnaire_ngrammes([])
    assert d1 == {} and d2 == {}

    print("Test construire_dictionnaire_ngrammes : OK")

### Énoncé de la fonction 18 : BoW N-grammes

# **Titre de la fonction :**
# > encoder_bow_ngrammes(texte, n, vocab_ng, dico_direct_ng, type='binaire')

# **Consigne :**
# * Découper le texte en tokens.
# * Générer les n-grammes du texte (tuples de n mots).
# * Initialiser un vecteur de zéros de la taille du dictionnaire.
# * Pour chaque n-gramme généré :
#     * S'il est dans le dictionnaire (dico_direct_ng) :
#         * Si type='binaire' : mettre 1.
#         * Si type='occurrences' : incrémenter.

def encoder_bow_ngrammes(texte: str, n: int, vocab_ng: List[Tuple[str, ...]], dico_direct_ng: Dict[Tuple[str, ...], int], type_enc: str = 'binaire') -> List[int]:
    """
    Encode un texte en vecteur BoW (binaire ou comptage) basé sur les n-grammes.
    """
    if not texte:
        return [0] * len(dico_direct_ng)
        
    # 1. Tokenisation et extraction des n-grammes candidats
    tokens = texte.lower().split()
    candidats = []
    
    # On ne peut faire des n-grammes que si le texte est assez long
    if len(tokens) >= n:
        for i in range(len(tokens) - n + 1):
            # On crée un tuple de n mots (la fenêtre glissante)
            ngram = tuple(tokens[i : i + n])
            candidats.append(ngram)
            
    # 2. Initialisation du vecteur
    taille_vec = len(dico_direct_ng)
    vecteur = [0] * taille_vec
    
    # 3. Remplissage
    for ngram in candidats:
        idx = dico_direct_ng.get(ngram)
        
        if idx is not None:
            if type_enc == 'binaire':
                vecteur[idx] = 1
            elif type_enc == 'occurrences':
                vecteur[idx] += 1
                
    return vecteur

### 🔍 Explication des tests unitaires — BoW N-grammes
# Vocabulaire n-grammes (bi-grammes) : {('chat', 'dort'): 0, ('dort', 'bien'): 1}
# Texte : "Le chat dort bien"
# N-grammes du texte : [('le', 'chat'), ('chat', 'dort'), ('dort', 'bien')]
# ('le', 'chat') -> inconnu.
# ('chat', 'dort') -> connu (idx 0).
# ('dort', 'bien') -> connu (idx 1).
# Résultat attendu : [1, 1]

def test_encoder_bow_ngrammes():
    # Setup
    vocab_ng = [('chat', 'dort'), ('dort', 'bien')]
    dico_direct = {('chat', 'dort'): 0, ('dort', 'bien'): 1}
    texte = "Le chat dort bien"
    
    # Test Binaire
    res_bin = encoder_bow_ngrammes(texte, 2, vocab_ng, dico_direct, type_enc='binaire')
    assert res_bin == [1, 1]
    
    # Test Occurrences (avec répétition)
    texte_repet = "chat dort et chat dort" 
    # n-grammes : ('chat', 'dort'), ('dort', 'et'), ('et', 'chat'), ('chat', 'dort')
    res_occ = encoder_bow_ngrammes(texte_repet, 2, vocab_ng, dico_direct, type_enc='occurrences')
    # ('chat', 'dort') apparait 2 fois -> index 0 vaut 2
    # ('dort', 'bien') n'apparait pas -> index 1 vaut 0
    assert res_occ == [2, 0]
    
    print("Test encoder_bow_ngrammes : OK")


### Énoncé de la fonction 19 : TF N-grammes

# **Titre de la fonction :**
# > encoder_tf_ngrammes(texte, n, vocab_ng, dico_direct_ng)

# **Consigne :**
# * Compter les occurrences des n-grammes connus.
# * Diviser par le nombre TOTAL de n-grammes générés dans le texte (et non le nombre de mots).
# * Retourner un vecteur de floats.

def encoder_tf_ngrammes(texte: str, n: int, vocab_ng: List[Tuple[str, ...]], dico_direct_ng: Dict[Tuple[str, ...], int]) -> List[float]:
    """
    Encode un texte en vecteur TF (fréquence relative) des n-grammes.
    """
    if not texte:
        return [0.0] * len(dico_direct_ng)
        
    tokens = texte.lower().split()
    candidats = []
    
    if len(tokens) >= n:
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            candidats.append(ngram)
            
    nb_total_ngrams_texte = len(candidats)
    taille_vec = len(dico_direct_ng)
    vecteur_tf = [0.0] * taille_vec
    
    if nb_total_ngrams_texte == 0:
        return vecteur_tf

    # Comptage puis division
    for ngram in candidats:
        idx = dico_direct_ng.get(ngram)
        if idx is not None:
            vecteur_tf[idx] += 1.0
            
    # Normalisation
    vecteur_tf = [x / nb_total_ngrams_texte for x in vecteur_tf]
    
    return vecteur_tf

### 🔍 Explication des tests unitaires — TF N-grammes
# Texte : "chat dort bien" (tokens: 3)
# Bi-grammes générés (n=2) : ('chat', 'dort'), ('dort', 'bien'). Total = 2.
# Vocabulaire : {('chat', 'dort'): 0} (on ignore 'dort bien' pour l'exemple).
# 'chat dort' apparait 1 fois sur 2 bi-grammes totaux -> TF = 0.5.

def test_encoder_tf_ngrammes():
    dico_direct = {('chat', 'dort'): 0}
    texte = "chat dort bien" # ngrammes: (chat, dort), (dort, bien)
    
    res = encoder_tf_ngrammes(texte, 2, [], dico_direct)
    
    # Index 0 ('chat dort') : 1 occurrence / 2 bigrammes totaux = 0.5
    assert res == [0.5]
    
    print("Test encoder_tf_ngrammes : OK")


### Énoncé de la fonction 20 : TF-IDF N-grammes

# **Titre de la fonction :**
# > encoder_tfidf_ngrammes(texte, n, vocab_ng, dico_direct_ng, idf_ng)

# **Consigne :**
# * Calculer le vecteur TF des n-grammes (via la fonction précédente).
# * Recevoir le vecteur IDF pré-calculé (idf_ng) correspondant aux n-grammes.
# * Multiplier terme à terme.

def encoder_tfidf_ngrammes(texte: str, n: int, vocab_ng: List[Tuple[str, ...]], dico_direct_ng: Dict[Tuple[str, ...], int], idf_ng: List[float]) -> List[float]:
    """
    Encode un texte en vecteur TF-IDF basé sur les n-grammes.
    """
    # 1. Calcul du TF
    vec_tf = encoder_tf_ngrammes(texte, n, vocab_ng, dico_direct_ng)
    
    # 2. Application de l'IDF
    # On suppose que idf_ng a la même taille que vec_tf
    vec_tfidf = []
    for i in range(len(vec_tf)):
        valeur = vec_tf[i] * idf_ng[i]
        vec_tfidf.append(valeur)
        
    return vec_tfidf

### 🔍 Explication des tests unitaires — TF-IDF N-grammes
# TF calculé précédemment : [0.5]
# IDF fictif pour ('chat', 'dort') : 1.2
# Résultat : 0.5 * 1.2 = 0.6

def test_encoder_tfidf_ngrammes():
    dico_direct = {('chat', 'dort'): 0}
    idf_fictif = [1.2]
    texte = "chat dort bien" # TF = [0.5]
    
    res = encoder_tfidf_ngrammes(texte, 2, [], dico_direct, idf_fictif)
    
    assert abs(res[0] - 0.6) < 0.0001
    
    print("Test encoder_tfidf_ngrammes : OK")


### Énoncé de la fonction 21 : Concaténation des vocabulaires

# **Titre de la fonction :**
# > concatener_vocab_ngrammes(liste_vocab_ng)

# **Consigne :**
# * Recevoir une liste contenant plusieurs listes de n-grammes.
#   Exemple : [ [('chat',), ('chien',)], [('chat', 'dort'), ('chien', 'mange')] ]
# * Aplatir ces listes en une seule.
# * Supprimer les doublons éventuels (via un set).
# * Trier la liste finale pour garantir l'ordre (déterminisme).
# * Reconstruire les dictionnaires d'index (direct et inverse).

def concatener_vocab_ngrammes(liste_vocab_ng: List[List[Tuple[str, ...]]]) -> Tuple[List[Tuple[str, ...]], Dict[Tuple[str, ...], int], Dict[int, Tuple[str, ...]]]:
    """
    Combine plusieurs vocabulaires de n-grammes en un seul ensemble indexé.
    
    Args:
        liste_vocab_ng: Liste de listes de n-grammes (ex: [vocab_uni, vocab_bi])
        
    Returns:
        vocab_final: Liste triée unique
        dico_direct_final: {ngramme: index}
        dico_inverse_final: {index: ngramme}
    """
    ensemble_temp = set()
    
    # 1. Fusion et dédoublonnage
    for vocab in liste_vocab_ng:
        for ngramme in vocab:
            # On s'assure d'ajouter des éléments hashables (tuples)
            ensemble_temp.add(ngramme)
            
    # 2. Tri (essentiel pour que l'index 0 soit toujours le même à chaque exécution)
    # Note : Python gère le tri des tuples élément par élément
    vocab_final = sorted(list(ensemble_temp))
    
    dico_direct_final = {}
    dico_inverse_final = {}
    
    # 3. Indexation
    for index, ngramme in enumerate(vocab_final):
        dico_direct_final[ngramme] = index
        dico_inverse_final[index] = ngramme
        
    return vocab_final, dico_direct_final, dico_inverse_final

### 🔍 Explication des tests unitaires — Combinaison
# 1. **Fusion simple :** Unigrammes [A, B] + Bigrammes [C, D] -> [A, B, C, D].
# 2. **Doublons :** Si un n-gramme apparait dans deux listes (rare mais possible selon la construction), 
#    il ne doit apparaître qu'une fois.
# 3. **Indexation cohérente :** Le dictionnaire final doit couvrir l'ensemble de la fusion.

def test_concatener_vocab_ngrammes():
    # Simulation de données
    # Attention : pour que le tri fonctionne bien, il vaut mieux que tout soit des tuples
    vocab_uni = [('chat',), ('chien',)]
    vocab_bi = [('chat', 'dort'), ('chien', 'mange')]
    
    # Exécution
    vocab_final, d_direct, d_inv = concatener_vocab_ngrammes([vocab_uni, vocab_bi])
    
    # Test 1 : Taille (2 unigrammes + 2 bigrammes = 4)
    assert len(vocab_final) == 4
    assert len(d_direct) == 4
    
    # Test 2 : Vérification du contenu
    assert ('chat',) in d_direct
    assert ('chat', 'dort') in d_direct
    
    # Test 3 : Doublons (Simulation d'un cas sale)
    vocab_sale_1 = [('a',), ('b',)]
    vocab_sale_2 = [('b',), ('c',)] # 'b' est en doublon
    v_clean, _, _ = concatener_vocab_ngrammes([vocab_sale_1, vocab_sale_2])
    
    # Doit contenir a, b, c (taille 3) et non a, b, b, c (taille 4)
    assert len(v_clean) == 3
    assert ('b',) in v_clean

    print("Test concatener_vocab_ngrammes : OK")


# --- Exécution des tests ---
if __name__ == "__main__":
    print("\n--- Démarrage de la séquence de tests ---")
    test_concatener_vocab_ngrammes()
    test_encoder_bow_ngrammes()
    test_encoder_tf_ngrammes()
    test_encoder_tfidf_ngrammes()
    test_construire_dictionnaire_ngrammes()
    test_normaliser_L1()
    test_normaliser_L2()
    test_normaliser_minmax()
    test_standardiser_zscore()
    test_calcul_tf_idf()
    test_calcul_bm25()
    test_calcul_tf()
    test_calcul_idf()
    test_bag_of_words_occurrences()
    test_bag_of_words_occurrences_decode()
    test_bag_of_words_binaire()
    test_bag_of_words_decode()
    test_one_hot_encode()
    test_one_hot_decode()
    test_texte_vers_indices()
    test_indices_vers_texte()
    print("--- Tous les tests sont passés avec succès ---\n")