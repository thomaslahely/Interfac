import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

"""
### 📝 Énoncé de la fonction 2

**Titre de la fonction :**
> segmenter_phrases(texte, abreviations, option)

**Consigne :**
Cette fonction segmente un texte en phrases en respectant la ponctuation (. ? !) 
et en évitant les découpages erronés dus aux abréviations, décimaux, dates ou sigles. 
Le découpage se fait à partir des signes de ponctuation suivis d’un espace ou d’un retour à la ligne.

Elle doit gérer des options via un dictionnaire pour activer ou désactiver certaines protections (décimaux, dates, sigles).
"""

def segmenter_phrases(texte, abreviations=[], option=None):
    """
    Description :
    Segmente un texte en phrases en utilisant la ponctuation, tout en protégeant
    certains cas particuliers (abréviations, nombres, etc.) selon les options.
    
    Paramètres :
        texte (str): Le texte à segmenter.
        abreviations (list): Liste de chaînes (ex: ["Dr.", "M."]).
        option (dict): Dictionnaire de configuration.
            Clés attendues : "gerer_decimaux", "gerer_dates", "gerer_sigles" (booléens).
            
    Retour :
        list[str]: Une liste de phrases.
    """
    if option is None:
        option = {"gerer_decimaux": False, "gerer_dates": False, "gerer_sigles": False}
    
    # On travaille sur une copie pour ne pas modifier l'original en place si c'était une ref
    texte_traite = texte
    
    
    MARQUEUR = "<DOT>"

    # 1. Gestion des dates (ex: 12.05.2025) -> format JJ.MM.AAAA
    if option.get("gerer_dates", False):
        # Regex simple pour date JJ.MM.AAAA
        pattern_date = r'(\d{2})\.(\d{2})\.(\d{4})'
        texte_traite = re.sub(pattern_date, rf'\1{MARQUEUR}\2{MARQUEUR}\3', texte_traite)

    # 2. Gestion des décimaux (ex: 3.14)
    if option.get("gerer_decimaux", False):
        # Un chiffre, un point, un chiffre
        pattern_decimal = r'(\d+)\.(\d+)'
        texte_traite = re.sub(pattern_decimal, rf'\1{MARQUEUR}\2', texte_traite)

    # 3. Gestion des abréviations fournies (ex: "Dr.", "M.")
    # On trie par longueur décroissante pour éviter qu'une abréviation courte n'écrase une longue
    abreviations_sorted = sorted(abreviations, key=len, reverse=True)
    for abbr in abreviations_sorted:
        if abbr.endswith('.'):
            # On cherche l'abréviation dans le texte (insensible à la casse ou non selon besoin)
            # Ici on fait un remplacement littéral exact pour l'exercice
            # On échappe le point pour le regex
            abbr_escaped = re.escape(abbr)
            # On remplace le point final par le marqueur
            replacement = abbr[:-1] + MARQUEUR
            texte_traite = re.sub(abbr_escaped, replacement, texte_traite)

    if option.get("gerer_sigles", False):
        # Regex pour une lettre majuscule suivie d'un point, répété au moins 2 fois
        # Ex: A.B. ou A.B.C.
        def replace_sigle(match):
            return match.group(0).replace('.', MARQUEUR)
        
        pattern_sigle = r'([A-Z]\.)+[A-Z]\.?' 
        texte_traite = re.sub(pattern_sigle, replace_sigle, texte_traite)


    # Le pattern cherche [.?!] suivi d'un espace (\s) ou de la fin du texte ($)
    # On utilise split mais on veut garder le séparateur pour le recoller ou split intelligemment.
    # Ici, une approche simple est de remplacer la ponctuation de fin par "PONCT|SPLIT"
    
    pattern_split = r'([.?!])(\s+|$)'
    # On remplace par : le signe de ponctuation + un marqueur de split spécial
    texte_segmente = re.sub(pattern_split, r'\1<SPLIT>', texte_traite)
    
    phrases_brutes = texte_segmente.split('<SPLIT>')
    
    # On enlève les espaces en trop et on remet les points masqués
    resultat = []
    for phrase in phrases_brutes:
        phrase = phrase.strip()
        if phrase:
            # On restaure les points protégés
            phrase_finale = phrase.replace(MARQUEUR, '.')
            resultat.append(phrase_finale)

    return resultat


def test_segmenter_phrases():
    print("\n--- Démarrage des tests pour segmenter_phrases ---")

    # --- Cas 1 : Segmentation simple ---
    texte_simple = "Bonjour tout le monde. Comment allez-vous ? Très bien !"
    attendu_simple = [
        "Bonjour tout le monde.",
        "Comment allez-vous ?",
        "Très bien !"
    ]
    assert segmenter_phrases(texte_simple) == attendu_simple
    print("✅ Test 1 (Simple) passé.")

    # --- Cas 2 : Abréviations ---
    texte_abbr = "Dr. Martin habite à Reims. M. Dupont aussi."
    liste_abbr = ["Dr.", "M."]
    # Si on fournit les abréviations, ça ne doit pas couper après Dr.
    attendu_abbr = [
        "Dr. Martin habite à Reims.",
        "M. Dupont aussi."
    ]
    res_abbr = segmenter_phrases(texte_abbr, abreviations=liste_abbr)
    assert res_abbr == attendu_abbr
    print("✅ Test 2 (Abréviations) passé.")

    # --- Cas 3 : Options activées (Dates et Décimaux) ---
    texte_complexe = "Le 12.05.2025 est une date. Pi vaut 3.14 environ."
    options = {"gerer_decimaux": True, "gerer_dates": True, "gerer_sigles": False}
    
    attendu_complexe = [
        "Le 12.05.2025 est une date.",
        "Pi vaut 3.14 environ."
    ]
    res_complexe = segmenter_phrases(texte_complexe, option=options)
    assert res_complexe == attendu_complexe
    print("✅ Test 3 (Options Dates/Décimaux) passé.")

    # --- Cas 4 : Sigles (Option activée) ---
    texte_sigle = "Le P.D.G. a démissionné. C'est triste."
    options_sigle = {"gerer_sigles": True}
    attendu_sigle = [
        "Le P.D.G. a démissionné.",
        "C'est triste."
    ]
    res_sigle = segmenter_phrases(texte_sigle, option=options_sigle)
    assert res_sigle == attendu_sigle
    print("✅ Test 4 (Sigles) passé.")

    # --- Cas 5 : Échec attendu si option désactivée ---
    # Si on ne gère pas les décimaux, "3.14" sera vu comme "3." (fin de phrase) "14" (nouvelle phrase)
    texte_fail = "Pi vaut 3.14."
    res_fail = segmenter_phrases(texte_fail, option={"gerer_decimaux": False})
   
    texte_fail_v2 = "Il est 17. 30 minutes." # Ici le point est suivi d'un espace
    res_fail_v2 = segmenter_phrases(texte_fail_v2, option={"gerer_decimaux": False})
    # Sans protection, ça devrait couper
    assert len(res_fail_v2) == 2 
    print("✅ Test 5 (Sans protection) passé.")

    print("✅ Tous les tests unitaires pour segmenter_phrases sont passés avec succès !")


"""
### 📝 Énoncé des fonctions - Partie 2 : Tokenisation en mots

Cette section concerne la segmentation des phrases en mots (tokens), la gestion
des corpus et l'analyse basique (Hapax).
"""

def segmenter_mots(phrase, balise=False):
    """
    Découpe une phrase en tokens.
    Gère les mots composés (avec tirets) comme un seul token.
    Sépare la ponctuation si elle n'est pas déjà espacée.
    
    Paramètres:
        phrase (str): La phrase à tokeniser.
        balise (bool): Si False, tente de filtrer les balises HTML/XML simples.
                       Si True, garde tout ce qui ressemble à un token.
    
    Retour:
        list[str]: Liste des mots/tokens.
    """
    # Regex explicative :
    # \w+(?:-\w+)* : Un mot composé de lettres/chiffres, pouvant contenir des tirets internes (ex: porte-monnaie)
    # |            : OU
    # <[^>]+>      : Une balise type HTML (si on veut les capturer pour les filtrer ou garder)
    # |            : OU
    # [^\w\s]      : Tout caractère qui n'est ni un mot ni un espace (ponctuation)
    
    # Note: On suppose que les contractions (l', j') ont été traitées avant (ex: l' -> l' ) 
    # ou seront traitées par le regex comme "l" + "'" si elles ne sont pas collées.
    # Pour cet exercice, on utilise un regex qui capture les mots avec tirets et la ponctuation isolée.
    
    pattern = r'\w+(?:-\w+)*|<[^>]+>|[^\w\s]'
    tokens = re.findall(pattern, phrase)
    
    if not balise:
        # Filtre les tokens qui ressemblent strictement à des balises <...>
        tokens = [t for t in tokens if not (t.startswith('<') and t.endswith('>'))]
        
    return tokens

def tokeniser_document(texte, abreviations=[], option=None, balise=False):
    """
    Segmente un document en phrases, puis chaque phrase en mots.
    
    Retour:
        list[list[str]]: Liste de listes de tokens.
    """
    phrases = segmenter_phrases(texte, abreviations, option)
    doc_tokenise = []
    for phrase in phrases:
        mots = segmenter_mots(phrase, balise)
        doc_tokenise.append(mots)
    return doc_tokenise

def tokeniser_corpus(corpus, abreviations=[], option=None, balise=False):
    """
    Traite un dictionnaire de documents.
    
    Paramètres:
        corpus (dict): { id_doc : texte_doc }
        
    Retour:
        dict: { id_doc : [[tokens], [tokens]...] }
    """
    corpus_tokenise = {}
    for id_doc, texte in corpus.items():
        corpus_tokenise[id_doc] = tokeniser_document(texte, abreviations, option, balise)
    return corpus_tokenise

def aplatir_tokens(document_tokens):
    """
    Aplatit une liste de listes de tokens en une seule liste.
    
    Paramètres:
        document_tokens (list[list[str]])
        
    Retour:
        list[str]
    """
    if not document_tokens:
        return []
    if isinstance(document_tokens[0], str):
        return document_tokens
    return [token for phrase in document_tokens for token in phrase]

def tokens_hapax(document):
    """
    Identifie les hapax (tokens apparaissant une seule fois).
    
    Paramètres:
        document : Peut être une liste de listes (résultat de tokeniser_document)
                   ou une liste simple (résultat de aplatir_tokens).
    
    Retour:
        list[str]: Liste des tokens uniques (hapax).
    """
    # Si c'est une liste de listes (le premier élément est une liste), on aplatit
    if document and isinstance(document[0], list):
        tokens_plats = aplatir_tokens(document)
    else:
        tokens_plats = document
        
    compteur = Counter(tokens_plats)
    # On garde ceux qui ont un count de 1
    hapax = [mot for mot, count in compteur.items() if count == 1]
    return hapax


"""
### 📝 Énoncé des fonctions - Partie 3 : N-grammes

Cette section concerne la génération de n-grammes (paires de mots, trigrammes, etc.)
en tenant compte des niveaux (phrase, document, corpus).
"""

def generer_ngrammes(donnees, n, niveau='phrase', par_phrase=True):
    """
    Génère des n-grammes à partir des données fournies.
    
    Paramètres:
        donnees: 
            - Si niveau='phrase': liste de tokens [t1, t2...]
            - Si niveau='document': liste de listes de tokens [[t1, t2], [t3...]]
            - Si niveau='corpus': dictionnaire {id_doc: liste de listes de tokens}
        n (int): Taille du n-gramme (1=unigramme, 2=bigramme, etc.)
        niveau (str): 'phrase', 'document', ou 'corpus'.
        par_phrase (bool): 
            - Si True, ne mélange pas les phrases (les n-grammes s'arrêtent à la fin de la phrase).
            - Si False, concatène les phrases avant de générer les n-grammes (le contexte traverse la phrase).
            
    Retour:
        - Liste de tuples (pour niveau phrase ou document).
        - Dictionnaire {id_doc: liste de tuples} (pour niveau corpus).
    """
    
    def _creer_ngrammes_liste(liste_tokens, n_val):
        if n_val < 1:
            return []
        if len(liste_tokens) < n_val:
            return []
        # Utilisation d'une compréhension de liste pour créer les tranches
        # Si n=1, cela crée des tuples de 1 élément: ('mot',)
        return [tuple(liste_tokens[i:i+n_val]) for i in range(len(liste_tokens) - n_val + 1)]

    # --- Niveau Phrase ---
    if niveau == 'phrase':
        # donnees est supposé être une liste simple de tokens
        return _creer_ngrammes_liste(donnees, n)

    # --- Niveau Document ---
    elif niveau == 'document':
        # donnees est une liste de listes [[phrase1], [phrase2]]
        resultat = []
        if par_phrase:
            # On traite chaque phrase indépendamment
            for phrase in donnees:
                ngrammes_phrase = _creer_ngrammes_liste(phrase, n)
                resultat.extend(ngrammes_phrase)
        else:
            # On concatène tout le document en une seule liste plate
            # ce qui permet de créer des n-grammes à cheval sur deux phrases
            tokens_plats = aplatir_tokens(donnees)
            resultat = _creer_ngrammes_liste(tokens_plats, n)
        return resultat

    # --- Niveau Corpus ---
    elif niveau == 'corpus':
        # donnees est un dict {id: document}
        resultat_corpus = {}
        for id_doc, doc_contenu in donnees.items():
            # On réutilise la logique 'document' pour chaque entrée
            resultat_corpus[id_doc] = generer_ngrammes(
                doc_contenu, n, niveau='document', par_phrase=par_phrase
            )
        return resultat_corpus
    
    return []

"""
### 📝 Énoncé des fonctions - Partie 4 : Statistiques sur le corpus avant filtrage

Cette section concerne les comptages globaux permettant de mesurer la densité lexicale
et la structure brute du corpus avant nettoyage approfondi.
"""

def compter_tokens_phrase(phrase):
    """
    Calcule le nombre de tokens présents dans une phrase.
    
    Paramètres:
        phrase (list[str]): Liste de tokens.
        
    Retour:
        int: Nombre de tokens.
    """
    return len(phrase)

def compter_tokens_document(document):
    """
    Compte le nombre total de tokens dans un document (toutes phrases confondues).
    
    Paramètres:
        document (list[list[str]]): Document tokenisé en phrases.
        
    Retour:
        int: Nombre total de tokens.
    """
    return sum(compter_tokens_phrase(phrase) for phrase in document)

def compter_tokens_corpus(corpus):
    """
    Agrège les résultats pour l’ensemble du corpus.
    
    Paramètres:
        corpus (dict): {id_doc : document_tokenisé}
        
    Retour:
        int: Nombre total de tokens dans le corpus.
    """
    return sum(compter_tokens_document(doc) for doc in corpus.values())

def calculer_mots_vides_document(document, stopwords):
    """
    Compte le nombre de mots vides (stopwords) dans un document.
    
    Paramètres:
        document (list[list[str]]): Document tokenisé.
        stopwords (list[str]): Liste des mots vides.
        
    Retour:
        int: Nombre de mots vides trouvés.
    """
    stopwords_set = set(mot.lower() for mot in stopwords) # Optimisation pour recherche rapide
    count = 0
    tokens_plats = aplatir_tokens(document)
    for token in tokens_plats:
        if token.lower() in stopwords_set:
            count += 1
    return count

def calculer_tokens_vides_document(document):
    """
    Identifie et compte le nombre de tokens vides ou non alphabétiques.
    Exemples non alphabétiques: "!", "123", "", etc.
    
    Paramètres:
        document (list[list[str]]): Document tokenisé.
        
    Retour:
        int: Nombre de tokens considérés comme "bruit" non alphabétique.
    """
    count = 0
    tokens_plats = aplatir_tokens(document)
    for token in tokens_plats:
        # isalpha() renvoie False pour "", "123", "!", "a1", etc.
        if not token.isalpha():
            count += 1
    return count

def calculer_mots_vides_corpus(corpus, stopwords):
    """
    Compte le nombre de mots vides dans l'ensemble du corpus.
    """
    return sum(calculer_mots_vides_document(doc, stopwords) for doc in corpus.values())

def calculer_tokens_vides_corpus(corpus):
    """
    Compte le nombre de tokens vides ou non alphabétiques dans le corpus.
    """
    return sum(calculer_tokens_vides_document(doc) for doc in corpus.values())

def statistiques_globales_corpus(corpus, stopwords):
    """
    Génère un dictionnaire récapitulatif des statistiques.
    """
    nb_tokens_total = compter_tokens_corpus(corpus)
    nb_mots_vides = calculer_mots_vides_corpus(corpus, stopwords)
    nb_tokens_vides = calculer_tokens_vides_corpus(corpus)
    
    proportion_bruit = 0
    if nb_tokens_total > 0:
        proportion_bruit = (nb_mots_vides + nb_tokens_vides) / nb_tokens_total
        
    stats_corpus = {
        "nb_documents": len(corpus),
        "nb_tokens_total": nb_tokens_total,
        "nb_mots_vides": nb_mots_vides,
        "nb_tokens_vides": nb_tokens_vides,
        "proportion_bruit": proportion_bruit
    }
    return stats_corpus

"""
### 📝 Énoncé des fonctions - Partie 4b : Distribution des tokens

Cette section permet d'analyser la longueur des documents, des phrases et des mots.
"""

def distribution_longueur_documents(corpus):
    """
    Calcule la longueur (en nombre de tokens) de chaque document.
    
    Paramètres:
        corpus (dict): {id_doc : document_tokenisé}
        
    Retour:
        dict: {id_doc : longueur_document}
    """
    distribution = {}
    for id_doc, doc in corpus.items():
        distribution[id_doc] = compter_tokens_document(doc)
    return distribution

def distribution_longueur_phrases(corpus):
    """
    Calcule la longueur moyenne des phrases (en nombre de tokens) pour chaque document.
    
    Paramètres:
        corpus (dict): {id_doc : document_tokenisé}
        
    Retour:
        dict: {id_doc : longueur_moyenne_phrase}
    """
    distribution = {}
    for id_doc, doc in corpus.items():
        nb_phrases = len(doc)
        if nb_phrases > 0:
            nb_tokens = compter_tokens_document(doc)
            moyenne = nb_tokens / nb_phrases
            distribution[id_doc] = moyenne
        else:
            distribution[id_doc] = 0.0
    return distribution

def distribution_longueur_mots(corpus):
    """
    Calcule la distribution de la longueur des mots (en caractères) sur tout le corpus.
    
    Paramètres:
        corpus (dict): {id_doc : document_tokenisé}
        
    Retour:
        dict: {longueur : fréquence}
    """
    toutes_les_longueurs = []
    
    # Parcourir tous les documents
    for doc in corpus.values():
        tokens = aplatir_tokens(doc)
        # Récupérer la longueur de chaque token
        longueurs = [len(token) for token in tokens]
        toutes_les_longueurs.extend(longueurs)
    
    # Compter les fréquences
    compteur = Counter(toutes_les_longueurs)
    
    # Retourner sous forme de dictionnaire simple
    return dict(compteur)


def distribution_occurrences_tokens(corpus):
    """
    Compte le nombre d’occurrences de chaque token dans le corpus.
    
    Paramètres:
        corpus (dict): {id_doc : document_tokenisé}
        
    Retour:
        dict: {token : nb_occ}
    """
    tous_tokens = []
    for doc in corpus.values():
        tous_tokens.extend(aplatir_tokens(doc))
    return dict(Counter(tous_tokens))

def tokens_plus_frequents(corpus, n=20):
    """
    Extrait les n tokens les plus fréquents du corpus (hors ponctuation et tokens vides).
    
    Paramètres:
        corpus (dict): {id_doc : document_tokenisé}
        n (int): Nombre de tokens à retourner.
        
    Retour:
        list[tuple]: Liste de tuples (token, nb_occ) triée par fréquence décroissante.
    """
    # 1. Obtenir toutes les occurrences
    dist = distribution_occurrences_tokens(corpus)
    
    # 2. Filtrer (garder uniquement les tokens alphanumériques)
    # On exclut la ponctuation et les tokens vides en vérifiant isalnum()
    filtered_items = [
        (tok, count) for tok, count in dist.items() 
        if tok.strip() and tok.isalnum()
    ]
    
    # 3. Trier et prendre les n premiers
    # Tri par count décroissant
    sorted_items = sorted(filtered_items, key=lambda x: x[1], reverse=True)
    
    return sorted_items[:n]

"""
### 📝 Énoncé des fonctions - Partie 4c : Indicateurs de tendance centrale et visualisation

Cette section permet de calculer des statistiques globales sur la structure du corpus
(moyennes, écart-types) et de visualiser ces données.
"""

def statistiques_corpus(corpus):
    """
    Calcule les indicateurs de tendance centrale et de dispersion.
    
    Paramètres:
        corpus (dict): {id_doc : document_tokenisé}
        
    Retour:
        dict: Dictionnaire contenant les statistiques (moyenne, écart-type, min, max, etc.)
    """
    # 1. Longueurs des documents (en tokens)
    longueurs_docs = [compter_tokens_document(doc) for doc in corpus.values()]
    
    # 2. Nombre de phrases par document
    phrases_par_doc = [len(doc) for doc in corpus.values()]
    
    # 3. Nombre de tokens par phrase (sur l'ensemble du corpus)
    toutes_phrases = [phrase for doc in corpus.values() for phrase in doc]
    tokens_par_phrase = [len(phrase) for phrase in toutes_phrases]
    
    # Gestion du corpus vide pour éviter les erreurs de calcul
    if not longueurs_docs:
        return {
            "longueur_moyenne_doc": 0, "ecart_type_doc": 0,
            "longueur_min_doc": 0, "longueur_max_doc": 0,
            "moyenne_phrases_par_doc": 0, "moyenne_tokens_par_phrase": 0
        }

    stats = {
        "longueur_moyenne_doc": np.mean(longueurs_docs),
        "ecart_type_doc": np.std(longueurs_docs),
        "longueur_min_doc": np.min(longueurs_docs),
        "longueur_max_doc": np.max(longueurs_docs),
        "moyenne_phrases_par_doc": np.mean(phrases_par_doc),
        "moyenne_tokens_par_phrase": np.mean(tokens_par_phrase) if tokens_par_phrase else 0
    }
    
    return stats

def tableau_de_bord(corpus, stopwords=None):
    """
    Génère un tableau de bord graphique (matplotlib) des statistiques du corpus.
    
    Paramètres:
        corpus (dict): Le corpus tokenisé.
        stopwords (list): Liste optionnelle pour le pie chart 'Bruit vs Contenu'.
    """
    if not corpus:
        print("Corpus vide, impossible de générer le tableau de bord.")
        return

    # Préparation des données
    longueurs_docs = list(distribution_longueur_documents(corpus).values())
    longueurs_mots_dist = distribution_longueur_mots(corpus)
    top_tokens = tokens_plus_frequents(corpus, n=10)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Tableau de bord statistique du Corpus')

    # 1. Histogramme : Distribution des longueurs de documents
    axs[0, 0].hist(longueurs_docs, bins=10, color='skyblue', edgecolor='black')
    axs[0, 0].set_title('Distribution de la longueur des documents (en tokens)')
    axs[0, 0].set_xlabel('Nombre de tokens')
    axs[0, 0].set_ylabel('Nombre de documents')

    # 2. Barplot : Top 10 tokens les plus fréquents
    if top_tokens:
        tokens, counts = zip(*top_tokens)
        axs[0, 1].bar(tokens, counts, color='lightgreen', edgecolor='black')
        axs[0, 1].set_title('Top 10 Tokens les plus fréquents')
        axs[0, 1].tick_params(axis='x', rotation=45)
    else:
        axs[0, 1].text(0.5, 0.5, "Pas assez de tokens", ha='center')

    # 3. Pie Chart : Proportion Bruit vs Contenu Utile
    if stopwords:
        nb_total = compter_tokens_corpus(corpus)
        nb_mots_vides = calculer_mots_vides_corpus(corpus, stopwords)
        nb_tokens_vides = calculer_tokens_vides_corpus(corpus)
        nb_bruit = nb_mots_vides + nb_tokens_vides
        nb_utile = nb_total - nb_bruit
        
        labels = ['Utile', 'Bruit (Stopwords/Non-Alpha)']
        sizes = [nb_utile, nb_bruit]
        colors = ['#ff9999', '#66b3ff']
        axs[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axs[1, 0].set_title('Proportion Contenu vs Bruit')
    else:
        axs[1, 0].text(0.5, 0.5, "Stopwords non fournis", ha='center')
        axs[1, 0].axis('off')

    # 4. Courbe/Barre : Distribution longueur des mots
    # Trie par longueur de mot (x)
    lengths_sorted = sorted(longueurs_mots_dist.keys())
    freqs_sorted = [longueurs_mots_dist[l] for l in lengths_sorted]
    axs[1, 1].plot(lengths_sorted, freqs_sorted, marker='o', linestyle='-', color='purple')
    axs[1, 1].set_title('Fréquence des longueurs de mots (caractères)')
    axs[1, 1].set_xlabel('Longueur du mot')
    axs[1, 1].set_ylabel('Fréquence')
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show() 
    print("Tableau de bord généré avec succès (voir plot).")


"""
### 📝 Énoncé des fonctions - Partie 5 : Construction du vocabulaire

Cette section permet de recenser la totalité des unités lexicales et n-grammes
présents dans le corpus avant filtrage.
"""

def construire_vocabulaire(corpus):
    """
    Retourne la liste des tokens distincts (uniques) présents dans le corpus.
    
    Paramètres:
        corpus (dict): {id_doc: document_tokenisé}
    
    Retour:
        list[str]: Liste des tokens uniques.
    """
    tous_tokens = []
    for doc in corpus.values():
        tous_tokens.extend(aplatir_tokens(doc))
    # Utilisation de set pour l'unicité, puis conversion en liste
    return list(set(tous_tokens))

def vocabulaire_ngrammes(corpus, n):
    """
    Retourne la liste des n-grammes uniques présents dans le corpus.
    
    Paramètres:
        corpus (dict): {id_doc: document_tokenisé}
        n (int): Taille du n-gramme.
        
    Retour:
        list[tuple]: Liste des n-grammes uniques.
    """
    # generer_ngrammes avec niveau='corpus' retourne {id_doc: [liste_ngrammes]}
    dict_ngrammes = generer_ngrammes(corpus, n, niveau='corpus', par_phrase=True)
    
    tous_ngrammes = []
    for grams in dict_ngrammes.values():
        tous_ngrammes.extend(grams)
        
    return list(set(tous_ngrammes))

def construire_dictionnaire_vocabulaire(vocabulaire):
    """
    Construit les dictionnaires d'indexation (mot -> index et index -> mot).
    
    Paramètres:
        vocabulaire (list): Liste de tokens uniques (ou dict de fréquences).
        
    Retour:
        tuple: (dictionnaire direct {mot: index}, dictionnaire inverse {index: mot})
    """
    # Si on reçoit un dictionnaire de fréquences, on prend les clés
    if isinstance(vocabulaire, dict):
        mots_uniques = sorted(list(vocabulaire.keys()))
    else:
        mots_uniques = sorted(list(vocabulaire))
    
    word2id = {mot: i for i, mot in enumerate(mots_uniques)}
    id2word = {i: mot for i, mot in enumerate(mots_uniques)}
    
    return word2id, id2word

"""
### 📝 Énoncé des fonctions - Partie 6 : Diversité lexicale

Cette section permet de calculer des indicateurs de richesse, de rareté (hapax)
et de dispersion pour évaluer la diversité du vocabulaire.
"""

def calculer_richesse_lexicale(vocabulaire):
    """
    Calcule la diversité du vocabulaire (Type-Token Ratio - TTR).
    
    Formule : Richesse = Nombre de mots uniques / Nombre de tokens total
    
    Paramètres:
        vocabulaire (dict): Dictionnaire de fréquences {token: nb_occ}, 
                            tel que retourné par distribution_occurrences_tokens.
                            
    Retour:
        float: Indice de richesse lexicale.
    """
    if not vocabulaire:
        return 0.0
        
    nb_uniques = len(vocabulaire)
    nb_tokens_total = sum(vocabulaire.values())
    
    if nb_tokens_total == 0:
        return 0.0
        
    return nb_uniques / nb_tokens_total

def calculer_taux_hapax(vocabulaire):
    """
    Calcule la proportion de mots apparaissant une seule fois (Hapax).
    
    Formule : Taux Hapax = Nombre d'hapax / Nombre de mots uniques
    
    Paramètres:
        vocabulaire (dict): Dictionnaire de fréquences {token: nb_occ}.
    
    Retour:
        float: Taux d'hapax.
    """
    if not vocabulaire:
        return 0.0
        
    # On compte combien de mots ont une occurrence de 1
    nb_hapax = sum(1 for count in vocabulaire.values() if count == 1)
    nb_uniques = len(vocabulaire)
    
    if nb_uniques == 0:
        return 0.0
        
    return nb_hapax / nb_uniques

def calculer_dispersion_lexicale(vocabulaire):
    """
    Mesure la variabilité des occurrences des tokens dans le vocabulaire.
    
    Paramètres:
        vocabulaire (dict): Dictionnaire de fréquences {token: nb_occ}.
    
    Retour:
        float: Écart-type des occurrences (dispersion).
    """
    if not vocabulaire:
        return 0.0
        
    occurrences = list(vocabulaire.values())
    return float(np.std(occurrences))

def indicateurs_lexicaux(corpus):
    """
    Fonction utilitaire pour calculer et regrouper tous les indicateurs.
    """
    # 1. On génère le "vocabulaire" au sens fréquences
    vocab_freq = distribution_occurrences_tokens(corpus)
    
    stats_vocabulaire = {
        "richesse_lexicale": calculer_richesse_lexicale(vocab_freq),
        "taux_hapax": calculer_taux_hapax(vocab_freq),
        "dispersion_lexicale": calculer_dispersion_lexicale(vocab_freq)
    }
    return stats_vocabulaire

"""
### 📝 Énoncé des fonctions - Partie 7 : Filtrage lexical (Stopwords)

Cette section concerne la suppression des mots vides (stopwords) pour réduire
le bruit lexical.
"""

def construire_liste_stopwords(langue):
    """
    Construit une liste de stopwords de base selon la langue.
    
    Paramètres:
        langue (str): 'fr' ou 'en'.
        
    Retour:
        list[str]: Liste de stopwords.
    """
    if langue == 'fr':
        return [
            "le", "la", "les", "un", "une", "des", "du", "de", "d'", "l'",
            "et", "est", "il", "elle", "ils", "elles", "je", "tu", "nous", "vous",
            "ce", "cet", "cette", "ces", "mon", "ton", "son", "sa", "ses",
            "que", "qui", "où", "a", "à", "dans", "pour", "par", "sur", "en", "au", "aux"
        ]
    elif langue == 'en':
        return [
            "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been",
            "in", "on", "at", "to", "for", "with", "by", "of", "from",
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
            "my", "your", "his", "its", "our", "their", "that", "this", "these", "those"
        ]
    else:
        return []

def supprimer_mots_vides(tokens, liste_stopwords):
    """
    Retire les mots vides d'une liste de tokens.
    
    Paramètres:
        tokens (list[str]): Liste de tokens à filtrer.
        liste_stopwords (list[str]): Liste des mots à exclure.
        
    Retour:
        list[str]: Liste de tokens filtrée.
    """
    # Création d'un set pour une recherche O(1) et insensible à la casse
    stopwords_set = set(mot.lower() for mot in liste_stopwords)
    
    tokens_filtres = []
    for token in tokens:
        if token.lower() not in stopwords_set:
            tokens_filtres.append(token)
            
    return tokens_filtres

def supprimer_tokens_courts(tokens, longueur_min=3):
    """
    Retire les tokens dont la longueur est inférieure au seuil spécifié.
    
    Paramètres:
        tokens (list[str]): Liste de tokens.
        longueur_min (int): Longueur minimale requise pour garder le token.
        
    Retour:
        list[str]: Liste filtrée.
    """
    return [t for t in tokens if len(t) >= longueur_min]

def supprimer_non_alphabetiques(tokens):
    """
    Retire les tokens qui ne sont pas purement alphabétiques.
    Exemple: "123", "a1", "!" sont supprimés.
    
    Paramètres:
        tokens (list[str]): Liste de tokens.
        
    Retour:
        list[str]: Liste filtrée (uniquement alphabétique).
    """
    return [t for t in tokens if t.isalpha()]


def filtrer_par_occurrence(vocabulaire, occ_min=2, occ_max=None):
    """
    Conserve uniquement les tokens dont le nombre d'occurrences est compris entre les seuils.
    
    Paramètres:
        vocabulaire (dict): Dictionnaire de fréquences {token: count}.
        occ_min (int): Nombre minimum d'occurrences.
        occ_max (int or None): Nombre maximum d'occurrences (optionnel).
        
    Retour:
        dict: Dictionnaire de fréquences filtré.
    """
    vocab_filtre = {}
    for token, count in vocabulaire.items():
        if count >= occ_min:
            if occ_max is None or count <= occ_max:
                vocab_filtre[token] = count
    return vocab_filtre

def pipeline_filtrage(tokens, config, langue="fr"):
    """
    Applique successivement les opérations de filtrage définies dans la configuration.
    
    Paramètres:
        tokens (list[str]): Liste brute de tokens.
        config (dict): Dictionnaire de configuration (ex: stopwords=True, longueur_min=3...)
        langue (str): Langue pour les stopwords ('fr' ou 'en').
        
    Retour:
        list[str]: Liste finale de tokens filtrés.
    """
    resultat = tokens
    
    # 1. Non alphabétique (souvent le premier filtre pour virer le bruit évident)
    if config.get("non_alphabetiques", False):
        resultat = supprimer_non_alphabetiques(resultat)
        
    # 2. Stopwords
    if config.get("stopwords", False):
        stopwords = construire_liste_stopwords(langue)
        resultat = supprimer_mots_vides(resultat, stopwords)
        
    # 3. Longueur min
    longueur_min = config.get("longueur_min", 0)
    if longueur_min > 0:
        resultat = supprimer_tokens_courts(resultat, longueur_min)
        
    # 4. Occurrences (Min/Max)
    # Nécessite de compter sur l'ensemble restant
    occ_min = config.get("occ_min", 1)
    occ_max = config.get("occ_max", None)
    
    # Si on a des contraintes d'occurrence significatives
    if occ_min > 1 or occ_max is not None:
        # On compte les fréquences actuelles
        compteur = Counter(resultat)
        # On filtre le dictionnaire de fréquences
        vocab_valide = filtrer_par_occurrence(dict(compteur), occ_min, occ_max)
        # On ne garde que les tokens présents dans le vocabulaire valide
        set_valide = set(vocab_valide.keys())
        resultat = [t for t in resultat if t in set_valide]
        
    return resultat

"""
### 📝 Énoncé des fonctions - Partie 8 : Normalisation morphologique (Stemming & Lemmatisation)

Cette section concerne la réduction des mots à leur racine (stem) ou forme canonique (lemme).
"""

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
def pipeline_morphologique(tokens, config, langue="fr", lemmes=None):
    """
    Applique la normalisation morphologique selon la configuration.
    
    Paramètres:
        tokens (list[str]): Liste de tokens.
        config (dict): Configuration { "stemming": bool, "lemmatisation": bool }.
        langue (str): 'fr' ou 'en'.
        lemmes (dict): Dictionnaire de lemmes optionnel.
        
    Retour:
        list[str]: Liste de tokens normalisés.
    """
    resultat = tokens
    
    # Ordre : Lemmatisation (plus précis) puis Stemming (plus agressif) si les deux sont activés
    if config.get("lemmatisation", False):
        resultat = appliquer_lemmatisation(resultat, langue, lemmes)
        
    if config.get("stemming", False):
        resultat = appliquer_stemming(resultat, langue)
        
    return resultat

"""
### 📝 Énoncé des fonctions - Partie 9 : Statistiques et Visualisation après traitement

Cette section permet d'évaluer l'impact du filtrage et de la normalisation sur le corpus.
"""

def calculer_statistiques_post_traitement(corpus_filtre, vocabulaire_filtre, stopwords=None):
    """
    Calcule les statistiques clés après traitement (filtrage/normalisation).
    
    Paramètres:
        corpus_filtre (dict): Le corpus traité {id_doc: liste_tokens}.
        vocabulaire_filtre (dict): Dictionnaire de fréquences {token: count}.
        stopwords (list): Optionnel, pour recalculer le bruit résiduel.
        
    Retour:
        dict: Dictionnaire de statistiques.
    """
    nb_documents = len(corpus_filtre)
    
    # Somme des valeurs du dictionnaire de fréquences pour le total tokens
    nb_tokens_total = sum(vocabulaire_filtre.values())
    
    taille_vocabulaire = len(vocabulaire_filtre)
    
    richesse = calculer_richesse_lexicale(vocabulaire_filtre)
    taux_hapax = calculer_taux_hapax(vocabulaire_filtre)
    
    # Estimation du bruit résiduel (si stopwords fournis ou non-alpha)
    bruit_count = 0
    if stopwords:
        stopwords_set = set(stopwords)
        for token, count in vocabulaire_filtre.items():
            if token.lower() in stopwords_set or not token.isalpha():
                bruit_count += count
    else:
        # Si pas de stopwords fournis, on compte au moins les non-alpha
        for token, count in vocabulaire_filtre.items():
            if not token.isalpha():
                bruit_count += count
                
    proportion_bruit = bruit_count / nb_tokens_total if nb_tokens_total > 0 else 0
    
    stats_post = {
        "nb_documents": nb_documents,
        "nb_tokens_total": nb_tokens_total,
        "taille_vocabulaire": taille_vocabulaire,
        "richesse_lexicale": richesse,
        "taux_hapax": taux_hapax,
        "proportion_bruit_residuel": proportion_bruit
    }
    return stats_post

def visualiser_statistiques_post_traitement(stats_initiales, stats_post):
    """
    Affiche des graphiques comparatifs avant/après traitement.
    
    Paramètres:
        stats_initiales (dict): Stats retournées par statistiques_globales_corpus
                                + info vocabulaire (à construire si manquant).
        stats_post (dict): Stats retournées par calculer_statistiques_post_traitement.
    """
    # Préparation des données pour l'affichage
    labels = ['Avant', 'Après']
    
    # 1. Taille Vocabulaire & Tokens Total (nécessite double axe ou subplot)
    # Note: stats_initiales ne contient pas toujours "taille_vocabulaire" selon l'implémentation précédente.
    # On gère le cas où la clé manque (0 par défaut)
    vocab_init = stats_initiales.get("taille_vocabulaire", 0) 
    # Si 0, c'est peut-être que stats_initiales vient de 'statistiques_globales_corpus' qui ne calcule pas le vocab.
    # Dans ce cas, on ne peut pas comparer le vocabulaire proprement sans le recalculer.
    # Pour l'exercice, on suppose que les données sont là ou on affiche ce qu'on peut.
    
    vocab_post = stats_post["taille_vocabulaire"]
    
    tokens_init = stats_initiales["nb_tokens_total"]
    tokens_post = stats_post["nb_tokens_total"]
    
    # 2. Richesse & Hapax
    # Idem, si stats_initiales n'a pas richesse, on met 0
    rich_init = stats_initiales.get("richesse_lexicale", 0)
    rich_post = stats_post["richesse_lexicale"]
    
    hapax_init = stats_initiales.get("taux_hapax", 0)
    hapax_post = stats_post["taux_hapax"]
    
    # Création de la figure
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Comparaison Avant / Après Traitement')
    
    # Graphique 1 : Volume (Tokens & Vocabulaire)
    x = np.arange(len(labels))
    width = 0.35
    
    # On normalise ou on affiche sur deux échelles différentes ?
    # Ici on affiche simplement les tokens car le vocab est souvent beaucoup plus petit
    axs[0].bar(x - width/2, [tokens_init, tokens_post], width, label='Total Tokens', color='skyblue')
    # Pour le vocabulaire, on utilise un axe jumeau si on veut, ou on le met à côté si les ordres de grandeur sont proches.
    # Souvent Tokens >> Vocab. On va faire un twinx pour la lisibilité.
    ax0_twin = axs[0].twinx()
    ax0_twin.bar(x + width/2, [vocab_init, vocab_post], width, label='Taille Vocab', color='orange')
    
    axs[0].set_ylabel('Nombre de Tokens')
    ax0_twin.set_ylabel('Taille Vocabulaire')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels)
    axs[0].set_title('Réduction du Volume')
    # Légendes combinées
    lines1, labels1 = axs[0].get_legend_handles_labels()
    lines2, labels2 = ax0_twin.get_legend_handles_labels()
    ax0_twin.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    # Graphique 2 : Qualité Lexicale (Richesse & Hapax)
    width = 0.35
    axs[1].bar(x - width/2, [rich_init, rich_post], width, label='Richesse Lexicale', color='#90ee90')
    axs[1].bar(x + width/2, [hapax_init, hapax_post], width, label='Taux Hapax', color='#ffcccb')
    axs[1].set_ylabel('Ratio (0-1)')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels)
    axs[1].set_title('Indicateurs Lexicaux')
    axs[1].legend()
    axs[1].set_ylim(0, 1.1) # Ratios
    
    # Graphique 3 : Bruit Résiduel
    # Comparaison proportion bruit
    bruit_init = stats_initiales.get("proportion_bruit", 0)
    bruit_post = stats_post["proportion_bruit_residuel"]
    
    axs[2].bar(labels, [bruit_init, bruit_post], color=['#d3d3d3', '#66cdaa'])
    axs[2].set_ylabel('Proportion de Bruit')
    axs[2].set_title('Réduction du Bruit')
    axs[2].set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.show()
    print("Graphiques comparatifs générés.")

"""
### 📝 Énoncé des fonctions - Partie 10 : Pipeline Complet et Analyse Comparative

Cette section orchestre l'ensemble du prétraitement et permet de comparer plusieurs
configurations (A, B, C, D, E) pour choisir la meilleure stratégie.
"""

def pipeline_pretraitement(corpus, config, langue="fr", stopwords=None, lemmes=None):
    """
    Applique le pipeline complet (filtrage + morphologie) sur un corpus.
    
    Paramètres:
        corpus (dict): {id_doc: liste_tokens} ou {id_doc: [[phrases]]}
        config (dict): Paramètres complets (stopwords, longueur_min, stemming, lemmatisation...)
        langue (str): 'fr' ou 'en'.
        stopwords (list): Liste personnalisée (si None, utilise défaut).
        lemmes (dict): Dictionnaire de lemmes (si None, utilise défaut).
    
    Retour:
        dict: Nouveau corpus traité {id_doc: liste_tokens_traités}.
    """
    corpus_traite = {}
    
    for id_doc, contenu in corpus.items():
        # 1. Aplatir si nécessaire (si c'est une liste de phrases)
        if contenu and isinstance(contenu[0], list):
            tokens = aplatir_tokens(contenu)
        else:
            tokens = contenu
            
        # 2. Pipeline de filtrage
        # Note: on passe 'config' qui contient les clés de filtrage (stopwords, longueur...)
        tokens_filtres = pipeline_filtrage(tokens, config, langue)
        
        # 3. Pipeline morphologique
        # Note: on passe 'config' qui contient les clés morpho (stemming, lemmatisation)
        tokens_final = pipeline_morphologique(tokens_filtres, config, langue, lemmes)
        
        corpus_traite[id_doc] = tokens_final
        
    return corpus_traite

def analyser_configurations(corpus, configurations=None, langue="fr"):
    """
    Compare plusieurs configurations de prétraitement sur le même corpus.
    
    Paramètres:
        corpus (dict): Corpus initial.
        configurations (dict): Dictionnaire de configs { "Nom": {param: val...} }.
                               Si None, utilise les configs par défaut (A, B, C, D, E).
        langue (str): Langue du corpus.
        
    Retour:
        dict: Synthèse des résultats { "NomConfig": {stats...} }.
    """
    if configurations is None:
        # Définition des 5 configurations standards
        # A: Brut
        # B: Filtrage (Stopwords, Len>3, Alpha)
        # C: B + Stemming
        # D: B + Lemmatisation
        # E: B + Stemming + Lemmatisation
        configurations = {
            "A (Brut)": {
                "stopwords": False, "longueur_min": 0, "non_alphabetiques": False,
                "stemming": False, "lemmatisation": False
            },
            "B (Filtré)": {
                "stopwords": True, "longueur_min": 3, "non_alphabetiques": True,
                "stemming": False, "lemmatisation": False
            },
            "C (Filtré + Stem)": {
                "stopwords": True, "longueur_min": 3, "non_alphabetiques": True,
                "stemming": True, "lemmatisation": False
            },
            "D (Filtré + Lem)": {
                "stopwords": True, "longueur_min": 3, "non_alphabetiques": True,
                "stemming": False, "lemmatisation": True
            },
            "E (Tout)": {
                "stopwords": True, "longueur_min": 3, "non_alphabetiques": True,
                "stemming": True, "lemmatisation": True
            }
        }
        
    resultats = {}
    
    print(f"\n--- Analyse Comparative ({len(configurations)} configs) ---")
    
    for nom_config, params in configurations.items():
        # Exécution du pipeline
        corpus_res = pipeline_pretraitement(corpus, params, langue)
        
        # Calcul du vocabulaire
        vocab_freq = distribution_occurrences_tokens(corpus_res)
        
        # Calcul des stats
        stats = calculer_statistiques_post_traitement(corpus_res, vocab_freq)
        
        # Stockage
        resultats[nom_config] = stats
        
        # Affichage rapide
        print(f"[{nom_config}] Vocab: {stats['taille_vocabulaire']}, "
              f"Tokens: {stats['nb_tokens_total']}, "
              f"Richesse: {stats['richesse_lexicale']:.3f}, "
              f"Hapax: {stats['taux_hapax']:.3f}")
              
    return resultats


"""
### 📝 Énoncé des fonctions - Partie 11 : Visualisation comparative des configurations

Cette fonction permet de visualiser graphiquement les différences entre les
configurations testées (vocabulaire, richesse, bruit).
"""

def visualiser_comparaison_configurations(resultats_configurations):
    """
    Produit des graphiques comparatifs entre plusieurs configurations.
    
    Graphiques :
    1. Barplot de la taille du vocabulaire.
    2. Barres groupées : Richesse lexicale et Taux d'hapax.
    3. Barplot de la proportion de bruit lexical.
    
    Paramètres:
        resultats_configurations (dict): { "NomConfig": {stats...} }
    """
    if not resultats_configurations:
        print("Aucun résultat à visualiser.")
        return

    # Extraction des données
    labels = list(resultats_configurations.keys())
    vocab_sizes = [res["taille_vocabulaire"] for res in resultats_configurations.values()]
    richesses = [res["richesse_lexicale"] for res in resultats_configurations.values()]
    hapaxes = [res["taux_hapax"] for res in resultats_configurations.values()]
    bruits = [res["proportion_bruit_residuel"] for res in resultats_configurations.values()]
    
    x = np.arange(len(labels))
    width = 0.35

    # Création de la figure (1 ligne, 3 colonnes)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Comparaison des Configurations de Prétraitement')

    # 1. Taille du Vocabulaire
    axs[0].bar(x, vocab_sizes, color='skyblue', edgecolor='black')
    axs[0].set_ylabel('Taille du Vocabulaire (mots uniques)')
    axs[0].set_title('Taille du Vocabulaire par Config')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels, rotation=45, ha='right')

    # 2. Richesse et Hapax (Barres groupées)
    axs[1].bar(x - width/2, richesses, width, label='Richesse Lexicale', color='#90ee90')
    axs[1].bar(x + width/2, hapaxes, width, label='Taux Hapax', color='#ffcccb')
    axs[1].set_ylabel('Ratio (0-1)')
    axs[1].set_title('Richesse Lexicale & Taux Hapax')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels, rotation=45, ha='right')
    axs[1].legend()
    axs[1].set_ylim(0, 1.1)

    # 3. Proportion de Bruit (Barplot simple ou Pie si 1 config, mais ici comparatif)
    # Un barplot est plus lisible pour comparer plusieurs configs qu'une série de camemberts
    axs[2].bar(x, bruits, color='#d3d3d3', edgecolor='black')
    axs[2].set_ylabel('Proportion de Bruit')
    axs[2].set_title('Proportion de Bruit Lexical Résiduel')
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(labels, rotation=45, ha='right')
    axs[2].set_ylim(0, max(bruits) * 1.1 if bruits and max(bruits) > 0 else 1.0)

    plt.tight_layout()
    plt.show()
    print("Graphiques comparatifs des configurations générés avec succès.")


"""
### 🔍 Explication des tests unitaires

Stratégie de test :

1.  **Fonctions précédentes** : On garde les tests existants.
2.  **Statistiques** : Vérification des comptages (tokens, vides, bruit).
3.  **Distribution** : Vérification des longueurs (docs, phrases, mots) et fréquences.
4.  **Tendances & Visualisation** : Test des moyennes/écarts-types et smoke test du tableau de bord.
5.  **Vocabulaire** : Vérification de l'unicité des tokens et n-grammes.
6.  **Diversité Lexicale** : Vérification Richesse, Hapax et Dispersion sur corpus contrôlé.

"""

def test_tokenisation_complete():
    print("\n--- Démarrage des tests complets ---")

    # --- Tests existants (rapides) ---
    texte_complexe = "Le 12.05.2025 est une date."
    assert len(segmenter_phrases(texte_complexe, option={"gerer_dates": True})) == 1
    
    doc_tok = tokeniser_document("Hello world.")
    assert len(doc_tok[0]) == 3 # Hello, world, .
    
    bigrammes = generer_ngrammes(["A", "B"], 2, niveau='phrase')
    assert bigrammes == [("A", "B")]
    
    print("✅ Tests Tokenisation & N-grammes passés.")

    # --- Tests Statistiques ---
    print("\n--- Démarrage des tests Statistiques ---")
    
    # Corpus de test : 
    # Doc 1: "Le chat dort !" (4 tokens : Le, chat, dort, !)
    # Doc 2: "123 test." (3 tokens : 123, test, .)
    # Total tokens = 7
    doc1 = [["Le", "chat", "dort", "!"]]
    doc2 = [["123", "test", "."]]
    corpus_test = {"d1": doc1, "d2": doc2}
    stopwords_test = ["le", "la"] # "Le" devrait être compté comme stopword
    
    # 1. Comptages simples
    assert compter_tokens_document(doc1) == 4
    assert compter_tokens_corpus(corpus_test) == 7
    print("✅ Test Comptage Tokens passé.")
    
    # 2. Mots vides (Stopwords)
    # Dans doc1, "Le" est un stopword (insensible à la casse)
    nb_stop_d1 = calculer_mots_vides_document(doc1, stopwords_test)
    assert nb_stop_d1 == 1
    assert calculer_mots_vides_corpus(corpus_test, stopwords_test) == 1
    print("✅ Test Mots Vides passé.")
    
    # 3. Tokens vides / non alphabétiques
    # Doc 1 : "!" est non alpha -> 1
    # Doc 2 : "123" et "." sont non alpha -> 2
    # Total corpus : 3
    nb_vide_d1 = calculer_tokens_vides_document(doc1)
    assert nb_vide_d1 == 1
    nb_vide_d2 = calculer_tokens_vides_document(doc2)
    assert nb_vide_d2 == 2
    assert calculer_tokens_vides_corpus(corpus_test) == 3
    print("✅ Test Tokens Non-Alphabétiques passé.")
    
    # 4. Stats globales
    stats = statistiques_globales_corpus(corpus_test, stopwords_test)
    assert stats["nb_documents"] == 2
    assert stats["nb_tokens_total"] == 7
    assert stats["nb_mots_vides"] == 1
    assert stats["nb_tokens_vides"] == 3
    # Bruit = (1 + 3) / 7 = 4/7
    assert abs(stats["proportion_bruit"] - (4/7)) < 0.001
    print("✅ Test Dictionnaire Stats Globales passé.")

    # --- Tests Distribution ---
    print("\n--- Démarrage des tests Distribution ---")

    # 5. Longueur Documents
    # d1 -> 4 tokens, d2 -> 3 tokens
    dist_doc = distribution_longueur_documents(corpus_test)
    assert dist_doc["d1"] == 4
    assert dist_doc["d2"] == 3
    print("✅ Test Distribution Longueur Documents passé.")

    # 6. Longueur Moyenne Phrases
    # d1 -> 1 phrase de 4 tokens -> moy 4.0
    # d2 -> 1 phrase de 3 tokens -> moy 3.0
    dist_phrase = distribution_longueur_phrases(corpus_test)
    assert dist_phrase["d1"] == 4.0
    assert dist_phrase["d2"] == 3.0
    print("✅ Test Distribution Longueur Phrases passé.")

    # 7. Longueur Mots
    # Tokens: "Le" (2), "chat" (4), "dort" (4), "!" (1), "123" (3), "test" (4), "." (1)
    # Longueurs: 2, 4, 4, 1, 3, 4, 1
    # Compte: 1->2, 2->1, 3->1, 4->3
    dist_mots = distribution_longueur_mots(corpus_test)
    assert dist_mots[1] == 2
    assert dist_mots[2] == 1
    assert dist_mots[3] == 1
    assert dist_mots[4] == 3
    print("✅ Test Distribution Longueur Mots passé.")

    # 8. Occurrences & Plus fréquents
    # Créons un corpus un peu plus riche pour les fréquences
    # "chat" apparait 3 fois, "chien" 2 fois, "souris" 1 fois, "!" 5 fois (mais c'est de la ponctuation)
    doc3 = [["chat", "chat", "chien", "!", "!"]]
    doc4 = [["chien", "chat", "souris", "!"]]
    corpus_freq = {"d3": doc3, "d4": doc4}
    
    # Test distribution
    occ = distribution_occurrences_tokens(corpus_freq)
    assert occ["chat"] == 3
    assert occ["chien"] == 2
    assert occ["!"] == 3
    print("✅ Test Distribution Occurrences passé.")
    
    # Test Plus Fréquents (hors ponctuation)
    # On demande top 2.
    # Attendu: [("chat", 3), ("chien", 2)]. "souris" (1) est moins fréquent. "!" est ignoré.
    top = tokens_plus_frequents(corpus_freq, n=2)
    assert len(top) == 2
    assert top[0] == ("chat", 3)
    assert top[1] == ("chien", 2)
    
    # Vérification que "!" n'est pas là
    top_all = tokens_plus_frequents(corpus_freq, n=10)
    tokens_top = [t[0] for t in top_all]
    assert "!" not in tokens_top
    print("✅ Test Tokens Plus Fréquents passé.")

    # --- Tests Statistiques Corpus & Visualisation ---
    print("\n--- Démarrage des tests Stats & Visu ---")
    
    # Corpus test simple : d1 (4 tokens), d2 (3 tokens) -> longueurs [4, 3]
    # Moyenne = 3.5
    # Ecart-type = 0.5
    stats_c = statistiques_corpus(corpus_test)
    assert stats_c["longueur_moyenne_doc"] == 3.5
    assert abs(stats_c["ecart_type_doc"] - 0.5) < 0.001
    assert stats_c["longueur_max_doc"] == 4
    
    print("✅ Test Statistiques Corpus passé.")
    
    # Test Tableau de bord (Smoke test : vérifier que ça ne plante pas)
    print("Génération du tableau de bord (simulation)...")
    tableau_de_bord(corpus_test, stopwords=stopwords_test)
    print("✅ Test Tableau de Bord exécuté (voir sortie console).")

    # --- Tests Vocabulaire ---
    print("\n--- Démarrage des tests Vocabulaire ---")
    doc_vocab = [["a", "b", "a", "c"]]
    corpus_vocab = {"d1": doc_vocab}
    
    # 1. Vocabulaire simple
    # doc contient a, b, a, c -> uniques : a, b, c
    vocab = construire_vocabulaire(corpus_vocab)
    assert len(vocab) == 3
    assert "a" in vocab
    assert "b" in vocab
    print("✅ Test Construction Vocabulaire passé.")
    
    # 2. Vocabulaire N-grammes (bigrammes)
    # a b a c -> (a,b), (b,a), (a,c) -> 3 bigrammes uniques
    vocab_bi = vocabulaire_ngrammes(corpus_vocab, 2)
    assert len(vocab_bi) == 3
    assert ("a", "b") in vocab_bi
    assert ("b", "a") in vocab_bi
    print("✅ Test Vocabulaire N-grammes passé.")

     # 2. Dictionnaire Index
    word2id, id2word = construire_dictionnaire_vocabulaire(vocab)
    assert len(word2id) == 3
    # Test cohérence
    idx_a = word2id["a"]
    assert id2word[idx_a] == "a"
    print("✅ Test Indexation passé.")

 
    # --- Tests Diversité Lexicale ---
    print("\n--- Démarrage des tests Diversité Lexicale ---")
    
    # Corpus "Richesse" : 4 tokens total, 3 uniques ("le", "chat", "dort")
    # "le": 2, "chat": 1, "dort": 1
    doc_rich = [["le", "chat", "dort", "le"]]
    corpus_rich = {"d1": doc_rich}
    
    # Génération du vocabulaire fréquentiel
    vocab_freq_test = distribution_occurrences_tokens(corpus_rich)
    # vocab_freq_test = {'le': 2, 'chat': 1, 'dort': 1}
    
    # 1. Richesse (3 uniques / 4 total = 0.75)
    richesse = calculer_richesse_lexicale(vocab_freq_test)
    assert richesse == 0.75
    print("✅ Test Richesse Lexicale passé.")
    
    # 2. Taux Hapax (2 hapax ("chat", "dort") / 3 uniques = 0.666...)
    hapax_rate = calculer_taux_hapax(vocab_freq_test)
    assert abs(hapax_rate - (2/3)) < 0.001
    print("✅ Test Taux Hapax passé.")
    
    # 3. Dispersion (Standard deviation de [2, 1, 1])
    # Moyenne = 1.333
    # Variances = (2-1.33)^2 + (1-1.33)^2 + (1-1.33)^2 ...
    dispersion = calculer_dispersion_lexicale(vocab_freq_test)
    attendue = np.std([2, 1, 1])
    assert abs(dispersion - attendue) < 0.001
    print("✅ Test Dispersion Lexicale passé.")

    # --- Tests Filtrage (Stopwords) ---
    print("\n--- Démarrage des tests Filtrage ---")
    
    # 1. Construction liste
    sw_fr = construire_liste_stopwords('fr')
    assert "le" in sw_fr
    assert "pour" in sw_fr
    sw_en = construire_liste_stopwords('en')
    assert "the" in sw_en
    assert "is" in sw_en
    print("✅ Test Liste Stopwords passé.")
    
    # 2. Suppression mots vides
    phrase_brute = ["Le", "chat", "est", "sur", "la", "table"]
    # Stopwords attendus à filtrer : Le (le), est, sur, la
    # Reste : chat, table
    tokens_propres = supprimer_mots_vides(phrase_brute, sw_fr)
    assert "chat" in tokens_propres
    assert "table" in tokens_propres
    assert "Le" not in tokens_propres # Vérifie l'insensibilité à la casse
    assert len(tokens_propres) == 2
    
    # 3. Filtrage par longueur
    tokens_courts = ["un", "de", "chat", "est", "là", "a"]
    # Longueur min 3 => chat, est. "un" (2), "de" (2), "là" (2), "a" (1) virés.
    res_long = supprimer_tokens_courts(tokens_courts, longueur_min=3)
    assert "chat" in res_long
    assert "un" not in res_long
    assert len(res_long) == 2
    print("✅ Test Filtrage Longueur passé.")

    # 4. Filtrage non alphabétiques
    tokens_bruit = ["chat", "123", "chat1", "!", "test"]
    # 123 (digit), chat1 (alnum mais pas alpha), ! (punct) virés.
    # Reste: chat, test
    res_alpha = supprimer_non_alphabetiques(tokens_bruit)
    assert "chat" in res_alpha
    assert "test" in res_alpha
    assert "123" not in res_alpha
    assert "chat1" not in res_alpha
    print("✅ Test Filtrage Non-Alphabétiques passé.")
     # 5. Filtrage par occurrence
    vocab_occ = {"rare": 1, "commun": 5, "trop_frequent": 100}
    # Garder occ_min=2, occ_max=50 => "commun" seulement
    res_occ = filtrer_par_occurrence(vocab_occ, occ_min=2, occ_max=50)
    assert "commun" in res_occ
    assert "rare" not in res_occ
    assert "trop_frequent" not in res_occ
    print("✅ Test Filtrage Occurrences passé.")
    
    # 6. Pipeline Complet
    tokens_pipeline = ["Le", "chat", "mange", "une", "souris", ".", "chat", "a", "123"]
    # Config : 
    # - non_alphabetiques=True -> virer ".", "123"
    # - stopwords=True -> virer "Le", "une", "a" (attention "a" est aussi < 3 mais est stopword)
    # - longueur_min=3 -> virer "a" (déjà fait), potentiellement d'autres si courts. "chat", "mange", "souris" >=3.
    # - occ_min=2 -> garder seulement ceux qui apparaissent >= 2 fois après nettoyage ?
    # Attention: l'ordre compte.
    # 1. Alpha: ["Le", "chat", "mange", "une", "souris", "chat", "a"]
    # 2. Stopwords: ["chat", "mange", "souris", "chat"]
    # 3. Len >=3: ["chat", "mange", "souris", "chat"]
    # 4. Occ >= 2: "chat" (2), "mange" (1), "souris" (1) -> Reste ["chat", "chat"]
    
    config_test = {
        "stopwords": True,
        "longueur_min": 3,
        "non_alphabetiques": True,
        "occ_min": 2,
        "occ_max": None
    }
    
    res_pipeline = pipeline_filtrage(tokens_pipeline, config_test, langue="fr")
    assert "chat" in res_pipeline
    assert "mange" not in res_pipeline # occ=1
    assert "123" not in res_pipeline # non alpha
    assert len(res_pipeline) == 2 # chat, chat
    print("✅ Test Pipeline Filtrage passé.")

  # --- Tests Normalisation & Pipeline Morphologique ---
    print("\n--- Démarrage des tests Normalisation ---")
    
    # 1. Stemming
    tokens_stem_fr = ["rapidement", "chanteuses", "as", "tapis"] 
    stems_fr = appliquer_stemming(tokens_stem_fr, langue="fr")
    assert "rapid" in stems_fr
    assert "chant" in stems_fr
    assert "as" in stems_fr # Protection longueur
    
    # 2. Lemmatisation
    tokens_lemm = ["étudiants", "mangeaient", "robot"]
    lemmes_fr = appliquer_lemmatisation(tokens_lemm, langue="fr")
    assert "étudiant" in lemmes_fr
    assert "manger" in lemmes_fr
    assert "robot" in lemmes_fr # Pas dans le dico -> reste robot
    print("✅ Test Stemming & Lemmatisation passés.")

    print("\n--- Comparaison Stratégies Morphologiques ---")
    # Corpus de test :
    # "étudiants" (lemme: étudiant, stem: étudiant)
    # "étudiante" (lemme: étudiant, stem: étudiant)
    # "étudiées" (lemme: étudier, stem: étudi)
    # "chanteuses" (lemme: chanteuses (inconnu), stem: chant)
    tokens_bruts = ["étudiants", "étudiante", "étudiées", "chanteuses"]
    
    configs = {
        "Aucun": {"stemming": False, "lemmatisation": False},
        "Stemming seul": {"stemming": True, "lemmatisation": False},
        "Lemmatisation seule": {"stemming": False, "lemmatisation": True},
        "Les deux (Lemm -> Stem)": {"stemming": True, "lemmatisation": True}
    }
    
    for nom, conf in configs.items():
        res = pipeline_morphologique(tokens_bruts, conf, langue="fr")
        
        # Calcul stats
        vocab_unique = set(res)
        taille_vocab = len(vocab_unique)
        
        # Calcul hapax (tokens apparaissant 1 seule fois dans res)
        compte = Counter(res)
        nb_hapax = sum(1 for c in compte.values() if c == 1)
        taux_hapax = nb_hapax / taille_vocab if taille_vocab > 0 else 0
        
        print(f"Stratégie '{nom}':")
        print(f"  -> Tokens: {res}")
        print(f"  -> Taille Vocab: {taille_vocab}")
        print(f"  -> Taux Hapax: {taux_hapax:.2f}")
        
    print("✅ Test Pipeline Morphologique & Comparaison exécuté.")
   
    # --- Tests Post-Traitement ---
    print("\n--- Démarrage des tests Post-Traitement ---")
    
    # Simulation corpus post-traitement
    # Corpus original simulé : 2 docs, 10 tokens, 8 uniques, bruit élevé
    stats_init = {
        "nb_documents": 2,
        "nb_tokens_total": 10,
        "taille_vocabulaire": 8,
        "richesse_lexicale": 0.8,
        "taux_hapax": 0.6,
        "proportion_bruit": 0.4
    }
    
    # Corpus filtré (plus propre) : "chat", "chat", "chien"
    corpus_filtre_simu = {"d1": ["chat", "chat"], "d2": ["chien"]}
    vocab_filtre_simu = {"chat": 2, "chien": 1}
    
    # Calcul stats post
    stats_post = calculer_statistiques_post_traitement(corpus_filtre_simu, vocab_filtre_simu)
    
    assert stats_post["nb_tokens_total"] == 3
    assert stats_post["taille_vocabulaire"] == 2
    assert stats_post["richesse_lexicale"] == 2/3 # 0.66
    assert stats_post["taux_hapax"] == 0.5 # 1 hapax (chien) sur 2 mots
    assert stats_post["proportion_bruit_residuel"] == 0 # Tout est propre/alpha
    
    print("✅ Test Calcul Stats Post-Traitement passé.")
    
    # Visualisation (Smoke test)
    print("Génération des graphiques comparatifs...")
    visualiser_statistiques_post_traitement(stats_init, stats_post)
    print("✅ Test Visualisation Post-Traitement exécuté.")

    # --- Tests Analyse Comparative Complète (A, B, C, D, E) ---
    print("\n--- Démarrage des tests Analyse Comparative (A-E) ---")
    
    # Construction d'un corpus de test plus complet
    # Phrase 1: "Les chats mangent des souris." (stopwords: Les, des. Stem: chat, mang, souri)
    # Phrase 2: "Le chat dort sur le tapis." (stopwords: Le, sur, le. Stem: chat, dort, tapi)
    # Phrase 3: "123 ! ." (Bruit total)
    corpus_comp = {
        "d1": [["Les", "chats", "mangent", "des", "souris", "."]],
        "d2": [["Le", "chat", "dort", "sur", "le", "tapis", "."]],
        "d3": [["123", "!", "."]]
    }
    
    res_comparatifs = analyser_configurations(corpus_comp, langue="fr")
    
    # Vérifications simples
    # Config A (Brut): Tout est gardé.
    assert res_comparatifs["A (Brut)"]["nb_tokens_total"] == 16 # Total tokens
    
    # Config B (Filtré): Stopwords virés, Non-alpha virés
    # Reste: chats, mangent, souris, chat, dort, tapis (6 tokens)
    # (Les, des, Le, sur, le sont stopwords. . ! 123 sont non alpha)
    # Note: "tapis" a 5 lettres > 3. "chats" 5 lettres. "dort" 4.
    assert res_comparatifs["B (Filtré)"]["nb_tokens_total"] == 6
    
    # Config C (Stemming): chats->chat, mangent->mang
    # Vocabulaire devrait baisser par rapport à B si regroupement
    # B vocab: chats, mangent, souris, chat, dort, tapis (6 uniques car chats != chat)
    # C vocab: chat, mang, souris, chat, dort, tapi (5 uniques : chat apparait 2 fois)
    assert res_comparatifs["C (Filtré + Stem)"]["taille_vocabulaire"] < res_comparatifs["B (Filtré)"]["taille_vocabulaire"]
    
    print("✅ Test Analyse Comparative A-E passé.")

    # --- Test Visualisation Comparative ---
    print("Génération du graphique comparatif final...")
    visualiser_comparaison_configurations(res_comparatifs)
    print("✅ Test Visualisation Comparative exécuté.")

    print("\n✅ Tous les tests sont passés avec succès !")

if __name__ == "__main__":
    test_tokenisation_complete()