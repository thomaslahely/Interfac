# ---------------------------------------------
# Gestion des fiches de requête pour l'IHM
# ---------------------------------------------

def initialiser_fiche_requete(id_requete, texte, type_unite="phrase",
                              langue=None, config_id=None):
    """
    Initialise une fiche de requête avec les informations de base.
    
    Paramètres :
    - id_requete : identifiant unique de la requête
    - texte : texte brut de la requête
    - type_unite : "phrase" ou "document"
    - langue : langue de la requête (optionnel)
    - config_id : identifiant de la configuration de prétraitement (optionnel)
    
    Retourne : dictionnaire représentant la fiche de requête
    """
    fiche_requete = {
        "id": id_requete,
        "texte": texte,
        "type_unite": type_unite,
        "langue": langue,
        "config_id": config_id,
        "statut_calcul": "non_calculé",
        "descripteurs": {},   # pour stocker TF, TF-IDF, embeddings...
        "strategie_oov": "ignore"  # comportement par défaut pour OOV
    }
    return fiche_requete

# ---------------------------------------------
# Définition du type d’unité linguistique
# ---------------------------------------------

def choisir_type_requete(fiche_requete, type_unite):
    """
    Définit si la requête est traitée comme une phrase ou un document.
    """
    if type_unite not in ["phrase", "document"]:
        raise ValueError("type_unite doit être 'phrase' ou 'document'")
    fiche_requete["type_unite"] = type_unite

# ---------------------------------------------
# Définition de la source de la requête
# ---------------------------------------------

def choisir_source_requete(fiche_requete, source, chemin_fichier=None):
    """
    Définit l'origine de la requête :
    - 'utilisateur' : saisie directe
    - 'corpus' : sélection depuis le corpus
    - 'fichier' : chargement depuis un fichier externe
    """
    if source not in ["utilisateur", "corpus", "fichier"]:
        raise ValueError("source doit être 'utilisateur', 'corpus' ou 'fichier'")
    
    fiche_requete["source"] = source
    
    if source == "fichier":
        if chemin_fichier is None:
            raise ValueError("Pour une source 'fichier', chemin_fichier doit être fourni")
        # Charger le texte depuis le fichier
        with open(chemin_fichier, "r", encoding="utf-8") as f:
            texte = f.read()
        fiche_requete["texte"] = texte

# ---------------------------------------------
# Définition de la langue
# ---------------------------------------------

def definir_langue_requete(fiche_requete, langue):
    """
    Assigne la langue de la requête.
    """
    fiche_requete["langue"] = langue

# ---------------------------------------------
# Définition de la configuration de prétraitement
# ---------------------------------------------

def choisir_configuration_pretraitement(fiche_requete, config_id):
    """
    Associe une configuration de prétraitement à la requête.
    """
    fiche_requete["config_id"] = config_id
    # Tout changement de config réinitialise les descripteurs
    fiche_requete["descripteurs"] = {}
    fiche_requete["statut_calcul"] = "non_calculé"

# ---------------------------------------------
# Définition de la stratégie OOV
# ---------------------------------------------

def choisir_strategie_oov_requete(fiche_requete, strategie_oov):
    """
    Définit la stratégie de gestion des mots absents du vocabulaire (OOV)
    Stratégies possibles :
    - 'ignore' : ignorer les mots absents
    - 'substituer' : remplacer par synonymes ou sous-mots
    - 'alerte' : signaler explicitement les mots absents
    """
    if strategie_oov not in ["ignore", "substituer", "alerte"]:
        raise ValueError("strategie_oov doit être 'ignore', 'substituer' ou 'alerte'")
    fiche_requete["strategie_oov"] = strategie_oov

# ---------------------------------------------
# Configuration du corpus ciblé pour la recherche
# ---------------------------------------------

def definir_granularite_corpus(fiche_requete, granularite="document"):
    """
    Définit le niveau de granularité de la recherche : 'phrase' ou 'document'.
    """
    if granularite not in ["phrase", "document"]:
        raise ValueError("granularite doit être 'phrase' ou 'document'")
    fiche_requete["granularite"] = granularite

def definir_portee_corpus(fiche_requete, sous_corpus=None, langues=None):
    """
    Définit la portée de la recherche :
    - sous_corpus : liste d'identifiants ou de noms de sous-corpus à inclure (optionnel)
    - langues : liste de langues à inclure (optionnel)
    """
    fiche_requete["portee_corpus"] = {
        "sous_corpus": sous_corpus,   # None = tout le corpus
        "langues": langues            # None = toutes les langues
    }

def definir_filtres_corpus(fiche_requete, filtres=None):
    """
    Applique des filtres supplémentaires pour restreindre la recherche.
    - filtres : dictionnaire de critères, par exemple :
      { "taille_min": 50, "taille_max": 500, "sous_corpus": "sous_corpus1", ... }
    """
    fiche_requete["filtres_corpus"] = filtres if filtres else {}

# ---------------------------------------------
# Choix des descripteurs et paramètres
# ---------------------------------------------

def choisir_descripteurs(fiche_requete, descripteurs=["TF-IDF"],
                         methode_aggregation=None,
                         normalisation_norme=None,
                         normalisation_avancee=None):
    """
    Associe à la requête les descripteurs à utiliser pour le calcul de similarité.
    
    Paramètres :
    - descripteurs : liste de descripteurs symboliques ou vectoriels
      Exemples : ["BOW", "TF", "TF-IDF", "BM25", "embeddings"]
    - methode_aggregation : méthode pour agréger des embeddings (moyenne, somme, Doc2Vec)
    - normalisation_norme : type de normalisation de la norme (L1, L2, None)
    - normalisation_avancee : normalisation statistique optionnelle (MinMax, Z-score, None)
    """
    fiche_requete["descripteurs_utilises"] = {
        "types": descripteurs,
        "aggregation": methode_aggregation,
        "norme": normalisation_norme,
        "normalisation_avancee": normalisation_avancee
    }

    # Réinitialisation des descripteurs calculés si la configuration change
    fiche_requete["descripteurs"] = {}
    fiche_requete["statut_calcul"] = "non_calculé"

# ---------------------------------------------
# Définition de la mesure de similarité ou distance
# ---------------------------------------------

def definir_distance(fiche_requete, type_distance="cosinus"):
    """
    Associe à la requête une mesure de similarité ou distance.
    
    Paramètres possibles :
    - Cosinus : 'cosinus' (TF-IDF, BM25, embeddings)
    - Euclidienne : 'euclidienne' (embeddings)
    - Manhattan : 'manhattan' (embeddings)
    - Minkowski : 'minkowski' (embeddings, nécessite p)
    - Jaccard : 'jaccard' (symbolique)
    - Scores pondérés ou combinés : 'combine' (hybride)
    
    La fonction ne calcule rien, elle configure la fiche de requête pour le moteur.
    Les distances non applicables selon le type de descripteur devront être désactivées
    dans l'IHM.
    """
    distances_valides = ["cosinus", "euclidienne", "manhattan", "minkowski", "jaccard", "combine"]
    
    if type_distance not in distances_valides:
        raise ValueError(f"Type de distance '{type_distance}' non reconnu. "
                         f"Choisir parmi {distances_valides}.")
    
    fiche_requete["type_distance"] = type_distance

# ---------------------------------------------
# Paramètres avancés de la requête
# ---------------------------------------------

def definir_options_avancees(fiche_requete,
                              expansion_requete=False,
                              multi_langues=False,
                              pre_calcul_normes=True,
                              nb_termes_expansion=5):
    """
    Définit les paramètres avancés pour la recherche :
    - expansion_requete : bool, active ou non l'enrichissement automatique de la requête
    - multi_langues : bool, active la stratégie multi-langues
    - pre_calcul_normes : bool, active le pré-calcul des normes pour accélérer les similarités
    - nb_termes_expansion : nombre de termes/vecteurs ajoutés si expansion activée
    """
    fiche_requete["options_avancees"] = {
        "expansion_requete": expansion_requete,
        "multi_langues": multi_langues,
        "pre_calcul_normes": pre_calcul_normes,
        "nb_termes_expansion": nb_termes_expansion if expansion_requete else 0
    }
