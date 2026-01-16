import sys
from contextlib import redirect_stdout
import io
import unicodedata
from pathlib import Path
import shutil
from unittest.mock import patch
from collections import defaultdict
import xml
import string
import re


BASE_TEST_DIR = Path("corpus_de_test_temporaire")

def creer_arborescence_test():
    """Crée une arborescence de dossiers/fichiers fictive pour les tests."""
    if BASE_TEST_DIR.exists():
        shutil.rmtree(BASE_TEST_DIR)
    BASE_TEST_DIR.mkdir()

    (BASE_TEST_DIR / "UFR_LIMA").mkdir()
    (BASE_TEST_DIR / "UFR_LIMA" / "etudiant1_fr.txt").touch()

def nettoyer_arborescence_test():
    """Supprime l'arborescence de test."""
    if BASE_TEST_DIR.exists():
        shutil.rmtree(BASE_TEST_DIR)

### Énoncé de la fonction 1 : Lecture

def lire_document(fichier: Path) -> str:
    """Ouvre un fichier texte et retourne son contenu."""
    try:
        with open(fichier, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Erreur : Le fichier {fichier} n'a pas été trouvé.")
        return ""
    except Exception as e:
        return ""

### 🔍 Explication des tests unitaires — fonction 1
def test_lire_document():
    creer_arborescence_test()
    p = BASE_TEST_DIR / "test.txt"
    p.write_text("Contenu test", encoding="utf-8")
    assert lire_document(p) == "Contenu test"
    print("Test lire_document : OK")


### Énoncé des fonctions de Standardisation

# **Titres :**
# > convertir_vers_minuscule(texte)
# > supprimer_balises_html_xml(texte)
# > normaliser_unicode(texte)

# **Consigne :**
# * Harmoniser la casse, retirer les balises techniques et normaliser les accents (NFC).

def convertir_vers_minuscule(texte: str) -> str:
    """Convertit tout le texte en minuscules."""
    return texte.lower() if isinstance(texte, str) else ""

def supprimer_balises_html_xml(texte: str) -> str:
    """Retire les balises de type <tag>."""
    return re.sub(r'<[^>]+>', '', texte)

def normaliser_unicode(texte: str) -> str:
    """Normalise le texte selon la forme Unicode standard NFC."""
    return unicodedata.normalize('NFC', texte)

### 🔍 Explication des tests unitaires — Standardisation

# 1. **Minuscule :** Tester "TeSt" -> "test".
# 2. **HTML :** Tester "<div>Coucou</div>" -> "Coucou".
# 3. **Unicode :** Tester un caractère décomposé (e + accent) vs recomposé (é).

def test_standardisation():
    # Minuscule
    assert convertir_vers_minuscule("BoNjOuR") == "bonjour"
    # HTML
    assert supprimer_balises_html_xml("<p>Texte</p>") == "Texte"
    # Unicode
    nfd = "e\u0301cole" # é décomposé
    assert len(nfd) == 6
    nfc = normaliser_unicode(nfd)
    assert len(nfc) == 5 and nfc == "école"
    print("Test Standardisation : OK")


### Énoncé des fonctions Accents

# **Titres :**
# > corriger_accents(texte)
# > uniformiser_accents(texte)
# > traiter_accents(texte, options)

# **Consigne :**
# *  simplifier les accents pour la comparaison.

def corriger_accents(texte: str) -> str:
    """Corrige les erreurs d'encodage (ex: Ã© -> é)."""
    remplacements = {
        "Ã©": "é", "Ã¨": "è", "Ãª": "ê", "Ã ": "à",
        "Ã¢": "â", "Ã§": "ç", "Ã´": "ô", "Ã»": "û", "Ã¯": "ï"
    }
    res = texte
    for malform, bon in remplacements.items():
        res = res.replace(malform, bon)
    return res

def uniformiser_accents(texte: str) -> str:
    """Simplifie les accents (é -> e, à -> a)."""
    regles = str.maketrans({
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'à': 'a', 'â': 'a', 'ç': 'c',
        'ù': 'u', 'û': 'u', 'ô': 'o', 'î': 'i', 'ï': 'i'
    })
    return texte.translate(regles)

def traiter_accents(texte: str, options: dict = None) -> str:
    """Combine correction et uniformisation selon options."""
    if options is None: options = {"corriger_erreurs": True, "uniformiser": True}
    res = texte
    if options.get("corriger_erreurs"): res = corriger_accents(res)
    if options.get("uniformiser"): res = uniformiser_accents(res)
    return res

### 🔍 Explication des tests unitaires — Accents

# 1. **Correction :** "DÃ©jÃ " doit devenir "Déjà".
# 2. **Uniformisation :** "Elève" doit devenir "Eleve".

def test_accents():
    # Correction
    assert corriger_accents("DÃ©jÃ ") == "Déjà"
    # Uniformisation
    assert uniformiser_accents("Hétérogène") == "Heterogene"
    # Pipeline
    assert traiter_accents("DÃ©jÃ ", {"corriger_erreurs": True, "uniformiser": True}) == "Deja"
    print("Test Accents : OK")


###  Énoncé des fonctions Langue

# **Titres :**
# > detecter_langue_nom_fichier(fichier)
# > verifier_langue_contenu(contenu)
# > verifier_coherence_langue_contenu(fichier)
# > signaler_incoherences_langue(base)
# **Consigne :**
# * Déduire la langue du nom (_fr/_en) et vérifier la cohérence avec le contenu.

def detecter_langue_nom_fichier(fichier: Path) -> str:
    """Retourne 'fr', 'en' ou 'inconnu' selon le suffixe."""
    nom = fichier.name.lower()
    if "_fr" in nom: return "fr"
    if "_en" in nom: return "en"
    return "inconnu"

def verifier_langue_contenu(contenu: str) -> str:
    """Détermine la langue par comptage de stopwords."""
    mots = re.findall(r'\w+', contenu.lower())
    fr = {"le", "la", "et", "de", "du", "un", "est"}
    en = {"the", "and", "of", "to", "in", "is"}
    score_fr = sum(1 for m in mots if m in fr)
    score_en = sum(1 for m in mots if m in en)
    if score_fr > score_en: return "fr"
    if score_en > score_fr: return "en"
    return "indetermine"

def verifier_coherence_langue_contenu(fichier: Path) -> bool:
    """Vérifie si nom et contenu concordent."""
    langue_nom = detecter_langue_nom_fichier(fichier)
    if langue_nom == "inconnu": return True
    contenu = lire_document(fichier)
    langue_txt = verifier_langue_contenu(contenu)
    if langue_txt == "indetermine": return True
    return langue_nom == langue_txt

def signaler_incoherences_langue(base: Path) -> list:
    """
    Parcourt récursivement le dossier et liste les fichiers dont
    le suffixe de langue ne correspond pas au contenu.
    Retourne une liste de chemins (str).
    """
    anomalies = []
    # rglob('*') permet de scanner tous les sous-dossiers
    for fichier in base.rglob('*'):
        if fichier.is_file():
            # Si la cohérence est fausse, on ajoute à la liste
            if not verifier_coherence_langue_contenu(fichier):
                anomalies.append(str(fichier))
    return anomalies
### 🔍 Explication des tests unitaires — Langue

# 1. **Nom :** "file_fr.txt" -> "fr".
# 2. **Contenu :** "The cat is black" -> "en".
# 3. **Cohérence :** Fichier nommé _fr mais contenu anglais -> False.
# 4. **Signalement :** Scanner un dossier avec 1 fichier OK et 1 KO -> Doit retourner le KO.
def test_langue():
    creer_arborescence_test()
    
    # Cas 1 : Fichier COHÉRENT (Nom FR, Contenu FR)
    p_ok = BASE_TEST_DIR / "test_fr.txt"
    p_ok.write_text("Le chat est beau et le chien dort", encoding="utf-8")
    
    # Cas 2 : Fichier INCOHÉRENT (Nom FR, Contenu EN)
    p_ko = BASE_TEST_DIR / "fake_fr.txt"
    p_ko.write_text("The cat is beautiful and the dog sleeps", encoding="utf-8")
    
    # Tests unitaires
    assert detecter_langue_nom_fichier(p_ok) == "fr"
    assert verifier_langue_contenu("Le chat est beau") == "fr"
    assert verifier_coherence_langue_contenu(p_ok) is True
    assert verifier_coherence_langue_contenu(p_ko) is False
    
    # Test d'intégration (Signalement sur le dossier)
    liste_anomalies = signaler_incoherences_langue(BASE_TEST_DIR)
    
    # On doit trouver exactement 1 anomalie et ce doit être le fichier fake_fr.txt
    assert len(liste_anomalies) == 1
    assert "fake_fr.txt" in liste_anomalies[0]
    
    print("Test Langue (avec signalement) : OK")


### 📝 Énoncé des fonctions Ponctuation 

# **Objectif :** Implémenter toutes les variantes de traitement de ponctuation demandées.

# a. Supprimer toute la ponctuation
def supprimer_ponctuation(texte: str) -> str:
    """Supprime (. , ; : ! ? ( ) [ ] " ' … etc)."""
    if not texte: return ""
    motif = f"[{re.escape(string.punctuation)}]"
    return re.sub(motif, "", texte)

# b. Remplacer par une balise
def remplacer_ponctuation(texte: str, balise: str = "<PONCT>") -> str:
    """Remplace chaque signe de ponctuation par une balise."""
    if not texte: return ""
    motif = f"[{re.escape(string.punctuation)}]"
    return re.sub(motif, f" {balise} ", texte).strip()

# c. Supprimer sauf une liste définie
def supprimer_ponctuation_sauf(texte: str, ponct_a_conserver: list = None) -> str:
    """Supprime tout sauf les signes spécifiés (ex: ['.', '?'])."""
    if ponct_a_conserver is None:
        ponct_a_conserver = [".", "?"]
    
    toute_ponct = string.punctuation
    a_supprimer = "".join([c for c in toute_ponct if c not in ponct_a_conserver])
    
    if not a_supprimer: return texte
    
    motif = f"[{re.escape(a_supprimer)}]"
    resultat = re.sub(motif, "", texte)
    return re.sub(r'\s+', ' ', resultat).strip()  
# d. Espacer la ponctuation
def espacer_ponctuation(texte: str) -> str:
    """Ajoute un espace avant et après chaque signe de ponctuation collé."""
    if not texte: return ""
    # Capture un signe de ponctuation et ajoute des espaces autour
    motif = f"([{re.escape(string.punctuation)}])"
    return re.sub(motif, r" \1 ", texte).strip()

# e. Normaliser (guillemets, tirets...)
def normaliser_ponctuation(texte: str) -> str:
    """Uniformise guillemets, tirets et points de suspension."""
    texte = re.sub(r'[«»“”]', '"', texte)
    texte = re.sub(r'[–—]', '-', texte)
    texte = texte.replace('…', '...')
    return texte

# f. Réduire les répétitions
def reduire_ponctuation_multiple(texte: str) -> str:
    """Remplace les répétitions (!!! -> !)."""
    # Cherche un caractère de ponctuation suivi du MEME caractère (\1) une ou plusieurs fois
    motif = f"([{re.escape(string.punctuation)}])\\1+"
    return re.sub(motif, r"\1", texte)

# g. Remplacement contextuel (sémantique)
def remplacer_ponctuation_contextuelle(texte: str) -> str:
    """Convertit ! ? ... en balises sémantiques."""
    # Ordre important : traiter ... avant .
    texte = texte.replace("...", " <HESITATION> ")
    texte = texte.replace("!", " <EXCLAMATION> ")
    texte = texte.replace("?", " <QUESTION> ")
    return texte.strip()

# h. Wrapper global pour la ponctuation
def traiter_ponctuation(texte: str, options: dict = None) -> str:
    """Orchestre toutes les fonctions de ponctuation selon les options."""
    if options is None: options = {}
    res = texte
    
    # 1. Normalisation et Réduction (Nettoyage préliminaire)
    if options.get("normaliser"): 
        res = normaliser_ponctuation(res)
    if options.get("reduire"): 
        res = reduire_ponctuation_multiple(res)
        
    # 2. Traitement sémantique (avant de supprimer la ponctuation)
    if options.get("contextuel"): 
        res = remplacer_ponctuation_contextuelle(res)
        
    # 3. Espacement
    if options.get("espacer"): 
        res = espacer_ponctuation(res)
        
    # 4. Remplacement ou Suppression (Destructif)
    if options.get("remplacer"): 
        balise = options.get("balise", "<PONCT>")
        res = remplacer_ponctuation(res, balise)
        
    elif options.get("supprimer_sauf"): # Priorité sur "supprimer" tout court
        keep = options.get("garder", [".", "?"])
        res = supprimer_ponctuation_sauf(res, keep)
        
    elif options.get("supprimer"): 
        res = supprimer_ponctuation(res)
    
    # Nettoyage final des espaces multiples créés
    return re.sub(r'\s+', ' ', res).strip()


### 📝 Énoncé des fonctions Nombres et Symboles (Rappel concis pour le bloc)
def traiter_nombres_symboles(texte: str, options: dict = None) -> str:
    """Gère suppression ou remplacement des nombres et symboles."""
    if options is None: options = {}
    res = texte
    if options.get("remplacer_nombres"):
        res = re.sub(r'\d+', " <NUM> ", res)
    elif options.get("supprimer_nombres"):
        res = re.sub(r'\d+', "", res)
    if options.get("remplacer_symboles"):
        res = re.sub(r'[^\w\s]', " <SYM> ", res)
    return re.sub(r'\s+', ' ', res).strip()


### 🔍 Explication des tests unitaires — Structure détaillée

# La stratégie couvre chaque sous-fonction :
# 1. **Suppression** : "a,b!" -> "ab"
# 2. **Remplacement** : "a!" -> "a <PONCT>"
# 3. **Sauf** : "a! b?" (garder ?) -> "a b?"
# 4. **Espacer** : "a,b" -> "a , b"
# 5. **Normaliser** : "«a»" -> '"a"'
# 6. **Réduire** : "Stop!!!" -> "Stop!"
# 7. **Contextuel** : "Vraiment?" -> "Vraiment <QUESTION>"
# 8. **Wrapper** : Combinaison (Normaliser + Réduire + Contextuel)

def test_structure_detaillee():
    # a. Supprimer
    assert supprimer_ponctuation("Salut, toi!") == "Salut toi", "Echec a. Supprimer"
    
    # b. Remplacer
    assert remplacer_ponctuation("Hi!", "<P>") == "Hi <P>", "Echec b. Remplacer"
    
    # c. Sauf
    assert supprimer_ponctuation_sauf("Salut! Ça va?", ["?"]) == "Salut Ça va?", "Echec c. Sauf"
    
    # d. Espacer
    assert espacer_ponctuation("Non,merci") == "Non , merci", "Echec d. Espacer"
    
    # e. Normaliser
    assert normaliser_ponctuation("« Test »…") == '" Test "...', "Echec e. Normaliser"
    
    # f. Réduire
    assert reduire_ponctuation_multiple("Waouh!!!") == "Waouh!", "Echec f. Réduire"
    
    # g. Contextuel
    assert remplacer_ponctuation_contextuelle("Quoi?") == "Quoi <QUESTION>", "Echec g. Contextuel"
    
    # h. Wrapper (Test complet)
    # Scénario : On veut normaliser les guillemets, réduire les !!! et marquer les ?
    opts = {
        "normaliser": True,
        "reduire": True,
        "contextuel": True,
        "espacer": False
    }
    raw = "« C'est fou!!! » Vraiment?..."
    # Étapes attendues :
    # 1. Normaliser : " C'est fou!!! " Vraiment...
    # 2. Réduire : " C'est fou! " Vraiment...
    # 3. Contextuel : " C'est fou <EXCLAMATION> " Vraiment <HESITATION>
    res = traiter_ponctuation(raw, opts)
    
    assert '"' in res # Guillemets normalisés
    assert "<EXCLAMATION>" in res # ! traité
    assert "!!!" not in res # Réduit
    assert "<HESITATION>" in res # ... traité
    
    print("Test Structure Détaillée (a-h) : OK")


### 📝 Énoncé des fonctions Nombres et Symboles

# **Objectif :** Nettoyer, remplacer ou convertir les données numériques et symboliques.

# a. Supprimer les nombres
def supprimer_nombres(texte: str) -> str:
    """Supprime tous les chiffres et nombres (entiers ou décimaux)."""
    # \d+ capture une suite de chiffres. 
    # Pour gérer les décimaux (ex: 3.5), on pourrait complexifier, 
    # mais \d+ suffit pour "2025" ou "3".
    return re.sub(r'\d+', '', texte).strip()

# b. Remplacer les nombres par une balise
def remplacer_nombres(texte: str, balise: str = "<NUM>") -> str:
    """Remplace chaque nombre par une balise donnée."""
    # On ajoute des espaces autour de la balise pour la lisibilité
    return re.sub(r'\d+', f" {balise} ", texte).strip()

# c. Convertir nombres en lettres (Simplifié)
def nombres_en_lettres(texte: str, langue: str = "fr") -> str:
    """
    Convertit les chiffres simples (0-9) en mots.
    Note : Pour une conversion complète (ex: 2025 -> deux mille...),
    il faut installer la librairie 'num2words'. Ici, implémentation légère.
    """
    mapping = {
        '0': 'zéro', '1': 'un', '2': 'deux', '3': 'trois', '4': 'quatre',
        '5': 'cinq', '6': 'six', '7': 'sept', '8': 'huit', '9': 'neuf'
    }
    
    # Fonction de remplacement interne
    def _remplace(match):
        chiffre = match.group(0)
        return mapping.get(chiffre, chiffre)
    
    # \b assure qu'on ne remplace pas le '3' dans 'mp3'
    return re.sub(r'\b\d\b', _remplace, texte)

# d. Supprimer les symboles
def supprimer_symboles(texte: str) -> str:
    """Supprime les symboles non alphanumériques (@, #, $, etc.)."""
    # [^\w\s] = Tout ce qui n'est PAS (^) un mot (\w) ou un espace (\s)
    return re.sub(r'[^\w\s]', '', texte).strip()

# e. Remplacer les symboles
def remplacer_symboles(texte: str, balise: str = "<SYMBOLE>") -> str:
    """Remplace chaque symbole par une balise."""
    return re.sub(r'[^\w\s]', f" {balise} ", texte).strip()

# f. Remplacer les unités
def remplacer_unites(texte: str, balise: str = "<UNITE>") -> str:
    """Remplace €, $, %, km, kg, °C."""
    # On utilise | pour "OU". On échappe le $ et le . si besoin.
    unites_regex = r'(km|kg|cm|mm|€|\$|£|%|°C)'
    return re.sub(unites_regex, f" {balise} ", texte, flags=re.IGNORECASE).strip()

# g. Supprimer non alphabétiques (Radical)
def supprimer_non_alphabetiques(texte: str) -> str:
    """Ne garde que les lettres et espaces (supprime chiffres et symboles)."""
    # On garde a-z, A-Z, les accents courants et les espaces.
    # Tout le reste est supprimé.
    pattern = r'[^a-zA-Z\sàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]'
    return re.sub(pattern, '', texte).strip()

# h. Remplacer non alphabétiques
def remplacer_non_alphabetiques(texte: str, balise: str = "<NON_ALPHA>") -> str:
    """Marque les caractères non alphabétiques (chiffres/symboles) par une balise."""
    pattern = r'[^a-zA-Z\sàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]+' # Le + groupe les séries
    return re.sub(pattern, f" {balise} ", texte).strip()

# i. Normaliser symboles
def normaliser_symboles(texte: str) -> str:
    """Convertit +, -, =, % en équivalents textuels."""
    table = {
        "+": " plus ",
        "-": " moins ",
        "=": " égal ",
        "%": " pourcent "
    }
    res = texte
    for sym, mot in table.items():
        res = res.replace(sym, mot)
    return res.strip()

# j. Wrapper global pour Nombres et Symboles
def traiter_nombres_symboles(texte: str, options: dict = None) -> str:
    """Applique sélectivement les transformations selon les options."""
    if options is None: options = {}
    res = texte
    
    # 1. Normalisation (ex: % -> pourcent) avant de supprimer les symboles
    if options.get("normaliser"):
        res = normaliser_symboles(res)
        
    # 2. Traitement des unités (avant les symboles génériques)
    if options.get("remplacer_unites"):
        res = remplacer_unites(res, options.get("balise_unite", "<UNITE>"))
        
    # 3. Traitement des Nombres
    if options.get("nombres_en_lettres"):
        res = nombres_en_lettres(res, options.get("langue", "fr"))
        
    if options.get("remplacer_nombres"):
        res = remplacer_nombres(res, options.get("balise_nombre", "<NUM>"))
    elif options.get("supprimer_nombres"):
        res = supprimer_nombres(res)
        
    # 4. Traitement des Symboles / Non-Alphabetique
    if options.get("remplacer_non_alphabetiques"):
         res = remplacer_non_alphabetiques(res, options.get("balise_non_alpha", "<NON_ALPHA>"))
    elif options.get("supprimer_non_alphabetiques"):
         res = supprimer_non_alphabetiques(res)
    else:
        # Traitement fin des symboles si on n'a pas fait le "non_alphabetique" radical
        if options.get("remplacer_symboles"):
            res = remplacer_symboles(res, options.get("balise_symbole", "<SYMBOLE>"))
        elif options.get("supprimer_symboles"):
            res = supprimer_symboles(res)
            
    # Nettoyage final des espaces
    return re.sub(r'\s+', ' ', res).strip()


### 🔍 Explication des tests unitaires — Nombres et Symboles

# La stratégie de test couvre :
# 1. **Nombres** : Suppression ("a 1 b" -> "a b"), Remplacement ("a 1 b" -> "a <NUM> b"), Conversion ("2" -> "deux").
# 2. **Symboles** : Suppression ("#a" -> "a"), Remplacement ("@" -> "<SYMBOLE>").
# 3. **Unités** : "5km" -> "5 <UNITE>".
# 4. **Radical** : "User123!" -> "User".
# 5. **Normalisation** : "10%" -> "10 pourcent".
# 6. **Pipeline (j)** : Tester une combinaison d'options (ex: normaliser % puis remplacer nombres).

def test_nombres_symboles_detaille():
    # a. Supprimer nombres
    assert supprimer_nombres("En 2025 il aura 3 ans") == "En  il aura  ans"
    
    # b. Remplacer nombres
    assert "<NUM>" in remplacer_nombres("Le prix est 30 euros")
    
    # c. Nombres en lettres
    assert nombres_en_lettres("Il a 2 enfants") == "Il a deux enfants"
    
    # d. Supprimer symboles
    assert supprimer_symboles("#Hello@World!") == "HelloWorld"
    
    # e. Remplacer symboles
    assert remplacer_symboles("Prix = 10") == "Prix <SYMBOLE> 10"
    
    # f. Remplacer unités
    assert remplacer_unites("Distance 5 km") == "Distance 5 <UNITE>"
    
    # g. Supprimer non alphabétiques
    assert supprimer_non_alphabetiques("Bonjour123 !!!") == "Bonjour"
    
    # h. Remplacer non alphabétiques
    assert "<NON_ALPHA>" in remplacer_non_alphabetiques("123!!!")
    
    # i. Normaliser symboles
    assert normaliser_symboles("10% + 5") == "10 pourcent   plus  5"
    
    # j. Wrapper (Pipeline)
    # Scénario : On veut "10% de +" -> "<NUM> pourcent de plus"
    opts = {
        "normaliser": True, # % -> pourcent, + -> plus
        "remplacer_nombres": True, # 10 -> <NUM>
        "supprimer_symboles": False
    }
    entree = "10% de + que l'an dernier"
    res = traiter_nombres_symboles(entree, opts)
    
    # Vérifications
    assert "pourcent" in res
    assert "plus" in res
    assert "<NUM>" in res
    assert "10" not in res
    
    print("Test Nombres et Symboles (a-j) : OK")
### 📝 Énoncé des fonctions Web et Social

# **Objectif :** Gérer les artefacts spécifiques au web et aux réseaux sociaux.

# --- URLs et E-mails ---

def supprimer_urls(texte: str) -> str:
    """Supprime les adresses web (http, https, ftp, www)."""
    # Regex : cherche http/ftp OU www. suivi de caractères non-blancs
    return re.sub(r'(https?://|ftp://|www\.)\S+', '', texte).strip()

def remplacer_urls(texte: str, balise: str = "<URL>") -> str:
    """Remplace les URLs par une balise."""
    return re.sub(r'(https?://|ftp://|www\.)\S+', f" {balise} ", texte).strip()

def supprimer_emails(texte: str) -> str:
    """Supprime les adresses e-mail."""
    # Regex standard pour email
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.sub(regex, '', texte).strip()

def remplacer_emails(texte: str, balise: str = "<EMAIL>") -> str:
    """Remplace les emails par une balise."""
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.sub(regex, f" {balise} ", texte).strip()


# --- Mentions et Hashtags ---

def supprimer_mentions(texte: str) -> str:
    """Supprime les mentions (@nom)."""
    # \w+ capture lettres, chiffres et underscores
    return re.sub(r'@\w+', '', texte).strip()

def remplacer_mentions(texte: str, balise: str = "<MENTION>") -> str:
    """Remplace les mentions par une balise."""
    return re.sub(r'@\w+', f" {balise} ", texte).strip()

def supprimer_hashtags(texte: str) -> str:
    """Supprime les hashtags (#mot)."""
    return re.sub(r'#\w+', '', texte).strip()

def remplacer_hashtags(texte: str, balise: str = "<HASHTAG>") -> str:
    """Remplace les hashtags par une balise."""
    return re.sub(r'#\w+', f" {balise} ", texte).strip()


# --- Emojis ---

def supprimer_emojis(texte: str) -> str:
    """Supprime les émojis et pictogrammes via Unicode."""
    # 'So' = Symbol, other (contient la majorité des émojis graphiques)
    return "".join(c for c in texte if unicodedata.category(c) != 'So')

def remplacer_emojis(texte: str, balise: str = "<EMOJI>") -> str:
    """Remplace les émojis par une balise."""
    res = ""
    for c in texte:
        if unicodedata.category(c) == 'So':
            res += f" {balise} "
        else:
            res += c
    return re.sub(r'\s+', ' ', res).strip()


# --- Caractères non standard ---

def supprimer_caracteres_speciaux(texte: str) -> str:
    """Supprime les caractères invisibles ou non imprimables (Contrôle)."""
    # Catégorie 'C' (Cc, Cf, Cn...) = Caractères de contrôle (tab, return, null...)
    # On garde les espaces classiques (' ') explicitement
    return "".join(c for c in texte if not unicodedata.category(c).startswith('C') or c == ' ')

def remplacer_caracteres_speciaux(texte: str, balise: str = "<SPEC>") -> str:
    """Remplace les caractères spéciaux par une balise."""
    res = ""
    for c in texte:
        if unicodedata.category(c).startswith('C') and c != ' ':
            res += balise
        else:
            res += c
    return res


# --- Pipeline Global Web ---

def traiter_web_et_emojis(texte: str, options: dict = None) -> str:
    """
    Combine les traitements Web/Social selon un dictionnaire d'options.
    """
    if options is None: options = {}
    res = texte
    
    # 1. Normalisation Unicode (optionnelle ici, mais souvent requise avant regex)
    if options.get("normaliser_unicode"):
        res = unicodedata.normalize('NFC', res)
        
    # 2. URLs (Prioritaire car contient souvent des points ou @)
    if options.get("remplacer_urls"): res = remplacer_urls(res)
    elif options.get("supprimer_urls"): res = supprimer_urls(res)
        
    # 3. Emails
    if options.get("remplacer_emails"): res = remplacer_emails(res)
    elif options.get("supprimer_emails"): res = supprimer_emails(res)
        
    # 4. Réseaux sociaux
    if options.get("remplacer_mentions"): res = remplacer_mentions(res)
    elif options.get("supprimer_mentions"): res = supprimer_mentions(res)
    
    if options.get("remplacer_hashtags"): res = remplacer_hashtags(res)
    elif options.get("supprimer_hashtags"): res = supprimer_hashtags(res)

    # 5. Emojis
    if options.get("remplacer_emojis"): res = remplacer_emojis(res)
    elif options.get("supprimer_emojis"): res = supprimer_emojis(res)
        
    # 6. Caractères Spéciaux (en dernier pour ne pas casser la structure avant)
    if options.get("remplacer_caracteres_speciaux"): 
        res = remplacer_caracteres_speciaux(res)
    elif options.get("supprimer_caracteres_speciaux"): 
        res = supprimer_caracteres_speciaux(res)
        
    # Nettoyage des espaces multiples générés par les remplacements
    return re.sub(r'\s+', ' ', res).strip()


### 🔍 Explication des tests unitaires — Web et Social

# La stratégie de test valide chaque composant isolément, puis le pipeline :
# 1. **URL** : Tester les formes http et www.
# 2. **Email** : Tester suppression et remplacement.
# 3. **Social** : Tester @user et #hashtag.
# 4. **Emoji** : Tester un caractère graphique (ex: 🔥).
# 5. **Spécial** : Tester une tabulation (\t) ou saut de ligne (\n).
# 6. **Pipeline** : Combiner plusieurs éléments pour vérifier qu'ils ne se marchent pas dessus.

def test_web_social_detaille():
    # a/b. URLs
    txt_url = "Visitez https://test.com pour info"
    assert supprimer_urls(txt_url) == "Visitez pour info"
    assert "<URL>" in remplacer_urls(txt_url)

    # c/d. Emails
    txt_mail = "Mail: contact@test.fr"
    assert supprimer_emails(txt_mail) == "Mail:"
    assert "<EMAIL>" in remplacer_emails(txt_mail)

    # Social (Mentions/Hashtags)
    txt_social = "Merci @Pierre pour le #code"
    assert supprimer_mentions(txt_social) == "Merci pour le #code"
    assert supprimer_hashtags(txt_social) == "Merci @Pierre pour le"
    assert "<MENTION>" in remplacer_mentions(txt_social)
    
    # Emojis
    txt_emoji = "Bravo 👏🔥"
    assert supprimer_emojis(txt_emoji) == "Bravo"
    assert "<EMOJI>" in remplacer_emojis(txt_emoji)

    # Caractères spéciaux
    txt_spec = "Ligne1\nLigne2\tTab"
    # supprimer_caracteres_speciaux retire \n et \t
    assert supprimer_caracteres_speciaux(txt_spec) == "Ligne1Ligne2Tab"
    
    # Pipeline Global
    opts = {
        "remplacer_urls": True,
        "supprimer_mentions": True,
        "remplacer_emojis": True,
        "normaliser_unicode": True
    }
    raw = "Voir www.site.com @moi 😉"
    # Attendu : "Voir <URL> <EMOJI>" (@moi supprimé)
    res = traiter_web_et_emojis(raw, opts)
    
    assert "<URL>" in res
    assert "@moi" not in res
    assert "<EMOJI>" in res
    
    print("Test Web et Social (Complet) : OK")
# ==============================================================================
# BLOC 7 : EXPANSION LINGUISTIQUE
# ==============================================================================
### 📝 Énoncé des fonctions d'Expansion et Correction

# **Objectif :** Normaliser les formes contractées, abréviations et variantes lexicales.

# c. Normaliser les apostrophes (Préalable indispensable)
def normaliser_apostrophes(texte: str) -> str:
    """
    Uniformise les apostrophes (’, ‘, `) vers l'apostrophe ASCII (').
    Facilite grandement les regex suivantes.
    """
    # Remplace l'apostrophe typographique, le guillemet ouvrant et l'accent grave
    return re.sub(r"[’‘`]", "'", texte)

# a. Expansion Anglaise
def expand_contractions_en(texte: str) -> str:
    """Développe les contractions anglaises (I'm -> I am)."""
    contractions_en = { 
        "I'm": "I am", "you're": "you are", "he's": "he is", "she's": "she is", 
        "it's": "it is", "we're": "we are", "they're": "they are", 
        "I've": "I have", "we've": "we have", "they've": "they have", 
        "I'll": "I will", "we'll": "we will", "won't": "will not", 
        "can't": "cannot", "n't": " not" 
    }
    
    res = texte
    # On utilise \b pour s'assurer qu'on matche un mot entier (ou début de mot)
    for k, v in contractions_en.items():
        res = re.sub(r"(?i)\b" + re.escape(k) + r"\b", v, res)
    return res

# b. Expansion Française
def expand_contractions_fr(texte: str) -> str:
    """
    Développe les contractions françaises (c'est -> ce est).
    Sensible à la casse via regex (?i) mais remplace par la forme du dictionnaire.
    """
    contractions_fr = { 
        "c'": "ce ", "C'": "Ce ", # Gestion explicite Majuscule
        "j'": "je ", "J'": "Je ",
        "l'": "le ", "L'": "Le ",
        "qu'": "que ", "Qu'": "Que ",
        "s'": "se ", "S'": "Se ",
        "t'": "te ", "n'": "ne ", "d'": "de ", "m'": "me " 
    }
    
    res = texte
    for k, v in contractions_fr.items():
        # Attention : en FR, l'apostrophe colle au mot suivant. 
        # On ne met PAS de \b après l'apostrophe, mais avant.
        res = re.sub(r"\b" + re.escape(k), v, res)
    return res

# d. Orchestrateur Expansion
def expand_contractions(texte: str, langue: str = "auto") -> str:
    """Détecte la langue et applique l'expansion correspondante."""
    # 1. Toujours normaliser les apostrophes d'abord
    texte = normaliser_apostrophes(texte)
    
    # 2. Détection simpliste si auto
    if langue == "auto":
        # Compte basique de mots très fréquents
        mots_en = {"the", "and", "is", "of"}
        tokens = set(re.findall(r'\w+', texte.lower()))
        score_en = len(tokens.intersection(mots_en))
        langue = "en" if score_en > 0 else "fr"
        
    if langue == "en":
        return expand_contractions_en(texte)
    else:
        return expand_contractions_fr(texte)

# Traitement des abréviations (a)
def developper_abreviations(texte: str, langue: str = "fr") -> str:
    """Remplace M. -> Monsieur, Dr -> Docteur, etc."""
    abrevs_fr = { "M.": "Monsieur", "Dr": "Docteur", "av.": "avenue", "n°": "numéro" }
    abrevs_en = { "Mr.": "Mister", "Dr.": "Doctor", "St.": "Street", "No.": "Number" }
    
    dico = abrevs_en if langue == "en" else abrevs_fr
    
    res = texte
    for abrev, full in dico.items():
        # On échappe le point éventuel de l'abréviation
        pattern = re.escape(abrev)
        # On remplace si c'est un mot entier (\b au début)
        # Pour la fin, si c'est un point, \b ne marche pas toujours comme on veut,
        # mais ici on suppose une abréviation suivie d'espace ou fin de phrase.
        res = re.sub(r"\b" + pattern, full, res)
    return res

# Correction contractions multiples (b)
def corriger_contractions_multiples(texte: str) -> str:
    """Traite les cas ambigus ou combinés (slang, oral)."""
    mapping = {
        "shouldn't've": "should not have",
        "y'all": "you all",
        "t'as": "tu as",
        "p'tit": "petit",
        "chk": "check"
    }
    res = texte
    for k, v in mapping.items():
         res = re.sub(r"(?i)\b" + re.escape(k) + r"\b", v, res)
    return res

# Correction typographique (a)
def corriger_fautes_typographiques(texte: str) -> str:
    """
    Réduit les répétitions de lettres (ex: suuuper -> super).
    Stratégie : remplace 3 caractères identiques ou plus par 1 seul.
    """
    return re.sub(r'(\w)\1{2,}', r'\1', texte)

# Uniformisation des variantes (b)
def uniformiser_variantes(texte: str) -> str:
    """Harmonise des termes techniques (ex: covid-19 -> covid)."""
    # (?i) = insensible à la casse
    # [-_\s]? = séparateur optionnel (tiret, underscore ou espace)
    res = re.sub(r'(?i)\bcovid[-_\s]?19\b', 'covid', texte)
    return res

# c. Wrapper Global Linguistique
def traiter_contractions(texte: str, options: dict = None) -> str:
    """Pipeline complet pour l'expansion et la correction linguistique."""
    if options is None: options = {}
    
    res = texte
    langue = options.get("langue", "auto")
    
    # 1. Apostrophes (Toujours prioritaire)
    if options.get("normaliser_apostrophes"):
        res = normaliser_apostrophes(res)
        
    # 2. Cas complexes
    if options.get("corriger_multiples"):
        res = corriger_contractions_multiples(res)
        
    # 3. Expansion classique
    if options.get("expand_contractions"):
        res = expand_contractions(res, langue=langue)
        
    # 4. Abréviations
    if options.get("developper_abreviations"):
        res = developper_abreviations(res, langue=langue)
        
    # 5. Typographie et Variantes
    if options.get("corriger_typo"):
        res = corriger_fautes_typographiques(res)
        
    if options.get("uniformiser"):
        res = uniformiser_variantes(res)
        
    # Nettoyage des espaces potentiels générés
    return re.sub(r'\s+', ' ', res).strip()


### 🔍 Explication des tests unitaires — Linguistique Détaillé

# La stratégie de test couvre :
# 1. **Apostrophes** : Vérifier que ’ devient '.
# 2. **Expansion EN** : "I'm" -> "I am".
# 3. **Expansion FR** : "C'est" -> "Ce est" (Gestion majuscule).
# 4. **Abréviations** : "M. Dupont" -> "Monsieur Dupont".
# 5. **Multiples** : "y'all" -> "you all".
# 6. **Typo** : "Boonjour" (double 'o' ok) vs "Booooonjour" (5 'o' -> 1).
# 7. **Pipeline** : Vérifier l'enchaînement Apostrophe -> Expansion -> Abréviation.

def test_linguistique_detaille():
    # c. Apostrophes
    assert normaliser_apostrophes("L’été") == "L'été"
    
    # a. Expansion EN
    assert expand_contractions_en("I'm happy") == "I am happy"
    
    # b. Expansion FR
    assert expand_contractions_fr("C'est la vie") == "Ce est la vie"
    assert expand_contractions_fr("l'arbre") == "le arbre"
    
    # d. Auto-détection
    assert expand_contractions("It's sunny", "auto") == "It is sunny"
    
    # Abréviations
    assert developper_abreviations("Bonjour M. Pierre", "fr") == "Bonjour Monsieur Pierre"
    assert developper_abreviations("Hello Mr. Smith", "en") == "Hello Mister Smith"
    
    # Contractions Multiples
    assert corriger_contractions_multiples("y'all") == "you all"
    
    # Correction Typo
    assert corriger_fautes_typographiques("Suuuper") == "Super"
    
    # Uniformisation
    assert uniformiser_variantes("Le virus COVID-19") == "Le virus covid"
    
    # Pipeline Global
    opts = {
        "langue": "fr",
        "normaliser_apostrophes": True,
        "expand_contractions": True,
        "developper_abreviations": True,
        "corriger_typo": True
    }
    raw = "C’est M. Dupond!!! Trooop bien."
    # 1. Apostrophe : C'est...
    # 2. Expansion : Ce est...
    # 3. Abréviation : ...Monsieur...
    # 4. Typo : Trop
    res = traiter_contractions(raw, opts)
    
    assert "Ce est" in res
    assert "Monsieur" in res
    assert "Trop" in res
    
    print("Test Linguistique (Expansion & Correction) : OK")

### 📝 Énoncé des fonctions de Mise en forme

# **Objectif :** Standardiser les espacements, les retours à la ligne et la présentation.

# a. Supprimer espaces multiples
def supprimer_espaces_multiples(texte: str) -> str:
    """Remplace toute suite d'espaces ou tabulations par un seul espace."""
    # [ \t]+ capture espace ou tab répété.
    return re.sub(r'[ \t]+', ' ', texte)

# b. Supprimer espaces aux bords
def supprimer_espaces_bords(texte: str) -> str:
    """Supprime les espaces inutiles au début/fin de la chaîne ET de chaque ligne."""
    # On découpe par ligne, on strip chaque ligne, et on recolle.
    lignes = [ligne.strip() for ligne in texte.splitlines()]
    return "\n".join(lignes).strip()

# c. Normaliser retours ligne
def normaliser_retours_ligne(texte: str) -> str:
    """Remplace \r, \r\n par \n."""
    return texte.replace('\r\n', '\n').replace('\r', '\n')

# d. Supprimer lignes vides
def supprimer_lignes_vides(texte: str) -> str:
    """Supprime les lignes vides ou successions de plusieurs retours à la ligne."""
    # Remplace 2 sauts de ligne ou plus (avec espaces potentiels entre) par un seul
    return re.sub(r'\n\s*\n', '\n', texte).strip()

# e. Remplacer tabulations
def remplacer_tabulations(texte: str, nb_espaces: int = 1) -> str:
    """Remplace chaque tabulation par n espaces."""
    return texte.replace('\t', ' ' * nb_espaces)

# f. Supprimer espaces AVANT ponctuation
def supprimer_espaces_avant_ponctuation(texte: str) -> str:
    """ "Bonjour , ça va ?" -> "Bonjour, ça va ?" """
    # Capture espaces suivis d'une ponctuation forte ou faible
    return re.sub(r'\s+([.,;:?!])', r'\1', texte)

# g. Ajouter espace APRÈS ponctuation
def ajouter_espace_apres_ponctuation(texte: str) -> str:
    """ "Bonjour,ça va?" -> "Bonjour, ça va?" """
    # Lookahead (?=[^\s]) : suivi par quelque chose qui n'est pas un espace
    return re.sub(r'([.,;:?!])(?=[^\s])', r'\1 ', texte)

# h. Nettoyer espaces (Combine tout)
def nettoyer_espaces(texte: str) -> str:
    """Combine les fonctions de base pour un nettoyage standard."""
    res = normaliser_retours_ligne(texte)
    res = remplacer_tabulations(res)
    res = supprimer_espaces_multiples(res)
    res = supprimer_espaces_avant_ponctuation(res)
    res = ajouter_espace_apres_ponctuation(res)
    res = supprimer_lignes_vides(res)
    res = supprimer_espaces_bords(res)
    return res

# i. Wrapper Mise en forme
def normaliser_mise_en_forme(texte: str, options: dict = None) -> str:
    """Applique les traitements d'espaces selon les options."""
    if options is None: options = {}
    res = texte
    
    if options.get("remplacer_tabulations"):
        res = remplacer_tabulations(res)
    if options.get("normaliser_retours_ligne"):
        res = normaliser_retours_ligne(res)
    if options.get("supprimer_espaces_multiples"):
        res = supprimer_espaces_multiples(res)
    if options.get("supprimer_espaces_avant_ponctuation"):
        res = supprimer_espaces_avant_ponctuation(res)
    if options.get("ajouter_espace_apres_ponctuation"):
        res = ajouter_espace_apres_ponctuation(res)
    if options.get("supprimer_lignes_vides"):
        res = supprimer_lignes_vides(res)
    
    # Toujours faire un strip final
    return res.strip()


### 📝 Énoncé des fonctions de Vérification

# **Objectif :** S'assurer que le nettoyage n'a pas détruit le document.

# a. Détecter texte vide
def detecter_texte_vide(texte: str) -> bool:
    """Vérifie si le document est vide ou ne contient que des espaces."""
    return not texte or not texte.strip()

# b. Comparer longueurs
def comparer_longueurs_texte(texte_avant: str, texte_apres: str, seuil: float = 0.3) -> bool:
    """
    Renvoie True (Alerte) si len(apres) < 30% de len(avant).
    Signifie une perte excessive d'information.
    """
    len_avant = len(texte_avant)
    len_apres = len(texte_apres)
    
    if len_avant == 0: return False # Pas d'alerte si vide au départ
    
    ratio = len_apres / len_avant
    return ratio < seuil

# c. Signaler textes vides dans un dossier
def signaler_textes_vides(base: Path) -> list:
    """Parcourt le corpus et liste les fichiers qui sont vides."""
    vides = []
    for fichier in base.rglob('*'):
        if fichier.is_file():
            try:
                content = fichier.read_text(encoding='utf-8')
                if detecter_texte_vide(content):
                    vides.append(str(fichier))
            except Exception:
                pass # Ignorer erreurs lecture binaire etc.
    return vides
def test_mise_en_forme_et_verification():
    """Teste les fonctions des blocs 9 (Espaces) et 10 (Vérification)."""
    
    # --- Test Mise en Forme ---
    # Cas : Tabulation + Sauts ligne multiples + Espace avant virgule + Manque espace après virgule
    raw = "Nom\tPrénom\n\n   Bonjour ,comment ça va ?"
    
    opts = {
        "remplacer_tabulations": True,
        "supprimer_lignes_vides": True,
        "supprimer_espaces_avant_ponctuation": True,
        "ajouter_espace_apres_ponctuation": True,
        "supprimer_espaces_multiples": True
    }
    
    # Résultat attendu : "Nom Prénom\nBonjour, comment ça va ?"
    res = normaliser_mise_en_forme(raw, opts)
    
    assert "\t" not in res
    assert "  " not in res
    assert "Bonjour," in res      # Espace avant supprimé
    assert "Bonjour, c" in res    # Espace après ajouté
    print("Test Mise en Forme : OK")

    # --- Test Vérification ---
    # 1. Texte vide
    assert detecter_texte_vide("   \t \n ") is True
    assert detecter_texte_vide("A") is False
    
    # 2. Comparaison longueur (Perte > 70% déclenche True)
    txt_long = "Ceci est un texte plutôt long."
    txt_court = "Court"
    assert comparer_longueurs_texte(txt_long, txt_court, seuil=0.3) is True
    
    # 3. Signalement fichiers vides
    creer_arborescence_test()
    (BASE_TEST_DIR / "vide.txt").touch() # Créer un fichier vide
    (BASE_TEST_DIR / "plein.txt").write_text("Texte", encoding="utf-8")
    
    vides = signaler_textes_vides(BASE_TEST_DIR)
    assert len(vides) == 1
    assert "vide.txt" in vides[0]
    
    print("Test Vérification : OK")
