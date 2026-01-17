from contextlib import redirect_stdout
import io
from pathlib import Path
import shutil
from unittest.mock import patch
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_TEST_DIR = Path("corpus_de_test_temporaire")

def creer_arborescence_test():
    """
    Crée une arborescence de dossiers et fichiers fictive 
    pour les besoins des tests.
    """
    # S'assurer que tout est propre avant de commencer
    if BASE_TEST_DIR.exists():
        shutil.rmtree(BASE_TEST_DIR)
    
    BASE_TEST_DIR.mkdir()

    # UFR_LIMA (anomalies)
    (BASE_TEST_DIR / "UFR_LIMA").mkdir()
    (BASE_TEST_DIR / "UFR_LIMA" / "etudiant1_fr.txt").touch()
    (BASE_TEST_DIR / "UFR_LIMA" / "etudiant1_en.txt").touch() # Paire OK
    (BASE_TEST_DIR / "UFR_LIMA" / "etudiant2_fr.txt").touch() # Manque _en
    (BASE_TEST_DIR / "UFR_LIMA" / "niveau2").mkdir()
    (BASE_TEST_DIR / "UFR_LIMA" / "niveau2" / "etudiant4_en.txt").touch() # Manque _fr

    # UFR_INFO (anomalies)
    (BASE_TEST_DIR / "UFR_INFO").mkdir()
    (BASE_TEST_DIR / "UFR_INFO" / "etudiant3_fr.txt").touch() # Manque _en
    (BASE_TEST_DIR / "UFR_INFO" / "README.md").touch() # Extension incorrecte

    # UFR_VIDE
    (BASE_TEST_DIR / "UFR_VIDE").mkdir()
    
    # PERFECT_UFR (OK)
    (BASE_TEST_DIR / "PERFECT_UFR").mkdir()
    (BASE_TEST_DIR / "PERFECT_UFR" / "etudiant_ok_fr.txt").touch()
    (BASE_TEST_DIR / "PERFECT_UFR" / "etudiant_ok_en.txt").touch()

def nettoyer_arborescence_test():
    """Supprime l'arborescence de test."""
    if BASE_TEST_DIR.exists():
        shutil.rmtree(BASE_TEST_DIR)
"""

### 📝 Énoncé de la fonction 1

**Titre de la fonction :**  
> choisir_document() 

**Consigne :**  
*  cette fonction ouvre une boîte de dialogue
permettant à l’utilisateur de sélectionner un fichier texte dans son
ordinateur, puis retourne le chemin complet du document choisi.
"""

def choisir_document():
    """
    Description :
    Cette fonction ouvre une boîte de dialogue
    permettant à l’utilisateur de sélectionner un fichier texte dans son
    ordinateur, puis retourne le chemin complet du document choisi.
    Paramètres :
        Aucun
    Retour :
        str, chemin complet du document choisi
    """
    dossier =Path("Textes/UFR")
    fichier = input ("donnez un fichier texte à prendre \n")
    langue  = input("fr ou en ? \n")
    chemin = dossier /( fichier+"_"+langue+".txt")
    if chemin.is_file():
        print(f"le fichier existe")
    else :
        print(f"le fichier n'existe pas")
    return chemin

"""### 🔍 Explication des tests unitaires — fonction 1

Décris ici la stratégie de test que tu vas suivre :

1. **Cas 1 :**L'utilisateur entre un nom de fichier qui existe  
2. **Cas 2 :**L'utilisateur entre un nom de fichier qui n'existe pas  

Nous utilisons  :
- soit la bibliothèque `unittest` pour structurer des tests plus complets 
avec patch pour simuler les entrées utilisateur.

"""

def test_choisir_document():

    # --- Cas 1 : L'utilisateur entre un nom de fichier qui existe ---
    dossier_test = Path("Textes/UFR")
    dossier_test.mkdir(parents=True, exist_ok=True)
    fichier_test = dossier_test / "test_fr.txt"
    fichier_test.touch()  
    with patch('builtins.input', side_effect=['test', 'fr'])as mock_input:
        chemin_attendu = Path("Textes/UFR/test_fr.txt")
        chemin_obtenu = choisir_document()
        
        assert chemin_obtenu == chemin_attendu
        # On vérifie que le fichier existe.
        assert  chemin_obtenu.is_file()
    fichier_test.unlink()

    # --- Cas 2 : L'utilisateur entre un nom de fichier qui n'existe pas ---
    with patch('builtins.input', side_effect=['fichier_inexistant', 'en']) as mock_input:
        chemin_attendu = Path("Textes/UFR/fichier_inexistant_en.txt")
        chemin_obtenu = choisir_document()
        
        assert chemin_obtenu == chemin_attendu
        # On vérifie que le fichier n'existe pas.
        assert not chemin_obtenu.is_file()

    print("✅ Tous les tests unitaires pour choisir_document sont passés avec succès !")

# Exécution des tests
test_choisir_document()

"""---

### 📝 Énoncé de la fonction 2

**Titre de la fonction :**  
> lister_document(chemin)

**Consigne :**  
*  Cette fonction liste tous les fichiers texte contenus dans un répertoire donné.
*   Elle prend en paramètre le chemin du répertoire à explorer et affiche les noms des fichiers texte trouvés.
"""
    
def lister_document(chemin):
    """
    Description :
        Cette fonction liste tous les fichiers texte contenus dans un répertoire donné.
    Paramètres :
        chemin -- Path, le chemin du répertoire à explorer
    Retour :
        None
    """
    for fichier in chemin.rglob("*.txt"):
        if fichier.is_file():
            print(fichier)

"""### 🔍 Explication des tests unitaires — fonction 2

Décris ici la stratégie de test que tu vas suivre :

1. **Cas 1 :** les entrées habituelles où la fonction doit fonctionner correctement.  
2. **Cas 2 :** des cas limites, comme un répertoire vide ou un répertoire sans fichiers texte. 


"""

def test_lister_document():
    # 1. Préparation
    creer_arborescence_test()
    
    # 2. Exécution et Capture
    f = io.StringIO()
    with redirect_stdout(f):
        lister_document(BASE_TEST_DIR)
    output = f.getvalue()

    # 3. Vérifications
    # Doit trouver les 5 fichiers .txt
    assert str(BASE_TEST_DIR / "UFR_LIMA" / "etudiant1_fr.txt") in output
    assert str(BASE_TEST_DIR / "UFR_LIMA" / "etudiant1_en.txt") in output
    assert str(BASE_TEST_DIR / "UFR_LIMA" / "etudiant2_fr.txt") in output
    assert str(BASE_TEST_DIR / "UFR_INFO" / "etudiant3_fr.txt") in output
    assert str(BASE_TEST_DIR / "UFR_LIMA" / "niveau2" / "etudiant4_en.txt") in output # Fichier récursif
    
    # Ne doit pas trouver le fichier .md
    assert "README.md" not in output
    
    # 4. Nettoyage
    nettoyer_arborescence_test()
    print("✅ Tous les tests unitaires pour lister_document sont passés avec succès !")

# Exécution du test 2
test_lister_document()

"""---

### 📝 Énoncé de la fonction 3

**Titre de la fonction :**  
> explorer_corpus(chemin:  Path)

**Consigne :**  
    *  cette fonction  parcourt récursivement l’ensemble de la base, affiche les sous-répertoires et le
    nombre de fichiers trouvés dans chacun.
*   Elle prend en paramètre le chemin du répertoire à explorer .
"""
    
def explorer_corpus(chemin:Path):
    """
    Description :
        Cette fonction liste tous les fichiers texte contenus dans un répertoire donné.
    Paramètres :
        chemin -- Path, le chemin du répertoire à explorer
    Retour :
        None
    """
    for fichier in chemin.rglob("*"):
        compteur = 0
        if fichier.is_dir():
            print(fichier)

"""### 🔍 Explication des tests unitaires — Exercice 2

Décris ici la stratégie de test que tu vas suivre :

1. **Cas 1 :** les entrées habituelles où la fonction doit fonctionner correctement.  
2. **Cas 2 :** des cas limites, comme un répertoire vide ou un répertoire sans fichiers texte. 


"""


def test_explorer_corpus():
    # 1. Préparation
    creer_arborescence_test()
    
    # 2. Exécution et Capture
    f = io.StringIO()
    with redirect_stdout(f):
        explorer_corpus(BASE_TEST_DIR)
    output = f.getvalue()
    
    # 3. Vérifications
    # Doit trouver les 4 répertoires
    assert str(BASE_TEST_DIR / "UFR_LIMA") in output
    assert str(BASE_TEST_DIR / "UFR_INFO") in output
    assert str(BASE_TEST_DIR / "UFR_VIDE") in output
    assert str(BASE_TEST_DIR / "UFR_LIMA" / "niveau2") in output # Répertoire récursif
    
    # 4. Nettoyage
    nettoyer_arborescence_test()
    print("✅ Tous les tests unitaires pour explorer_corpus sont passés avec succès !")

# Exécution du test 3
test_explorer_corpus()




"""---

### 📝 Énoncé de l’exercice 4

**Titre de l’exercice :**  
> afficher_structure(chemin_base: Path)

**Consigne :**  
    *  Cette fonction affiche 
    l'arborescence complète du chemin_base sous une forme visuelle (arbre).
*   Elle prend en paramètre le chemin du répertoire à explorer .
"""
    
def afficher_structure(chemin_base:Path):
    if not chemin_base.is_dir():
        print(f"Le chemin {chemin_base} n'est pas un répertoire valide.")
        return
    print(chemin_base)
    afficher_rec(chemin_base,"")

def afficher_rec(chemin:Path, prefixe:str):
    items = sorted([item for item in chemin.iterdir() if not item.name.startswith(".")], key=lambda x: x.name)
    
    for i, item in enumerate(items):
        est_dernier = (i == len(items) - 1)
        if est_dernier:
            print(f"{prefixe}└── {item.name}")
            nouveau_prefixe = prefixe + "    "
        else:
            print(f"{prefixe}├── {item.name}")
            nouveau_prefixe = prefixe + "│   "
        if item.is_dir():
            afficher_rec(item, nouveau_prefixe)

"""### 🔍 Explication des tests unitaires — Exercice 4

Cas 1 : Lancer la fonction sur le corpus de test. On vérifie que la structure de l'arbre (les préfixes ├──, └──, │ ) est correcte et que tous les fichiers/dossiers sont listés.

Cas 2 : Lancer la fonction sur un chemin qui n'existe pas. On vérifie que le message d'erreur est bien affiché.

"""


def test_afficher_structure():
    # 1. Préparation
    creer_arborescence_test()
    
    # --- Cas 1 : Chemin valide ---
    f = io.StringIO()
    with redirect_stdout(f):
        afficher_structure(BASE_TEST_DIR)
    output = f.getvalue()
    
    # On définit la sortie exacte attendue (grâce au tri alphabétique)
    attendus = [
        str(BASE_TEST_DIR),
        "├── PERFECT_UFR",
        "│   ├── etudiant_ok_en.txt",
        "│   └── etudiant_ok_fr.txt",
        "├── UFR_INFO",
        "│   ├── README.md",
        "│   └── etudiant3_fr.txt",
        "├── UFR_LIMA",
        "│   ├── etudiant1_en.txt",
        "│   ├── etudiant1_fr.txt",
        "│   ├── etudiant2_fr.txt",
        "│   └── niveau2",
        "│       └── etudiant4_en.txt",
        "└── UFR_VIDE"
    ]
    
    # On compare ligne par ligne
    output_lines = [line.strip() for line in output.strip().split('\n')]
    
    assert len(output_lines) == len(attendus)
    for i, ligne_attendue in enumerate(attendus):
        assert ligne_attendue == output_lines[i], f"Erreur à la ligne {i+1}: Attendu '{ligne_attendue}', Reçu '{output_lines[i]}'"

    # --- Cas 2 : Chemin invalide ---
    f_err = io.StringIO()
    chemin_invalide = Path("dossier/qui/nexiste/pas")
    with redirect_stdout(f_err):
        afficher_structure(chemin_invalide)
    output_err = f_err.getvalue()
    
    assert f"Le chemin {chemin_invalide} n'est pas un répertoire valide." in output_err
    
    # 4. Nettoyage
    nettoyer_arborescence_test()
    
    
    print("✅ Tous les tests unitaires pour afficher_structure sont passés avec succès !")

# Exécution du test 4
test_afficher_structure()



"""---

### 📝 Énoncé de l’exercice 5

**Titre de l’exercice :**  
> compter_sous_corpus(base: Path)

**Consigne :**  
    *  Cette fonction compte les sous-répertoires directs (au premier niveau) du chemin base
*   Elle affiche le nombre et la liste des noms, puis retourne un tuple : (nombre, liste_des_noms).
"""
    
def compter_sous_corpus(base:Path):
    sous_corpus = [d for d in base.iterdir() if d.is_dir()]
    print (f"Il y a {len(sous_corpus)} sous-corpus :")
    print([d.name for d in sous_corpus])
    return len(sous_corpus), [d.name for d in sous_corpus]

"""### 🔍 Explication des tests unitaires — Exercice 4

Cas 1 : On lance la fonction sur notre corpus de test .
"""

def test_compter_sous_corpus():
    # 1. Préparation
    creer_arborescence_test()
    
    # 2. Exécution et Capture (on cache les prints)
    f = io.StringIO()
    with redirect_stdout(f):
        count, names = compter_sous_corpus(BASE_TEST_DIR)

    # 3. Vérifications
    # Il y a 4 dossiers directs
    assert count == 4
    
    # On trie les listes pour la comparaison
    attendus = ["UFR_INFO", "UFR_LIMA", "UFR_VIDE", "PERFECT_UFR"]
    assert sorted(names) == sorted(attendus)
    
    # 4. Nettoyage
    nettoyer_arborescence_test()
    print("✅ Tous les tests unitaires pour compter_sous_corpus sont passés avec succès !")

# Exécution du test 5
test_compter_sous_corpus()

"""---

### 📝 Énoncé de l’exercice 6

**Titre de l’exercice :**  
> compter_documents(base: Path)

**Consigne :**  
    *  Cette fonction compte et retourne le nombre total de fichiers .txt trouvés dans base, de manière récursive..
"""
    
def compter_documents(base:Path):
    # La fonction calcule la liste deux fois.
    print(len(list(base.rglob("*.txt"))))
    return len(list(base.rglob("*.txt")))

"""### 🔍 Explication des tests unitaires — Exercice 6

Cas 1 : Lancer la fonction sur le corpus de test.
"""

def test_compter_documents():
    # 1. Préparation
    creer_arborescence_test()
    
    # 2. Exécution et Capture (on cache le print)
    f = io.StringIO()
    with redirect_stdout(f):
        count = compter_documents(BASE_TEST_DIR)
        
    # 3. Vérifications
    # 7 fichiers .txt au total dans l'arborescence
    assert count == 7 
    
    # 4. Nettoyage
    nettoyer_arborescence_test()
    print("✅ Tous les tests unitaires pour compter_documents sont passés avec succès !")

# Exécution du test 6
test_compter_documents()


"""---

### 📝 Énoncé de l’exercice 7

**Titre de l’exercice :**  
> compter_par_langue(base: Path)

**Consigne :**  
    *  Cette fonction compte récursivement le nombre de fichiers _fr.txt et _en.txt dans base.
    *Elle retourne un dictionnaire : {"fr": nb_fr, "en": nb_en}.
"""
    
def compter_par_langue(base: Path):
    """Calcule le nombre de documents par langue."""
    nb_fr = len(list(base.rglob("*_fr.txt")))
    nb_en = len(list(base.rglob("*_en.txt")))
    print(f"Nombre de documents en français : {nb_fr}")
    print(f"Nombre de documents en anglais : {nb_en}")
    return {"fr": nb_fr, "en": nb_en}

"""### 🔍 Explication des tests unitaires — Exercice 7

Cas 1 : Lancer sur l'ensemble du corpus de test (BASE_TEST_DIR).
Cas 2 : Lancer sur un sous-dossier spécifique (UFR_LIMA) pour vérifier que le comptage récursif (rglob) fonctionne aussi à ce niveau.
"""

def test_compter_par_langue():
    # 1. Préparation
    creer_arborescence_test()
    
    # --- Cas 1 : Corpus complet ---
    f = io.StringIO()
    with redirect_stdout(f):
        repartition = compter_par_langue(BASE_TEST_DIR)
        
    # fr: etudiant1_fr, etudiant2_fr, etudiant3_fr, etudiant_ok_fr (total 4)
    # en: etudiant1_en, etudiant4_en, etudiant_ok_en (total 3)
    assert repartition == {"fr": 4, "en": 3}

    # --- Cas 2 : Sous-corpus UFR_LIMA (doit être récursif) ---
    f_lima = io.StringIO()
    with redirect_stdout(f_lima):
        repartition_lima = compter_par_langue(BASE_TEST_DIR / "UFR_LIMA")
    
    # fr: etudiant1_fr, etudiant2_fr (total 2)
    # en: etudiant1_en, etudiant4_en (dans niveau2) (total 2)
    assert repartition_lima == {"fr": 2, "en": 2}
    
    # 4. Nettoyage
    nettoyer_arborescence_test()
    print("✅ Tous les tests unitaires pour compter_par_langue sont passés avec succès !")

# Exécution du test 7
test_compter_par_langue()

"""---

### 📝 Énoncé de l’exercice 8

**Titre de l’exercice :**  
> repartition_langue_par_sous_corpus(base: Path)

**Consigne :**  
    * Cette fonction affiche (via print) un tableau de la répartition FR/EN pour chaque sous-corpus direct (non récursif).
    *Elle utilise les fonctions compter_sous_corpus et compter_par_langue pour y parvenir.
"""
    
def repartition_langue_par_sous_corpus(base: Path):
    """Affiche la répartition FR/EN pour chaque sous-corpus."""
    
    # On capture la sortie de la fonction helper (exercice 5)
    f_helper = io.StringIO()
    with redirect_stdout(f_helper):
        _, sous_corpus_noms = compter_sous_corpus(base)
    
    print("\n--- Répartition par langue et sous-corpus ---")
    print(f"{'Sous-corpus':<15} | {'FR':>5} | {'EN':>5}")
    print("-" * 30)
    

    for nom in sorted(sous_corpus_noms):
        chemin_sc = base / nom

        f_helper_inner = io.StringIO()
        with redirect_stdout(f_helper_inner):
            repartition = compter_par_langue(chemin_sc)
        
            
        print(f"{nom:<15} | {repartition['fr']:>5} | {repartition['en']:>5}")

"""### 🔍 Explication des tests unitaires — Exercice 8

Cas 1 : Lancer la fonction sur le corpus de test.
"""

def test_repartition_langue_par_sous_corpus():
    # 1. Préparation
    creer_arborescence_test()
    
    # 2. Exécution et Capture
    f = io.StringIO()
    with redirect_stdout(f):
        repartition_langue_par_sous_corpus(BASE_TEST_DIR)
    output = f.getvalue()
    
    # 3. Vérifications
    # On vérifie la présence des lignes du tableau (triées par ordre alpha)
    assert "UFR_INFO        |     1 |     0" in output
    assert "UFR_LIMA        |     2 |     2" in output
    assert "UFR_VIDE        |     0 |     0" in output
    assert "PERFECT_UFR     |     1 |     1" in output
    
    # On vérifie l'en-tête
    assert "Sous-corpus" in output
    assert "FR" in output and "EN" in output
    assert ("-" * 30) in output
    
    # 4. Nettoyage
    nettoyer_arborescence_test()
    print("✅ Tous les tests unitaires pour repartition_langue_par_sous_corpus sont passés avec succès !")

# Exécution du test 8
test_repartition_langue_par_sous_corpus()


"""---

### 📝 Énoncé de l’exercice 9

**Titre de l’exercice :**  
> compter_etudiants(base: Path)

**Consigne :**  
    * Cette fonction estime le nombre d'"étudiants" uniques en comptant récursivement les noms de fichiers de base (les "stems" sans _fr ou _en).
    *Elle utilise les fonctions compter_sous_corpus et compter_par_langue pour y parvenir.
"""
    
def compter_etudiants(base: Path):
    """Estime le nombre d'étudiants (couples de fichiers FR/EN)."""
    noms_base = set()
    for f in base.rglob("*.txt"):
        if f.stem.endswith("_fr") or f.stem.endswith("_en"):
            noms_base.add(f.stem[:-3]) # Ex: "etudiant1_fr" -> "etudiant1"
    print(f"Nombre estimé d'étudiants : {len(noms_base)}")
    return len(noms_base)

"""### 🔍 Explication des tests unitaires — Exercice 9

Cas 1 : Lancer la fonction sur le corpus de test.
"""
def test_compter_etudiants():
    # 1. Préparation
    creer_arborescence_test()
    
    # 2. Exécution et Capture
    f = io.StringIO()
    with redirect_stdout(f):
        count = compter_etudiants(BASE_TEST_DIR)
        
    # 3. Vérifications
    # Stems attendus : etudiant1, etudiant2, etudiant3, etudiant4
    assert count == 5
    
    # 4. Nettoyage
    nettoyer_arborescence_test()
    print("✅ Tous les tests unitaires pour compter_etudiants sont passés avec succès !")

# Exécution du test 9
test_compter_etudiants()


"""---

### 📝 Énoncé de l’exercice 10

**Titre de l’exercice :**  
> verifier_extensions(base: Path)

**Consigne :**  
    * Cette fonction vérifie récursivement que tous les fichiers dans base ont l'extension .txt.
"""
def verifier_extensions(base:Path):
    problemes = []
    for f in base.rglob("*"):
        if f.is_file() and  f.suffix != ".txt":
            problemes.append(f)
    if not problemes:
        print("tout les fichiers sont bons")
    else:
        print("fichiers avec des extensions incorrectes :")
        for f in problemes:
            print(f)
    return problemes
"""### 🔍 Explication des tests unitaires — Exercice 10

Cas 1 : La liste retournée (problemes) ne doit pas être vide. 
Elle doit contenir 1 élément, et cet élément doit être le chemin vers README.md. 
On vérifie aussi que le print affiche le message d'erreur.

Cas 2 : La liste retournée (problemes_ok) doit être vide. 
On vérifie que le print affiche le message de succès ("tout les fichiers sont bons").
"""

def test_verifier_extensions():
    # 1. Préparation
    creer_arborescence_test()
    
    # --- Cas 1 : Fichier incorrect (README.md) ---
    f_err = io.StringIO()
    with redirect_stdout(f_err):
        problemes = verifier_extensions(BASE_TEST_DIR)
    output_err = f_err.getvalue()

    # Vérifications de la valeur de retour
    assert len(problemes) == 1
    assert problemes[0].name == "README.md"
    assert problemes[0] == BASE_TEST_DIR / "UFR_INFO" / "README.md"
    
    # Vérifications de la sortie console
    assert "extensions incorrectes" in output_err
    assert str(problemes[0]) in output_err # Le chemin doit être affiché

    
    # --- Cas 2 : Dossier "propre" (UFR_LIMA) ---
    f_ok = io.StringIO()
    with redirect_stdout(f_ok):
        # On teste sur un sous-dossier qui ne contient que des .txt
        problemes_ok = verifier_extensions(BASE_TEST_DIR / "UFR_LIMA")
    output_ok = f_ok.getvalue()

    # Vérifications de la valeur de retour
    assert len(problemes_ok) == 0
    
    # Vérifications de la sortie console
    assert "tout les fichiers sont bons" in output_ok
    
    # 4. Nettoyage
    nettoyer_arborescence_test()
    print("✅ Tous les tests unitaires pour verifier_extensions sont passés avec succès !")

# Exécution du test 10
test_verifier_extensions()


"""---

### 📝 Énoncé de l’exercice 11

**Titre de l’exercice :**  
> verifier_correspondance_langues(base: Path)

**Consigne :**  
    * Cette fonction vérifie récursivement pour chaque "étudiant" 
    s'il possède bien à la fois un fichier _fr.txt et un fichier _en.txt.
"""
def verifier_correspondance_langues(base: Path):
    """Vérifie que chaque étudiant a un fichier _fr et _en."""
    etudiants = defaultdict(lambda: {"fr": False, "en": False})
    anomalies = []
    
    for f in base.rglob("*.txt"):
        if f.stem.endswith("_fr"):
            nom_base = f.stem[:-3]
            # La clé est le chemin complet du stem (sans extension)
            etudiants[f.parent / nom_base]["fr"] = True
        elif f.stem.endswith("_en"):
            nom_base = f.stem[:-3]
            etudiants[f.parent / nom_base]["en"] = True
            
    for nom, langues in etudiants.items():
        if not langues["fr"]:
            anomalies.append(f"Manquant : {nom}_fr.txt")
        if not langues["en"]:
            anomalies.append(f"Manquant : {nom}_en.txt")
            
    if not anomalies:
        print("Correspondance des langues : OK")
    else:
        print("Anomalies de correspondance des langues :")
        for a in anomalies:
            print(f"- {a}")
    return anomalies
    
"""### 🔍 Explication des tests unitaires — Exercice 11

Cas 1 : Lancer sur le corpus de test complet (BASE_TEST_DIR), qui contient plusieurs anomalies.

Cas 2 : Lancer sur le sous-dossier PERFECT_UFR, qui ne contient aucune anomalie.

"""

def test_verifier_correspondance_langues():
    # 1. Préparation
    creer_arborescence_test()
    
    # --- Cas 1 : Corpus avec anomalies (BASE_TEST_DIR) ---
    f_err = io.StringIO()
    with redirect_stdout(f_err):
        anomalies = verifier_correspondance_langues(BASE_TEST_DIR)
    output_err = f_err.getvalue()

    # Vérifications de la valeur de retour (la liste)
    assert len(anomalies) == 3
    
    # On utilise un set pour vérifier les anomalies sans se soucier de l'ordre
    anomalies_set = set(anomalies)
    
    attendues = {
        f"Manquant : {BASE_TEST_DIR / 'UFR_LIMA' / 'etudiant2'}_en.txt",
        f"Manquant : {BASE_TEST_DIR / 'UFR_INFO' / 'etudiant3'}_en.txt",
        f"Manquant : {BASE_TEST_DIR / 'UFR_LIMA' / 'niveau2' / 'etudiant4'}_fr.txt"
    }
    
    assert anomalies_set == attendues
    
    # Vérifications de la sortie console
    assert "Anomalies de correspondance des langues :" in output_err
    assert f"- Manquant : {BASE_TEST_DIR / 'UFR_LIMA' / 'etudiant2'}_en.txt" in output_err


    # --- Cas 2 : Corpus parfait (PERFECT_UFR) ---
    f_ok = io.StringIO()
    with redirect_stdout(f_ok):
        anomalies_ok = verifier_correspondance_langues(BASE_TEST_DIR / "PERFECT_UFR")
    output_ok = f_ok.getvalue()
    
    # Vérifications de la valeur de retour
    assert len(anomalies_ok) == 0
    
    # Vérifications de la sortie console
    assert "Correspondance des langues : OK" in output_ok
    
    # 4. Nettoyage
    nettoyer_arborescence_test()
    print("✅ Tous les tests unitaires pour verifier_correspondance_langues sont passés avec succès !")

# Exécution du test 11
test_verifier_correspondance_langues()




"""---

### 📝 Énoncé de l’exercice 12

**Titre de l’exercice :**  
> statistiques_structure(base: Path)

**Consigne :**  
    * Elle appelle plusieurs autres fonctions d'analyse (comme compter_documents, verifier_extensions, etc.) 
    et compile leurs résultats dans un résumé final affiché (via print) sur la console.
"""
def statistiques_structure(base: Path):
    """Produit un rapport résumé de la structure du corpus."""
    nb_sc, noms_sc = compter_sous_corpus(base) 
    nb_docs = compter_documents(base)
    repartition_langues = compter_par_langue(base)
    nb_etudiants = compter_etudiants(base)
    anomalies_langues = verifier_correspondance_langues(base)
    anomalies_extensions = verifier_extensions(base)
    
    print("\n--- Rapport de structure du corpus ---")
    print(f"Nombre de sous-corpus : {nb_sc} ({', '.join(noms_sc)})")
    print(f"Nombre total de fichiers : {nb_docs}")
    print(f"Fichiers français : {repartition_langues['fr']}")
    print(f"Fichiers anglais : {repartition_langues['en']}")
    print(f"Nombre d'étudiants identifiés : {nb_etudiants}")
    print(f"Nombre d'anomalies détectées : {len(anomalies_langues) + len(anomalies_extensions)}")
    print("------------------------------------")

"""### 🔍 Explication des tests unitaires — Exercice 12

Cas 1 : On lance la fonction sur notre corpus de test (BASE_TEST_DIR).

"""
def test_statistiques_structure():
    # 1. Préparation
    creer_arborescence_test()
    
    # 2. Exécution et Capture
    f = io.StringIO()
    with redirect_stdout(f):
        statistiques_structure(BASE_TEST_DIR)
    output = f.getvalue()

    # 3. Vérifications
    # On vérifie que les lignes spécifiques du rapport final sont présentes
    # et contiennent les bonnes données agrégées.
    
    rapport_lines = []
    capturing = False
    for line in output.splitlines():
        if "--- Rapport de structure du corpus ---" in line:
            capturing = True
        if capturing:
            rapport_lines.append(line)
            
    assert "--- Rapport de structure du corpus ---" in rapport_lines[0]
    # Noms (triés) : INFO, LIMA, PERFECT_UFR, UFR_VIDE
    assert "Nombre de sous-corpus : 4 (UFR_INFO, UFR_LIMA, PERFECT_UFR, UFR_VIDE)" in rapport_lines[1]
    assert "Nombre total de fichiers : 7" in rapport_lines[2]
    assert "Fichiers français : 4" in rapport_lines[3]
    assert "Fichiers anglais : 3" in rapport_lines[4]
    assert "Nombre d'étudiants identifiés : 5" in rapport_lines[5]
    assert "Nombre d'anomalies détectées : 4" in rapport_lines[6] # 3 langues + 1 extension
    assert "------------------------------------" in rapport_lines[7]
    
    # 4. Nettoyage
    nettoyer_arborescence_test()
    print("✅ Tous les tests unitaires pour statistiques_structure sont passés avec succès !")

# Exécution du test 12
test_statistiques_structure()







"""---

### 📝 Énoncé de l’exercice 13

**Titre de l’exercice :**  
> afficher_repartition_sous_corpus(base: Path, ax=None)

**Consigne :**  
    * Affiche un diagramme en barres (matplotlib) montrant le nombre total de documents 
    (.txt) pour chaque sous-corpus direct.
"""
def afficher_repartition_sous_corpus(base: Path, ax=None):
    """Affiche un diagramme en barres du nombre de documents par sous-corpus."""
    show_plot = ax is None
    if show_plot:
        fig, ax = plt.subplots()

    # Capture des prints des fonctions helpers
    f = io.StringIO()
    with redirect_stdout(f):
        _, noms_sc = compter_sous_corpus(base)
        comptes = [compter_documents(base / nom) for nom in noms_sc]
    
    ax.bar(noms_sc, comptes, color='skyblue')
    ax.set_title('Nombre de documents par sous-corpus')
    ax.set_ylabel('Nombre de documents')
    ax.set_xlabel('Sous-corpus')
    
    if show_plot:
        plt.tight_layout()
        plt.show()







"""---

### 📝 Énoncé de l’exercice 14

**Titre de l’exercice :**  
> afficher_repartition_langues(base: Path, ax=None)

**Consigne :**  
    * Affiche un diagramme circulaire (camembert) montrant la proportion globale de fichiers _fr.txt 
    par rapport aux _en.txt dans l'ensemble du corpus (récursif).
"""
def afficher_repartition_langues(base: Path, ax=None):
    """Affiche un camembert de la proportion des langues."""
    show_plot = ax is None
    if show_plot:
        fig, ax = plt.subplots()

    # Capture des prints des fonctions helpers
    f = io.StringIO()
    with redirect_stdout(f):
        repartition = compter_par_langue(base)
        
    labels = ['Français', 'Anglais']
    sizes = [repartition['fr'], repartition['en']]
    colors = ['#ff9999','#66b3ff']
    
    # Gère le cas où il n'y a aucun document
    if sum(sizes) == 0:
        sizes = [1, 1] # Pour éviter une erreur matplotlib
        labels = ['Aucun document FR', 'Aucun document EN']

    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Proportion des langues dans le corpus')
    ax.axis('equal')  # Assure que le camembert est un cercle.
    
    if show_plot:
        plt.tight_layout()
        plt.show()

"""---

### 📝 Énoncé de l’exercice 15

**Titre de l’exercice :**  
> afficher_repartition_langues_par_sous_corpus(base: Path, ax=None)

**Consigne :**  
    * Affiche un diagramme en barres groupées, 
    comparant le nombre de fichiers FR et EN côte à côte pour chaque sous-corpus direct.
"""
def afficher_repartition_langues_par_sous_corpus(base: Path, ax=None):
    """Génère un diagramme groupé FR/EN par sous-corpus."""
    show_plot = ax is None
    if show_plot:
        fig, ax = plt.subplots()

    # Capture des prints des fonctions helpers
    f = io.StringIO()
    with redirect_stdout(f):
        _, noms_sc = compter_sous_corpus(base)
        comptes_fr = [compter_par_langue(base / nom)['fr'] for nom in noms_sc]
        comptes_en = [compter_par_langue(base / nom)['en'] for nom in noms_sc]

    x = np.arange(len(noms_sc))  # positions des labels
    width = 0.35  # largeur des barres

    rects1 = ax.bar(x - width/2, comptes_fr, width, label='Français', color='#ff9999')
    rects2 = ax.bar(x + width/2, comptes_en, width, label='Anglais', color='#66b3ff')

    ax.set_ylabel('Nombre de documents')
    ax.set_title('Répartition FR/EN par sous-corpus')
    ax.set_xticks(x)
    ax.set_xticklabels(noms_sc)
    ax.legend()

    if show_plot:
        fig.tight_layout()
        plt.show()


"""---

### 📝 Énoncé de l’exercice 16

**Titre de l’exercice :**  
> tableau_de_bord_corpus(base: Path)

**Consigne :**  
    * Crée une figure matplotlib unique (un "tableau de bord") 
    composée de 4 sous-graphiques : les 3 visualisations précédentes et un résumé textuel des statistiques.
"""
def tableau_de_bord_corpus(base: Path):
    """Combine les visualisations dans un tableau de bord."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Tableau de bord du Corpus', fontsize=16)

    # Graphique 1: Répartition par sous-corpus
    afficher_repartition_sous_corpus(base, ax=axs[0, 0])

    # Graphique 2: Répartition des langues (camembert)
    afficher_repartition_langues(base, ax=axs[0, 1])

    # Graphique 3: Répartition groupée par sous-corpus
    afficher_repartition_langues_par_sous_corpus(base, ax=axs[1, 0])
    
    # Texte de résumé (capture des prints)
    f = io.StringIO()
    with redirect_stdout(f):
        nb_sc, _ = compter_sous_corpus(base)
        nb_docs = compter_documents(base)
        nb_etudiants = compter_etudiants(base)
        anomalies = len(verifier_correspondance_langues(base)) + len(verifier_extensions(base))
    
    rapport_texte = (
        f"--- Résumé Statistique ---\n\n"
        f"Nombre de sous-corpus : {nb_sc}\n"
        f"Nombre total de documents : {nb_docs}\n"
        f"Nombre d'étudiants : {nb_etudiants}\n"
        f"Anomalies détectées : {anomalies}"
    )
    
    axs[1, 1].text(0.5, 0.5, rapport_texte, ha='center', va='center', fontsize=12, bbox={"boxstyle": "round,pad=0.5", "facecolor": "wheat", "alpha": 0.5})
    axs[1, 1].axis('off') # Cacher les axes pour le texte

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

print("\n\n--- Lancement des visualisations (Exercices 13 à 16) ---")

creer_arborescence_test()

afficher_repartition_sous_corpus(BASE_TEST_DIR)
afficher_repartition_langues(BASE_TEST_DIR)
afficher_repartition_langues_par_sous_corpus(BASE_TEST_DIR)
tableau_de_bord_corpus(BASE_TEST_DIR)
nettoyer_arborescence_test()
print("\n--- Fin des visualisations ---")