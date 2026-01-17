import streamlit as st
import sys
import os
import io
from pathlib import Path
from contextlib import redirect_stdout

# Configuration de la page
st.set_page_config(page_title="TP1 - Exploration de Corpus", layout="wide")

# Gestion de l'import dynamique de TP1
# TP1.py est dans Interfac/
# L'app est dans Interfac/pages/
# On doit remonter d'un niveau pour importer TP1
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

try:
    import TP1
except ImportError as e:
    st.error(f"Erreur d'importation de TP1 : {e}")
    st.info("Assurez-vous que TP1.py est bien dans le dossier Interfac/")
    st.stop()

st.title("TP1 - Exploration et Analyse de Corpus")
st.markdown("""
Cette interface permet de tester les fonctions de base du TP1 : exploration de dossiers, 
comptage de fichiers, vérifications de cohérence et statistiques basiques.
""")

# --- Sidebar: Sélection du Corpus ---
st.sidebar.header("Configuration")

# Recherche automatique du dossier Textes
default_path = "Textes"
root_path = Path(__file__).resolve().parent.parent.parent # Remonte à la racine du workspace
potential_textes = root_path / "Textes"
if potential_textes.exists():
    default_path = str(potential_textes)

corpus_path_str = st.sidebar.text_input("Chemin du corpus à analyser", value=default_path)
corpus_path = Path(corpus_path_str)

status_container = st.sidebar.empty()

if corpus_path.exists() and corpus_path.is_dir():
    status_container.success(f"✅ Dossier trouvé : {corpus_path.name}")
else:
    status_container.error("❌ Dossier introuvable")
    st.warning("Veuillez entrer un chemin valide vers un dossier (ex: 'Textes').")
    st.stop()

# --- Helpers ---
def capture_output(func, *args, **kwargs):
    """Capture le print() d'une fonction pour l'afficher dans Streamlit"""
    f = io.StringIO()
    with redirect_stdout(f):
        res = func(*args, **kwargs)
    return f.getvalue(), res

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📂 Exploration", "📊 Statistiques", "🛠️ Vérifications"])

# --- Tab 1 : Exploration ---
with tab1:
    st.header("Exploration de l'arborescence")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Structure (Arbre)")
        st.caption("Fonction : `afficher_structure`")
        if st.button("Afficher la structure"):
            output, _ = capture_output(TP1.afficher_structure, corpus_path)
            st.text(output) # On utilise st.text pour préserver l'espacement monospace

    with col2:
        st.subheader("2. Liste des documents")
        st.caption("Fonction : `lister_document`")
        if st.button("Lister les fichiers (.txt)"):
            output, _ = capture_output(TP1.lister_document, corpus_path)
            st.text_area("Résultat", output, height=300)

    st.markdown("---")
    st.subheader("3. Sous-corpus (Dossiers de 1er niveau)")
    st.caption("Fonction : `compter_sous_corpus`")
    
    if st.button("Compter les sous-corpus"):
        # Cette fonction retourne un tuple (nb, liste) + fait un print
        output, res = capture_output(TP1.compter_sous_corpus, corpus_path)
        nb, noms = res
        
        st.metric("Nombre de sous-corpus", nb)
        st.write("Noms des dossiers :", noms)
        with st.expander("Voir la sortie console"):
            st.code(output)

# --- Tab 2 : Statistiques ---
with tab2:
    st.header("Statistiques du Corpus")
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    
    with col_stat1:
        st.subheader("Nombre de Documents")
        if st.button("Compter (Total)"):
            output, count = capture_output(TP1.compter_documents, corpus_path)
            st.metric("Total fichiers .txt", count)
            
    with col_stat2:
        st.subheader("Répartition Langues")
        if st.button("Compter (FR / EN)"):
            output, data = capture_output(TP1.compter_par_langue, corpus_path)
            st.json(data)
            st.bar_chart(data)
            
    with col_stat3:
        st.subheader("Estimations Étudiants")
        if st.button("Compter Étudiants"):
            output, count = capture_output(TP1.compter_etudiants, corpus_path)
            st.metric("Nombre d'étudiants uniques", count, help="Basé sur les noms de fichiers sans suffixe _fr/_en")

    st.markdown("---")
    st.subheader("Répartition détaillée par Sous-Corpus")
    st.caption("Fonction : `repartition_langue_par_sous_corpus`")
    if st.button("Générer Tableau Répartition"):
        output, _ = capture_output(TP1.repartition_langue_par_sous_corpus, corpus_path)
        st.text(output) # Affichage du tableau formaté texte (print)

# --- Tab 3 : Vérifications ---
with tab3:
    st.header("Contrôle Qualité")
    
    col_check1, col_check2 = st.columns(2)
    
    with col_check1:
        st.subheader("Extensions Incorrectes")
        st.write("Vérifie que tous les fichiers sont bien des `.txt`.")
        if st.button("Vérifier extensions"):
            output, problemes = capture_output(TP1.verifier_extensions, corpus_path)
            
            if not problemes:
                st.success("✅ Tous les fichiers ont l'extension .txt")
            else:
                st.error(f"❌ {len(problemes)} fichiers incorrects détectés")
                for p in problemes:
                    st.write(f"- `{p}`")
    
    with col_check2:
        st.subheader("Correspondance Langues")
        st.write("Vérifie que chaque étudiant a bien une paire FR/EN.")
        if st.button("Vérifier paires FR/EN"):
            output, anomalies = capture_output(TP1.verifier_correspondance_langues, corpus_path)
            
            if not anomalies:
                st.success("✅ Toutes les paires sont complètes")
            else:
                st.warning(f"⚠️ {len(anomalies)} anomalies détectées")
                for a in anomalies:
                    st.write(f"- {a}")

    st.markdown("---")
    st.subheader("Rapport Complet")
    if st.button("Générer Statistiques Structurelles"):
        st.info("Exécution de `statistiques_structure`...")
        output, _ = capture_output(TP1.statistiques_structure, corpus_path)
        st.text_area("Rapport Console", output, height=400)
