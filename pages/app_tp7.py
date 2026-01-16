import streamlit as st
import sys
import os
from pathlib import Path

# Ajout dynamique pour trouver le module TP7 dans Interfac
# On suppose que l'app est lancée depuis la racine du workspace
if "Interfac" not in sys.path:
    sys.path.append(str(Path(__file__).parent / "Interfac"))

# Try importing TP7 from Interfac if it's a package, or adjusting path
try:
    from Interfac import TP7
except ImportError:
    # Fallback: maybe we are just needing to add it to path differently or create __init__.py
    sys.path.append("Interfac")
    import TP7

import re
import string
import pandas as pd

st.set_page_config(page_title="TP7 - Embeddings & Similarité", layout="wide")

st.title("TP7: Word Embeddings (Word2Vec / FastText)")

st.sidebar.header("Configuration")

# 1. Chargement du Corpus
st.header("1. Corpus d'entraînement")

corpus_text = ""
uploaded_file = st.sidebar.file_uploader("Charger un corpus (texte brut)", type=["txt"])

default_text = """Le chat mange la souris.
Le chien aboie après le facteur.
La souris mange du fromage.
Le facteur apporte le courrier.
Les animaux sont nos amis.
L'intelligence artificielle est fascinante.
Le machine learning utilise des statistiques.
Java et Python sont des langages de programmation.
"""

if uploaded_file:
    stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    corpus_text = stringio.read()
else:
    corpus_text = st.text_area("Ou utilisez ce texte par défaut (modifiable):", value=default_text, height=150)

# Tokenisation simple pour le TP
def simple_tokenizer(text):
    # Enlève ponctuation et met en minuscule
    text = text.lower()
    # Garde lettres et espaces
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    sentences = text.split('\n')
    tokenized_corpus = []
    for s in sentences:
        tokens = s.split()
        if tokens:
            tokenized_corpus.append(tokens)
    return tokenized_corpus

tokenized_corpus = simple_tokenizer(corpus_text)

st.write(f"**Nombre de phrases détectées :** {len(tokenized_corpus)}")
if st.checkbox("Voir les premières phrases tokenisées"):
    st.write(tokenized_corpus[:5])


# 2. Entraînement du Modèle
st.header("2. Entraînement des Embeddings")

model_type = st.radio("Type de modèle", ["Word2Vec", "FastText"])

col1, col2, col3 = st.columns(3)
with col1:
    vector_size = st.number_input("Taille vecteur", min_value=2, max_value=300, value=10) # Petit par défaut pour petit corpus
with col2:
    window = st.number_input("Fenêtre (Window)", min_value=1, max_value=10, value=2)
with col3:
    min_count = st.number_input("Min Count", min_value=1, max_value=5, value=1)

if st.button("Entraîner le modèle"):
    with st.spinner("Entraînement en cours..."):
        if model_type == "Word2Vec":
            model = TP7.entrainer_modele_word2vec(tokenized_corpus, taille_vecteur=vector_size, fenetre=window, min_count=min_count, epochs=50)
        else:
            model = TP7.entrainer_modele_fasttext(tokenized_corpus, taille_vecteur=vector_size, fenetre=window, min_count=min_count, epochs=50)
        
    st.success(f"Modèle {model_type} entraîné avec succès !")
    # Sauvegarde en session state pour réutilisation
    st.session_state['model'] = model
    st.session_state['model_type'] = model_type

if 'model' in st.session_state:
    model = st.session_state['model']
    
    # 3. Explorer les vecteurs
    st.header("3. Explorer les vecteurs de mots")
    word_to_check = st.text_input("Entrez un mot pour voir ses voisins :", "chat")
    
    if word_to_check:
        try:
            # On utilise directement les méthodes gensim via l'objet model
            # Mais TP7 a aussi des wrappers s'il faut, on va utiliser model.wv.most_similar directement c'est plus visuel
            if word_to_check in model.wv:
                similars = model.wv.most_similar(word_to_check)
                st.write(f"Mots les plus proches de **{word_to_check}** :")
                df_sim = pd.DataFrame(similars, columns=["Mot", "Similarité"])
                st.table(df_sim)
            else:
                st.warning(f"Le mot '{word_to_check}' n'est pas dans le vocabulaire.")
                if st.session_state['model_type'] == "FastText":
                    st.info("Avec FastText, on peut quand même avoir un vecteur (hors vocabulaire).")
                    vec = TP7.get_vecteur_mot(word_to_check, model, strategie_oov='fasttext')
                    if vec is not None:
                        st.write("Vecteur généré (premières dimensions) :", vec[:5])
                    
        except Exception as e:
            st.error(f"Erreur : {e}")

    # 4. Similarité de phrases
    st.header("4. Similarité de phrases")
    p1 = st.text_input("Phrase 1", "le chat mange")
    p2 = st.text_input("Phrase 2", "le chien mange")
    
    tok_p1 = simple_tokenizer(p1)[0] if simple_tokenizer(p1) else []
    tok_p2 = simple_tokenizer(p2)[0] if simple_tokenizer(p2) else []
    
    if st.button("Calculer similarité"):
        if tok_p1 and tok_p2:
            sim = TP7.similarite_phrases(tok_p1, tok_p2, model, strategie_agregation='moyenne', strategie_oov='ignore')
            st.metric("Similarité Cosinus", f"{sim:.4f}")
        else:
            st.warning("Veuillez entrer des phrases valides.")

    # 5. Recherche Sémantique
    st.header("5. Recherche dans le corpus")
    query = st.text_input("Requête (phrase à chercher)", "animal domestique")
    
    if st.button("Chercher phrases similaires"):
        tok_query = simple_tokenizer(query)[0] if simple_tokenizer(query) else []
        if tok_query:
            results = TP7.top_k_phrases_similaires(tok_query, tokenized_corpus, model, k=3)
            st.write("Phrases les plus proches :")
            for idx, score in results:
                st.write(f"- **Score {score:.4f}** : {' '.join(tokenized_corpus[idx])}")
        else:
             st.warning("Requête vide.")

