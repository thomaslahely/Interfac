import streamlit as st

st.set_page_config(
    page_title="Atelier IA",
    layout="wide"
)

st.write("# Bienvenue sur l'interface")

st.markdown("""
Cette application regroupe l'ensemble des TP du module **INFO0708 Ateliers IA**.

### Navigation
Utilisez la **barre latérale** à gauche pour accéder aux différents TPs :

- **TP2** : Traitement de texte et expressions régulières
- **TP3** : Correction orthographique et distances
- **TP4** : Modèles de langage n-grammes
- **TP5** : Classification de textes
- **TP6** : Extraction d'information
- **TP7** : Word Embeddings (Word2Vec / FastText)


""")
