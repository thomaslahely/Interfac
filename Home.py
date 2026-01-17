import streamlit as st

st.set_page_config(
    page_title="708 - Ateliers IA",
    layout="wide"
)

st.write("#Interface des TP")

st.markdown("""
Cette app regroupe l'ensemble des TP **INFO0708 Ateliers IA**.

### Navigation
Utilisez la **barre latérale ** à gauche pour accéder aux différents TPs :

- **TP1** : Traitement de texte et expressions régulières
- **TP2** : Correction orthographique et distances
- **TP3** : Modèles de langage n-grammes
- **TP4** : Classification de textes
- **TP5** : Extraction d'information
- **TP6** : Word Embeddings (Word2Vec / FastText)

""")
