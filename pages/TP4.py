import streamlit as st
import TP5
import pandas as pd
import math

st.set_page_config(page_title="TP4 - Vectorisation Avancée", layout="wide")

st.title("TP4 - Vectorisation de Documents et Agrégation")
st.markdown("""
Cette interface permet de tester les fonctions du TP4 : Vectorisation de phrases, de documents (sac de mots global) et agrégation de vecteurs.
""")

# --- Sidebar: Vocabulaire & IDF ---
st.sidebar.header("Configuration")
default_vocab_txt = "chat, chien, dort, mange, maison, souris, fromage"
vocab_input = st.sidebar.text_area("Mots du vocabulaire (séparés par virgule)", value=default_vocab_txt)

# Construction du vocabulaire
mots_vocab = [m.strip().lower() for m in vocab_input.split(",") if m.strip()]
vocab_direct = {mot: i for i, mot in enumerate(mots_vocab)}
vocab_inverse = {i: mot for i, mot in enumerate(mots_vocab)}

st.sidebar.write("Vocabulaire :")
st.sidebar.json(vocab_direct)

# Configuration IDF (simplifiée)
st.sidebar.subheader("Poids IDF (pour TF-IDF)")
idf_params = {}
for mot in mots_vocab:
    val = st.sidebar.number_input(f"IDF pour '{mot}'", 0.0, 10.0, 1.0, key=f"idf_{mot}")
    idf_params[mot] = val

# --- Tabs ---
tab1, tab2, tab3 = st.tabs([
    "Vectoriser Phrase", 
    "Vectoriser Document (Mots)", 
    "Vectoriser Document (Agrégé)"
])

# --- Tab 1: Vectoriser Phrase ---
with tab1:
    st.subheader("Vectorisation d'une Phrase")
    
    phrase_input = st.text_input("Phrase", value="le chat mange le fromage")
    methode = st.selectbox("Méthode", ["tf", "tfidf", "binaire"], key="methode_phrase")
    
    if st.button("Calculer Vecteur Phrase"):
        # TP5.vectoriser_phrase attend une chaine ou une liste. Ici on passe la chaine.
        # Elle fera le split elle-même si c'est une chaine (grâce à notre fix).
        vec = TP5.vectoriser_phrase(phrase_input, vocab_direct, methode=methode, idf=idf_params)
        
        st.write("Vecteur résultant :")
        df_vec = pd.DataFrame([vec], columns=mots_vocab)
        st.dataframe(df_vec)

# --- Tab 2: Vectoriser Document (Mots) ---
with tab2:
    st.subheader("Vectorisation Document (Sac de Mots Global)")
    st.markdown("Considère tout le document comme une seule grande liste de mots.")
    
    doc_input = st.text_area("Document (une phrase par ligne)", 
                             value="le chat dort\nle chien mange", key="doc_mots")
    
    # Préparation du document (liste de listes de mots)
    lignes = [l.strip() for l in doc_input.split("\n") if l.strip()]
    document_structure = [l.split() for l in lignes]
    
    methode_doc = st.selectbox("Méthode", ["tf", "tfidf", "binaire"], key="methode_doc_mots")
    
    if st.button("Calculer Vecteur Document (Mots)"):
        vec = TP5.vectoriser_document_mots(document_structure, vocab_direct, methode=methode_doc, idf=idf_params)
        
        st.write("Vecteur résultant :")
        df_vec = pd.DataFrame([vec], columns=mots_vocab)
        st.dataframe(df_vec)

# --- Tab 3: Vectoriser Document (Agrégé) ---
with tab3:
    st.subheader("Vectorisation Document (Agrégation de Phrases)")
    st.markdown("Vectorise chaque phrase individuellement puis agrège les vecteurs.")
    
    doc_input_agg = st.text_area("Document (une phrase par ligne)", 
                                 value="le chat dort\nle chien mange", key="doc_agg")
    
    lignes_agg = [l.strip() for l in doc_input_agg.split("\n") if l.strip()]
    document_structure_agg = [l.split() for l in lignes_agg]
    
    col1, col2 = st.columns(2)
    with col1:
        methode_agg = st.selectbox("Méthode (par phrase)", ["tf", "tfidf", "binaire"], key="methode_agg")
    with col2:
        strategie = st.selectbox("Stratégie d'agrégation", ["moyenne", "max", "somme"])
        
    if st.button("Calculer Vecteur Agrégé"):
        vec = TP5.vectoriser_document_agrege(document_structure_agg, vocab_direct, 
                                             methode=methode_agg, strategie=strategie, idf=idf_params)
        
        st.write("Vecteur résultant :")
        df_vec = pd.DataFrame([vec], columns=mots_vocab)
        st.dataframe(df_vec)
        
        st.write("Détail par phrase (pour vérification) :")
        details = []
        for phrase in document_structure_agg:
            v = TP5.vectoriser_phrase(phrase, vocab_direct, methode=methode_agg, idf=idf_params)
            details.append(v)
        
        df_details = pd.DataFrame(details, columns=mots_vocab, index=[f"Phrase {i+1}" for i in range(len(details))])
        st.dataframe(df_details)
