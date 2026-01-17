import streamlit as st
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from gensim.models import Word2Vec, FastText
import App_TP6 as TP6
# --- Titre ---
st.title("TP6 - Word Embeddings & Expansion de Requête")
st.markdown("""
Cette interface explore les modèles vectoriels distribués (Embeddings) et l'expansion de requêtes.
1.  **Entraînement** : Word2Vec / FastText sur un corpus.
2.  **Exploration** : Voisins sémantiques, Analogies.
3.  **Recherche** : Comparaison de phrases et Expansion de requête.
""")

# --- Session State pour le modèle ---
if "model_w2v" not in st.session_state:
    st.session_state["model_w2v"] = None
if "corpus_train" not in st.session_state:
    # Corpus jouet par défaut
    st.session_state["corpus_train"] = [
        ["le", "chat", "mange", "la", "souris"],
        ["le", "chien", "aboie", "après", "le", "chat"],
        ["la", "souris", "mange", "du", "fromage"],
        ["le", "chat", "dort", "sur", "le", "canapé"],
        ["l", "intelligence", "artificielle", "est", "le", "futur"],
        ["le", "nlp", "traite", "le", "langage", "naturel"],
        ["les", "embeddings", "sont", "des", "vecteurs", "denses"],
        ["word2vec", "est", "un", "algorithme", "de", "plongement"],
        ["la", "reconnaissance", "vocale", "utilise", "ia"],
        ["le", "deep", "learning", "est", "une", "branche", "de", "ia"]
    ]

# --- Sidebar ---
st.sidebar.header("Configuration Entraînement")
algo = st.sidebar.radio("Algorithme", ["Word2Vec", "FastText"])
vector_size = st.sidebar.slider("Dimension (vector_size)", 10, 300, 100)
window = st.sidebar.slider("Fenêtre (window)", 1, 10, 5)
min_count = st.sidebar.slider("Min Count", 1, 5, 1)
epochs = st.sidebar.slider("Epochs", 5, 100, 20)

# --- Onglets ---
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Entraînement Modèle", 
    "2. Exploration Sémantique", 
    "3. Similarité Phrases",
    "4. Expansion de Requête"
])

# ==============================================================================
# Tab 1 : Entraînement
# ==============================================================================
with tab1:
    st.header("Entraînement du Modèle")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Données")
        source_data = st.radio("Source", ["Corpus Jouet (Défaut)", "Texte Personnalisé"])
        
        if source_data == "Texte Personnalisé":
            raw_text = st.text_area("Entrez vos phrases (une par ligne)", 
                                    "l'apprentissage automatique est fascinant\nles réseaux de neurones apprennent vite")
            if st.button("Charger ce texte"):
                st.session_state["corpus_train"] = [line.lower().split() for line in raw_text.splitlines() if line.strip()]
                st.success(f"Chargé {len(st.session_state['corpus_train'])} phrases.")
        
        st.write(f"**Taille du corpus actuel :** {len(st.session_state['corpus_train'])} phrases")
        with st.expander("Voir le corpus"):
            st.write(st.session_state["corpus_train"])

    with col2:
        st.subheader("Lancer l'entraînement")
        if st.button("Entraîner le modèle"):
            with st.spinner(f"Entraînement de {algo}..."):
                try:
                    if algo == "Word2Vec":
                        model = TP6.entrainer_modele_word2vec(
                            st.session_state["corpus_train"], 
                            taille_vecteur=vector_size, 
                            fenetre=window, 
                            min_count=min_count, 
                            epochs=epochs
                        )
                    else:
                        model = TP6.entrainer_modele_fasttext(
                            st.session_state["corpus_train"], 
                            taille_vecteur=vector_size, 
                            fenetre=window, 
                            min_count=min_count, 
                            epochs=epochs
                        )
                    
                    st.session_state["model_w2v"] = model
                    st.success(f"Modèle {algo} entraîné avec succès !")
                    st.info(f"Vocabulaire : {len(model.wv.index_to_key)} mots.")
                    
                except Exception as e:
                    st.error(f"Erreur durant l'entraînement : {e}")

# ==============================================================================
# Tab 2 : Exploration
# ==============================================================================
with tab2:
    st.header("Exploration du Modèle")
    
    if st.session_state["model_w2v"] is None:
        st.warning("Veuillez d'abord entraîner un modèle dans l'onglet 1.")
    else:
        model = st.session_state["model_w2v"]
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Voisins les plus proches (Most Similar)")
            mot_target = st.text_input("Mot cible", "chat")
            
            if st.button("Chercher Voisins"):
                if mot_target in model.wv:
                    voisins = model.wv.most_similar(mot_target, topn=10)
                    df_v = pd.DataFrame(voisins, columns=["Mot", "Similarité"])
                    st.table(df_v)
                else:
                    st.error(f"Le mot '{mot_target}' n'est pas dans le vocabulaire.")
                    
        with c2:
            st.subheader("Calcul de Similarité (Mot à Mot)")
            m1 = st.text_input("Mot 1", "chat")
            m2 = st.text_input("Mot 2", "chien")
            
            if st.button("Calculer Similarité"):
                if m1 in model.wv and m2 in model.wv:
                    sim = model.wv.similarity(m1, m2)
                    st.metric(f"Sim({m1}, {m2})", f"{sim:.4f}")
                else:
                    st.warning("L'un des mots n'est pas dans le vocabulaire.")

# ==============================================================================
# Tab 3 : Similarité Phrases
# ==============================================================================
with tab3:
    st.header("Similarité de Phrases")
    
    if st.session_state["model_w2v"] is None:
        st.warning("Modèle non chargé.")
    else:
        model = st.session_state["model_w2v"]
        
        txt_p1 = st.text_input("Phrase 1", "le chat mange")
        txt_p2 = st.text_input("Phrase 2", "le chien mange")
        
        col_res1, col_res2 = st.columns(2)
        
        if st.button("Comparer Phrases"):
            tokens1 = txt_p1.lower().split()
            tokens2 = txt_p2.lower().split()
            
            # Utilisation de la fonction du TP6
            try:
                sim = TP6.similarite_phrases(
                    tokens1, tokens2, 
                    model, 
                    strategie_agregation="moyenne", 
                    mesure="cosinus"
                )
                st.success(f"Similarité Cosinus : **{sim:.4f}**")
                
                # Détails vecteurs
                v1 = TP6.plongement_phrase_par_mots(tokens1, model)
                v2 = TP6.plongement_phrase_par_mots(tokens2, model)
                
                with st.expander("Voir les vecteurs"):
                    st.write("Vecteur Phrase 1:", v1)
                    st.write("Vecteur Phrase 2:", v2)
                    
            except Exception as e:
                st.error(f"Erreur de calcul : {e}")

# ==============================================================================
# Tab 4 : Expansion de Requête
# ==============================================================================
with tab4:
    st.header("Expansion de Requête")
    
    if st.session_state["model_w2v"] is None:
        st.warning("Modèle requis.")
    else:
        model = st.session_state["model_w2v"]
        
        req = st.text_input("Requête utilisateur", "chat")
        k_val = st.slider("Nombre de termes d'expansion", 1, 10, 3)
        alpha = st.slider("Poids Original (alpha)", 0.1, 2.0, 1.0)
        beta = st.slider("Poids Expansion (beta)", 0.1, 2.0, 0.5)
        
        if st.button("Simuler Expansion"):
            tokens_req = req.lower().split()
            
            # Appel manuel des étapes pour visualisation
            # 1. Termes d'expansion
            expansion_dict = {}
            for token in tokens_req:
                if token in model.wv:
                    similaires = model.wv.most_similar(token, topn=k_val)
                    expansion_dict[token] = similaires
            
            st.subheader("1. Termes suggérés pour l'expansion")
            st.json(expansion_dict)
            
            # 2. Requête Étendue (via TP6 si possible, sinon manuel ici pour démo)
            try:
                req_etendue = TP6.construire_requete_etendue(tokens_req, expansion_dict, alpha, beta)
                st.subheader("2. Vecteur Requête Étendu (Pondéré)")
                st.write(req_etendue)
                
                # Visualisation WordCloud
                st.subheader("3. Visualisation Nuage de Mots (Requête Étendue)")
                wc = TP6.WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(req_etendue)
                
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Erreur durant l'expansion : {e}")

