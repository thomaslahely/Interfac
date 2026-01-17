import streamlit as st
import sys
import pandas as pd
import math
from pathlib import Path
import App_TP4 as TP4
# --- Configuration de la page ---
st.set_page_config(page_title="TP4 - Vectorisation & Modèles", layout="wide")


# --- Titre et Intro ---
st.title("TP4 - Vectorisation de Texte et Modèles Mathématiques")
st.markdown("""
Cette interface offre un banc d'essai interactif pour les techniques de vectorisation du **TP4**.
Explorez comment les mots sont transformés en vecteurs numériques utilisables par des algorithmes.
""")

# --- Sidebar : Configuration Globale ---
st.sidebar.header("Vocabulaire de Test")
default_vocab_txt = "chat, chien, dort, mange, maison, souris, fromage"
vocab_input = st.sidebar.text_area("Mots du vocabulaire (séparés par virgule)", value=default_vocab_txt)

# Construction du vocabulaire partagé par tous les onglets
mots_vocab = [m.strip().lower() for m in vocab_input.split(",") if m.strip()]
vocab_direct = {mot: i for i, mot in enumerate(mots_vocab)}
vocab_inverse = {i: mot for i, mot in enumerate(mots_vocab)}

st.sidebar.markdown("---")
st.sidebar.caption("Aperçu du Vocabulaire :")
st.sidebar.json(vocab_direct)


# --- Onglets ---
tabs = st.tabs([
    "1. Encodage & One-Hot", 
    "2. Bag of Words (BoW)", 
    "3. TF-IDF & BM25", 
    "4. Normalisation", 
    "5. N-grammes"
])

# ==============================================================================
# Tab 1 : Encodage & One-Hot
# ==============================================================================
with tabs[0]:
    st.header("Encodage et One-Hot Encoding")
    st.markdown("Transformer des mots en indices ou en vecteurs creux.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1.1 Texte vers Indices")
        txt_idx = st.text_input("Texte à convertir en indices", value="chat dort maison")
        
        if st.button("Convertir en Indices"):
            res = TP4.texte_vers_indices(txt_idx, vocab_direct)
            st.write(f"**Analyse :** '{txt_idx}'")
            st.write(f"**Indices :** `{res}`")
            st.caption("Note : -1 signifie que le mot n'est pas dans le vocabulaire.")
            
        st.divider()
        st.subheader("1.2 Indices vers Texte")
        idx_input = st.text_input("Indices à décoder (ex: 0, 3, -1)", value="0, 3, 1")
        
        if st.button("Décoder Indices"):
            try:
                # Parsing simple des indices
                indices_list = [int(x.strip()) for x in idx_input.split(",") if x.strip()]
                res_txt = TP4.indices_vers_texte(indices_list, vocab_inverse)
                st.write(f"**Indices :** {indices_list}")
                st.success(f"**Texte reconstruit :** {res_txt}")
            except ValueError:
                st.error("Format invalide. Entrez des entiers séparés par des virgules.")

    with col2:
        st.subheader("1.3 One-Hot Encoding")
        st.markdown("*Chaque mot devient un vecteur de la taille du vocabulaire avec un seul '1'.*")
        
        txt_oh = st.text_input("Texte pour One-Hot", value="chat chien")
        
        if st.button("Générer One-Hot"):
            vecs = TP4.one_hot_encode_mots(txt_oh, vocab_direct)
            
            st.write(f"Structure ({len(vecs)} vecteurs de taille {len(vocab_direct)}) :")
            
            # Affichage joli avec DataFrame
            df_oh = pd.DataFrame(vecs, columns=mots_vocab)
            # Ajout d'une colonne index pour les mots
            mots_split = txt_oh.lower().split()
            if len(mots_split) == len(vecs):
                df_oh.index = mots_split
                
            st.dataframe(df_oh)
            
            # Test de décodage immédiat
            decoded = TP4.one_hot_decode(vecs, vocab_inverse)
            st.info(f"Vérification décodage : '{decoded}'")


# ==============================================================================
# Tab 2 : Bag of Words
# ==============================================================================
with tabs[1]:
    st.header("Bag of Words (Sac de Mots)")
    st.markdown("Représentation globale d'un document. L'ordre des mots est perdu.")
    
    txt_bow = st.text_input("Document à vectoriser", value="chat mange chat et le chien dort")
    
    col_bow1, col_bow2 = st.columns(2)
    
    with col_bow1:
        st.subheader("BoW Binaire (Présence)")
        if st.button("Calculer BoW Binaire"):
            vec = TP4.bag_of_words_binaire(txt_bow, vocab_direct)
            st.write("Vecteur :")
            st.code(str(vec))
            
            df_viz = pd.DataFrame([vec], columns=mots_vocab)
            st.dataframe(df_viz)
            
            decoded = TP4.bag_of_words_decode(vec, vocab_inverse)
            st.caption(f"Reconstruction (ordre vocabulaire) : '{decoded}'")

    with col_bow2:
        st.subheader("BoW Occurrences (Comptage)")
        if st.button("Calculer BoW Occurrences"):
            vec = TP4.bag_of_words_occurrences(txt_bow, vocab_direct)
            st.write("Vecteur :")
            st.code(str(vec))
            
            df_viz = pd.DataFrame([vec], columns=mots_vocab)
            st.dataframe(df_viz)
            
            decoded = TP4.bag_of_words_occurrences_decode(vec, vocab_inverse)
            st.caption(f"Reconstruction (avec répétitions) : '{decoded}'")


# ==============================================================================
# Tab 3 : TF-IDF & BM25
# ==============================================================================
with tabs[2]:
    st.header("TF-IDF et BM25")
    st.markdown("Pondération avancée prenant en compte la rareté des mots (IDF).")
    
    st.subheader("Configuration du Corpus")
    corpus_input = st.text_area("Entrez un petit corpus (une phrase par ligne)", 
                 value="chat dort\\nchien dort\\nchat mange\\nchien mange maison")
    
    corpus = [line.strip() for line in corpus_input.split('\\n') if line.strip()]
    st.write(f"**Nombre de documents :** {len(corpus)}")
    
    col_meth1, col_meth2 = st.columns(2)
    
    with col_meth1:
        st.subheader("Matrice TF-IDF")
        methode_idf = st.selectbox("Formule IDF", 
                                   ['smooth', 'classique', 'smooth_P1', 'logplus', 'max', 'bm25', 'bm25_smooth'])
        
        if st.button("Calculer TF-IDF"):
            # 1. Vecteur IDF
            vecteur_idf = TP4.calcul_idf(corpus, vocab_direct, methode=methode_idf)
            st.markdown("**Vecteur IDF Global :**")
            st.dataframe(pd.DataFrame([vecteur_idf], columns=mots_vocab))
            
            # 2. Matrice TF-IDF
            matrice = TP4.calcul_tf_idf(corpus, vocab_direct, methode_idf=methode_idf)
            
            st.markdown("**Matrice TF-IDF (Documents x Mots) :**")
            df_tfidf = pd.DataFrame(matrice, columns=mots_vocab, index=[f"Doc {i+1}" for i in range(len(corpus))])
            st.dataframe(df_tfidf.style.background_gradient(cmap="Blues", axis=None))

    with col_meth2:
        st.subheader("Matrice BM25")
        st.markdown("Variante probabiliste utilisée par les moteurs de recherche.")
        k1 = st.slider("k1 (Saturation)", 0.0, 3.0, 1.5)
        b = st.slider("b (Normalisation longueur)", 0.0, 1.0, 0.75)
        
        if st.button("Calculer BM25"):
            matrice_bm25 = TP4.calcul_bm25(corpus, vocab_direct, k1=k1, b=b)
            
            st.markdown("**Matrice BM25 :**")
            df_bm25 = pd.DataFrame(matrice_bm25, columns=mots_vocab, index=[f"Doc {i+1}" for i in range(len(corpus))])
            st.dataframe(df_bm25.style.background_gradient(cmap="Greens", axis=None))


# ==============================================================================
# Tab 4 : Normalisation
# ==============================================================================
with tabs[3]:
    st.header("Normalisation de Vecteurs")
    st.markdown("Techniques pour mettre à l'échelle les vecteurs.")
    
    vec_input_str = st.text_input("Vecteur brut (séparé par virgule)", value="2.0, 5.0, 10.0, 2.0")
    
    try:
        vec_raw = [float(x.strip()) for x in vec_input_str.split(",") if x.strip()]
        
        col_norm1, col_norm2 = st.columns(2)
        
        # L1 & L2
        with col_norm1:
            st.subheader("Normes L1 & L2")
            if st.button("Calculer L1 / L2"):
                norm_l1 = TP4.normaliser_L1(vec_raw)
                norm_l2 = TP4.normaliser_L2(vec_raw)
                
                res_df = pd.DataFrame({
                    "Brut": vec_raw,
                    "L1 (Somme=1)": norm_l1,
                    "L2 (Euclidienne=1)": norm_l2
                })
                st.dataframe(res_df)
                st.json({"Somme L1": sum(norm_l1), "Norme L2": math.sqrt(sum(x**2 for x in norm_l2))})

        # MinMax & ZScore
        with col_norm2:
            st.subheader("Mise à l'échelle")
            if st.button("Calculer MinMax / Z-Score"):
                norm_mm = TP4.normaliser_minmax(vec_raw)
                norm_z = TP4.standardiser_zscore(vec_raw)
                
                res_df2 = pd.DataFrame({
                    "Brut": vec_raw,
                    "MinMax [0,1]": norm_mm,
                    "Z-Score (Std)": norm_z
                })
                st.dataframe(res_df2)

    except ValueError:
        st.error("Veuillez entrer uniquement des nombres valides séparés par des virgules.")


# ==============================================================================
# Tab 5 : N-grammes
# ==============================================================================
with tabs[4]:
    st.header("Vectorisation par N-grammes")
    st.markdown("Au lieu de compter les mots, on compte les séquences de N mots (ex: 'chat dort').")
    
    txt_ng = st.text_input("Phrase à analyser", value="le chat dort bien sur le tapis")
    n_val = st.slider("Taille N (1=unigramme, 2=bigramme...)", 1, 4, 2)
    
    # Génération à la volée d'un vocabulaire pour la démo
    # Pour que ça marche, on triche un peu en créant le vocabulaire à partir de la phrase elle-même
    # Sinon le vecteur serait vide car vocabulaire_direct (tab1) ne contient pas de tuples
    
    if st.button("Générer et Vectoriser"):
        # 1. Extraction manuelle pour démo
        tokens = txt_ng.lower().split()
        ngrams_demo = []
        if len(tokens) >= n_val:
            for i in range(len(tokens) - n_val + 1):
                ngrams_demo.append(tuple(tokens[i : i + n_val]))
        
        # On ajoute du bruit (faux ngrams) pour simuler un vocabulaire plus large
        ngrams_demo.append(("faux", "ngram"))
        ngrams_demo.append(("autre", "truc"))
        
        # 2. Construction Dictionnaires
        vocab_ng_list, dico_dir, dico_inv = TP4.construire_dictionnaire_ngrammes(ngrams_demo)
        
        st.subheader(f"Vocabulaire ({len(vocab_ng_list)} n-grammes)")
        st.write(vocab_ng_list)
        
        # 3. Vectorisation
        col_ng1, col_ng2, col_ng3 = st.columns(3)
        
        with col_ng1:
            st.markdown("**BoW Binaire**")
            vec_bin = TP4.encoder_bow_ngrammes(txt_ng, n_val, vocab_ng_list, dico_dir, 'binaire')
            st.write(vec_bin)
            
        with col_ng2:
            st.markdown("**TF (Fréquence)**")
            vec_tf = TP4.encoder_tf_ngrammes(txt_ng, n_val, vocab_ng_list, dico_dir)
            st.write([round(x, 3) for x in vec_tf])
            
        with col_ng3:
            st.markdown("**TF-IDF (IDF fictif=1.0)**")
            idf_fictif = [1.0] * len(vec_tf)
            vec_tfidf = TP4.encoder_tfidf_ngrammes(txt_ng, n_val, vocab_ng_list, dico_dir, idf_fictif)
            st.write([round(x, 3) for x in vec_tfidf])
