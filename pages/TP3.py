import streamlit as st
import TP4
import pandas as pd
import math

st.set_page_config(page_title="TP3 - Vectorisation & Modèles", layout="wide")

st.title("TP3 - Vectorisation de Texte et Modèles de Langage")
st.markdown("""
Cette interface permet de tester les fonctions du TP3 : Encodage, One-Hot, Bag-of-Words, TF-IDF, BM25, Normalisation et N-grammes.
""")

# --- Sidebar: Vocabulaire ---
st.sidebar.header("Configuration Vocabulaire")
default_vocab_txt = "chat, chien, dort, mange, maison, souris, fromage"
vocab_input = st.sidebar.text_area("Mots du vocabulaire (séparés par virgule)", value=default_vocab_txt)

# Construction du vocabulaire
mots_vocab = [m.strip().lower() for m in vocab_input.split(",") if m.strip()]
vocab_direct = {mot: i for i, mot in enumerate(mots_vocab)}
vocab_inverse = {i: mot for i, mot in enumerate(mots_vocab)}

st.sidebar.write("Vocabulaire actuel :")
st.sidebar.json(vocab_direct)

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Encodage & One-Hot", 
    "Bag of Words", 
    "TF-IDF & BM25",
    "Normalisation",
    "N-grammes"
])

# --- Tab 1: Encodage & One-Hot ---
with tab1:
    st.subheader("Encodage / Décodage (Indices)")
    col1, col2 = st.columns(2)
    
    with col1:
        txt_indices = st.text_input("Texte à encoder (Indices)", value="chat dort")
        if st.button("Encoder (Indices)"):
            res = TP4.texte_vers_indices(txt_indices, vocab_direct)
            st.write(f"Indices : {res}")
            
    with col2:
        indices_input = st.text_input("Indices à décoder (ex: 0, 2)", value="0, 2")
        if st.button("Décoder (Indices)"):
            try:
                idx_list = [int(x.strip()) for x in indices_input.split(",") if x.strip()]
                res = TP4.indices_vers_texte(idx_list, vocab_inverse)
                st.success(f"Texte : {res}")
            except ValueError:
                st.error("Erreur de format des indices.")

    st.markdown("---")
    st.subheader("One-Hot Encoding")
    
    txt_onehot = st.text_input("Texte à encoder (One-Hot)", value="chat maison")
    if st.button("Encoder (One-Hot)"):
        vecs = TP4.one_hot_encode_mots(txt_onehot, vocab_direct)
        st.write("Vecteurs One-Hot :")
        st.write(vecs)
        
        # Décodage immédiat pour vérification
        decoded = TP4.one_hot_decode(vecs, vocab_inverse)
        st.info(f"Reconstruction : {decoded}")

# --- Tab 2: Bag of Words ---
with tab2:
    st.subheader("Bag of Words (BoW)")
    
    txt_bow = st.text_input("Texte pour BoW", value="chat mange chat")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**BoW Binaire**")
        if st.button("Générer BoW Binaire"):
            vec = TP4.bag_of_words_binaire(txt_bow, vocab_direct)
            st.write(f"Vecteur : {vec}")
            decoded = TP4.bag_of_words_decode(vec, vocab_inverse)
            st.write(f"Décodé (ordre vocab) : {decoded}")
            
    with col2:
        st.markdown("**BoW Occurrences**")
        if st.button("Générer BoW Occurrences"):
            vec = TP4.bag_of_words_occurrences(txt_bow, vocab_direct)
            st.write(f"Vecteur : {vec}")
            decoded = TP4.bag_of_words_occurrences_decode(vec, vocab_inverse)
            st.write(f"Décodé (répété) : {decoded}")

# --- Tab 3: TF-IDF & BM25 ---
with tab3:
    st.subheader("TF-IDF & BM25")
    
    st.write("Définissez un petit corpus pour tester les calculs globaux.")
    corpus_txt = st.text_area("Corpus (une phrase par ligne)", 
                              value="chat dort\nchien dort\nchat mange\nchien mange maison")
    corpus = [line.strip() for line in corpus_txt.split("\n") if line.strip()]
    
    st.write(f"Nombre de documents : {len(corpus)}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### TF-IDF")
        methode_idf = st.selectbox("Méthode IDF", ["classique", "smooth", "smooth_P1", "logplus", "max", "bm25", "bm25_smooth"])
        
        if st.button("Calculer TF-IDF"):
            # Afficher IDF global
            idf_vec = TP4.calcul_idf(corpus, vocab_direct, methode=methode_idf)
            df_idf = pd.DataFrame([idf_vec], columns=mots_vocab, index=["IDF"])
            st.write("Vecteur IDF Global :")
            st.dataframe(df_idf)
            
            # Matrice TF-IDF
            matrice = TP4.calcul_tf_idf(corpus, vocab_direct, methode_idf=methode_idf)
            df_tfidf = pd.DataFrame(matrice, columns=mots_vocab, index=[f"Doc {i}" for i in range(len(corpus))])
            st.write("Matrice TF-IDF :")
            st.dataframe(df_tfidf)

    with col2:
        st.markdown("### BM25")
        k1 = st.number_input("k1", 0.0, 5.0, 1.5)
        b = st.number_input("b", 0.0, 1.0, 0.75)
        
        if st.button("Calculer BM25"):
            matrice_bm25 = TP4.calcul_bm25(corpus, vocab_direct, k1=k1, b=b)
            df_bm25 = pd.DataFrame(matrice_bm25, columns=mots_vocab, index=[f"Doc {i}" for i in range(len(corpus))])
            st.write("Matrice BM25 :")
            st.dataframe(df_bm25)

# --- Tab 4: Normalisation ---
with tab4:
    st.subheader("Normalisation de Vecteurs")
    
    vec_input = st.text_input("Vecteur (séparé par virgule)", value="1.0, 2.0, 3.0, 4.0")
    try:
        vec_float = [float(x.strip()) for x in vec_input.split(",") if x.strip()]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("L1 (Somme=1)"):
                res = TP4.normaliser_L1(vec_float)
                st.write(res)
                st.write(f"Somme : {sum(res):.2f}")
                
        with col2:
            if st.button("L2 (Euclidienne)"):
                res = TP4.normaliser_L2(vec_float)
                st.write(res)
                norme = math.sqrt(sum(x**2 for x in res))
                st.write(f"Norme : {norme:.2f}")
                
        with col3:
            if st.button("Min-Max [0,1]"):
                res = TP4.normaliser_minmax(vec_float)
                st.write(res)
                
        with col4:
            if st.button("Z-Score (Std)"):
                res = TP4.standardiser_zscore(vec_float)
                st.write(res)
                
    except ValueError:
        st.error("Veuillez entrer des nombres valides.")

# --- Tab 5: N-grammes ---
with tab5:
    st.subheader("Vectorisation N-grammes")
    
    txt_ng = st.text_input("Texte N-grammes", value="le chat dort bien sur le tapis")
    n_val = st.slider("N (taille)", 1, 3, 2)
    
    # Génération dynamique du vocabulaire n-grammes pour la démo
    # On prend les n-grammes du texte lui-même pour être sûr d'avoir des matches
    tokens = txt_ng.lower().split()
    ngrams_list = []
    if len(tokens) >= n_val:
        for i in range(len(tokens) - n_val + 1):
            ngrams_list.append(tuple(tokens[i : i + n_val]))
    
    # On ajoute quelques faux pour le test
    ngrams_list.append(("faux", "ngram"))
    
    # Construction dico
    dico_direct_ng, dico_inverse_ng = TP4.construire_dictionnaire_ngrammes(ngrams_list)
    vocab_ng_list = sorted(list(dico_direct_ng.keys()))
    
    st.write("Vocabulaire N-grammes généré (basé sur le texte + bruit) :")
    st.write(vocab_ng_list)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("BoW N-grammes (Binaire)"):
            vec = TP4.encoder_bow_ngrammes(txt_ng, n_val, vocab_ng_list, dico_direct_ng, 'binaire')
            st.write(vec)
            
    with col2:
        if st.button("TF N-grammes"):
            vec = TP4.encoder_tf_ngrammes(txt_ng, n_val, vocab_ng_list, dico_direct_ng)
            st.write(vec)
            
    with col3:
        if st.button("TF-IDF N-grammes"):
            # IDF fictif (tout à 1.0 pour simplifier)
            idf_fictif = [1.0] * len(dico_direct_ng)
            vec = TP4.encoder_tfidf_ngrammes(txt_ng, n_val, vocab_ng_list, dico_direct_ng, idf_fictif)
            st.write(vec)
            st.caption("(IDF fixé à 1.0 pour ce test)")
