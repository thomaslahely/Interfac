import streamlit as st
import sys
import pandas as pd
import math
from pathlib import Path
import app_TP5 as TP5

# --- Titre ---
st.title("TP5 - Vectorisation Avancée et Mesures de Distance")
st.markdown("""
Cette interface permet d'expérimenter les notions du TP5 :
1.  **Vectorisation** : Phrase vs Document (Mots ou Agrégé).
2.  **Distances** : Comparaison mathématique de vecteurs (Euclidienne, Cosinus, etc.).
""")

# --- Sidebar : Vocabulaire Commun ---
st.sidebar.header("Configuration Vocabulaire")
default_vocab = "chat, chien, dort, mange, maison, souris, fromage"
vocab_input = st.sidebar.text_area("Mots du vocabulaire", value=default_vocab)

mots_vocab = [m.strip().lower() for m in vocab_input.split(",") if m.strip()]
vocab_direct = {mot: i for i, mot in enumerate(mots_vocab)}
st.sidebar.caption(f"Vocabulaire ({len(vocab_direct)} mots)")
st.sidebar.json(vocab_direct)

# --- Onglets ---
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Vectorisation Phrase", 
    "2. Vectorisation Document", 
    "3. Distances & Similarités",
    "4. Métadonnées Corpus"
])

# ==============================================================================
# Tab 1 : Vectorisation Phrase
# ==============================================================================
with tab1:
    st.header("Vectorisation de Phrase")
    st.markdown("Transforme une liste de mots en un vecteur numérique.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        phrase_txt = st.text_input("Phrase à vectoriser", value="le chat dort")
        methode = st.selectbox("Méthode", ["tf", "bow", "tfidf"])
    
    with col2:
        # Configuration IDF pour TF-IDF
        use_fake_idf = False
        idf_vector = []
        if methode == "tfidf":
            st.markdown("**Configuration IDF** (simulée)")
            idf_vals = st.text_input("Valeurs IDF (séparées par virgule, même ordre que vocab)", 
                                     value="0.5, 0.5, 1.0, 1.0, 1.2, 1.5, 1.5")
            try:
                idf_vector = [float(x) for x in idf_vals.split(",")]
                if len(idf_vector) != len(mots_vocab):
                    st.warning(f"Le vecteur IDF doit avoir {len(mots_vocab)} valeurs (actuellement {len(idf_vector)}).")
                else:
                    use_fake_idf = True
            except:
                st.error("Erreur de format IDF")

    if st.button("Vectoriser Phrase"):
        # Préparation (la fonction attend une liste de tokens)
        tokens = phrase_txt.lower().split()
        
        params = {}
        if methode == "tfidf" and use_fake_idf:
            params["vecteur_idf"] = idf_vector
            
        try:
            vec = TP5.vectoriser_phrase(tokens, vocab_direct, methode=methode, **params)
            
            # Affichage
            st.write("Vecteur Résultat :")
            df = pd.DataFrame([vec], columns=mots_vocab)
            st.dataframe(df)
            
        except Exception as e:
            st.error(f"Erreur : {e}")


# ==============================================================================
# Tab 2 : Vectorisation Document
# ==============================================================================
with tab2:
    st.header("Vectorisation de Document")
    st.info("Un document est vu comme une **liste de phrases**.")
    
    doc_text = st.text_area("Contenu du document (une phrase par ligne)", 
                            value="le chat dort\nle chien mange la souris")
    
    # Parsing
    phrases = [line.strip().split() for line in doc_text.split("\n") if line.strip()]
    st.write("Structure interne :", phrases)
    
    type_vect = st.radio("Approche", ["Concaténation (Sac de mots global)", "Agrégation (Somme/Moyenne/Max)"])
    
    if "Concaténation" in type_vect:
        if st.button("Vectoriser (Document Mots)"):
            vec = TP5.vectoriser_document_mots(phrases, vocab_direct, methode="tf")
            st.success("Méthode : Term Frequency (TF) sur l'ensemble des mots aplati")
            st.dataframe(pd.DataFrame([vec], columns=mots_vocab))
            
    else:
        col_agg1, col_agg2 = st.columns(2)
        with col_agg1:
            strat = st.selectbox("Stratégie d'agrégation", ["moyenne", "somme", "max"])
        with col_agg2:
            methode_base = st.selectbox("Méthode par phrase", ["tf", "bow"])
            
        if st.button("Vectoriser (Document Agrégé)"):
            vec = TP5.vectoriser_document_agrege(phrases, vocab_direct, methode=methode_base, strategie=strat)
            st.dataframe(pd.DataFrame([vec], columns=mots_vocab))
            
            with st.expander("Voir le détail phrase par phrase"):
                for i, p in enumerate(phrases):
                    v_p = TP5.vectoriser_phrase(p, vocab_direct, methode=methode_base)
                    st.write(f"Phrase {i+1} : {v_p}")


# ==============================================================================
# Tab 3 : Distances
# ==============================================================================
with tab3:
    st.header("Calculateur de Distances")
    
    c1, c2 = st.columns(2)
    with c1:
        v1_txt = st.text_input("Vecteur 1", "0, 1, 1")
    with c2:
        v2_txt = st.text_input("Vecteur 2", "1, 0, 1")
        
    try:
        v1 = [float(x) for x in v1_txt.split(",")]
        v2 = [float(x) for x in v2_txt.split(",")]
        
        if len(v1) != len(v2):
            st.error("Les vecteurs doivent être de même taille.")
        else:
            st.markdown("### Résultats")
            
            res_data = {
                "Euclidienne (L2)": TP5.calcul_distance_euclidienne(v1, v2),
                "Manhattan (L1)": TP5.calcul_distance_manhattan(v1, v2),
                "Tchebychev (L-inf)": TP5.calcul_distance_tchebychev(v1, v2),
                "Bray-Curtis": TP5.calcul_distance_bray_curtis(v1, v2),
                "Similarité Cosinus": TP5.calcul_similarite_cosinus(v1, v2),
                "Distance Cosinus": TP5.calcul_distance_cosinus(v1, v2)
            }
            
            st.table(pd.DataFrame(list(res_data.items()), columns=["Métrique", "Valeur"]))
            
            # Minkowski spécifique
            st.markdown("---")
            p_val = st.number_input("Paramètre p pour Minkowski", min_value=1, value=3)
            st.write(f"Distance Minkowski (p={p_val}) : {TP5.calcul_distance_minkowski(v1, v2, p_val)}")
            
    except ValueError:
        st.warning("Entrez des nombres valides séparés par des virgules.")


# ==============================================================================
# Tab 4 : Métadonnées Corpus
# ==============================================================================
with tab4:
    st.header("Extraction de Métadonnées")
    st.markdown("Analyse automatique d'un dossier pour préparer l'indexation.")
    
    # Hack pour trouver le dossier Textes
    default_path = "Textes"
    root_path = Path(__file__).resolve().parent.parent.parent
    potential_textes = root_path / "Textes"
    if potential_textes.exists():
        default_path = str(potential_textes)
        
    path_input = st.text_input("Chemin du corpus", value=default_path)
    
    if st.button("Construire Méta-Corpus"):
        meta = TP5.construire_meta_corpus(Path(path_input))
        
        if meta:
            st.success(f"Analyse terminée : {len(meta)} documents trouvés.")
            st.write("Aperçu des données extraites :")
            st.json(meta)
        else:
            st.warning("Aucun document trouvé ou chemin invalide.")
