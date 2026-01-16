import streamlit as st
import TP3
import matplotlib.pyplot as plt
import io
import pandas as pd

st.set_page_config(page_title="TP3 - Analyse de Corpus", layout="wide")

st.title("TP3 - Analyse et Prétraitement de Corpus")
st.markdown("""
Cette interface permet de tester les fonctions du TP3 : segmentation, tokenisation, n-grammes, statistiques et pipelines de prétraitement.
""")

# --- Sidebar: Configuration Globale ---
st.sidebar.header("Configuration")
langue = st.sidebar.selectbox("Langue", ["fr", "en"])

# --- Input Data ---
st.header("1. Données d'entrée")
input_method = st.radio("Source des données :", ["Texte direct", "Exemple par défaut"])

if input_method == "Texte direct":
    raw_text = st.text_area("Entrez votre texte ici :", height=150, 
                            value="Bonjour tout le monde. C'est un test, n'est-ce pas ? 12.05.2025 est une date.")
else:
    raw_text = "Bonjour tout le monde. C'est un test, n'est-ce pas ? Le Dr. Martin habite à Paris. 12.05.2025 est une date importante. L'intelligence artificielle est fascinante."
    st.info(f"Texte utilisé : {raw_text}")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Segmentation & Tokenisation", 
    "N-grammes", 
    "Statistiques & Vocabulaire",
    "Filtrage & Morphologie",
    "Comparaison Pipelines"
])

# --- Tab 1: Segmentation & Tokenisation ---
with tab1:
    st.subheader("Segmentation en phrases")
    
    col1, col2 = st.columns(2)
    with col1:
        opt_dates = st.checkbox("Gérer dates (JJ.MM.AAAA)", value=True)
        opt_decimaux = st.checkbox("Gérer décimaux (3.14)", value=True)
    with col2:
        opt_sigles = st.checkbox("Gérer sigles (P.D.G.)", value=False)
        abbr_input = st.text_input("Abréviations (séparées par virgule)", value="Dr., M., Mme.")
    
    abbr_list = [a.strip() for a in abbr_input.split(",") if a.strip()]
    options_seg = {
        "gerer_dates": opt_dates,
        "gerer_decimaux": opt_decimaux,
        "gerer_sigles": opt_sigles
    }
    
    if st.button("Segmenter Phrases"):
        phrases = TP3.segmenter_phrases(raw_text, abreviations=abbr_list, option=options_seg)
        st.write(f"**Nombre de phrases :** {len(phrases)}")
        for i, p in enumerate(phrases):
            st.text(f"{i+1}: {p}")
            
    st.markdown("---")
    st.subheader("Tokenisation")
    keep_tags = st.checkbox("Garder balises HTML", value=False)
    
    if st.button("Tokeniser Document"):
        # On refait la segmentation pour être sûr d'avoir les dernières options
        doc_tokens = TP3.tokeniser_document(raw_text, abreviations=abbr_list, option=options_seg, balise=keep_tags)
        st.write("**Résultat (Liste de listes de tokens) :**")
        st.json(doc_tokens)
        
        flat_tokens = TP3.aplatir_tokens(doc_tokens)
        st.write(f"**Total tokens :** {len(flat_tokens)}")
        st.write(f"**Hapax (occurence unique) :** {TP3.tokens_hapax(flat_tokens)}")

# --- Tab 2: N-grammes ---
with tab2:
    st.subheader("Génération de N-grammes")
    n_val = st.slider("Taille N (N-gramme)", 1, 5, 2)
    niveau = st.selectbox("Niveau", ["phrase", "document"])
    par_phrase = st.checkbox("Respecter frontières de phrases", value=True, help="Si coché, ne crée pas de n-grammes à cheval sur deux phrases.")
    
    if st.button("Générer N-grammes"):
        # Préparation des données selon le niveau attendu par la fonction
        doc_tokens = TP3.tokeniser_document(raw_text, abreviations=abbr_list, option=options_seg)
        
        if niveau == "phrase":
            # Pour la démo, on montre les n-grammes de la première phrase ou de toutes les phrases séparément
            st.write("N-grammes par phrase :")
            for i, phrase in enumerate(doc_tokens):
                grams = TP3.generer_ngrammes(phrase, n_val, niveau='phrase')
                st.write(f"Phrase {i+1}: {grams}")
        else:
            # Niveau document
            grams = TP3.generer_ngrammes(doc_tokens, n_val, niveau='document', par_phrase=par_phrase)
            st.write(f"N-grammes du document ({len(grams)}) :")
            st.write(grams)

# --- Tab 3: Statistiques & Vocabulaire ---
with tab3:
    st.subheader("Analyse Statistique")
    
    # On crée un mini corpus avec 1 document pour utiliser les fonctions de corpus
    corpus_demo = {"doc1": TP3.tokeniser_document(raw_text, abreviations=abbr_list, option=options_seg)}
    stopwords_list = TP3.construire_liste_stopwords(langue)
    
    if st.button("Calculer Statistiques"):
        stats = TP3.statistiques_globales_corpus(corpus_demo, stopwords_list)
        st.json(stats)
        
        st.subheader("Distribution")
        dist_len_mots = TP3.distribution_longueur_mots(corpus_demo)
        st.bar_chart(pd.Series(dist_len_mots))
        
        st.subheader("Top Tokens")
        top = TP3.tokens_plus_frequents(corpus_demo, n=10)
        if top:
            df_top = pd.DataFrame(top, columns=["Token", "Fréquence"])
            st.dataframe(df_top)
        else:
            st.warning("Pas assez de tokens.")

# --- Tab 4: Filtrage & Morphologie ---
with tab4:
    st.subheader("Pipeline de Nettoyage")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Filtrage**")
        f_alpha = st.checkbox("Supprimer non-alphabétiques", value=True)
        f_stop = st.checkbox("Supprimer stopwords", value=True)
        f_len = st.number_input("Longueur min", 0, 10, 3)
        f_occ_min = st.number_input("Occurrences min", 1, 10, 1)
    
    with col2:
        st.markdown("**Morphologie**")
        do_stem = st.checkbox("Stemming", value=False)
        do_lemm = st.checkbox("Lemmatisation", value=False)
    
    if st.button("Appliquer Pipeline"):
        # Tokenisation initiale
        doc_tokens = TP3.tokeniser_document(raw_text, abreviations=abbr_list, option=options_seg)
        tokens_flat = TP3.aplatir_tokens(doc_tokens)
        
        st.write("Tokens initiaux :", tokens_flat)
        
        # Config
        config = {
            "non_alphabetiques": f_alpha,
            "stopwords": f_stop,
            "longueur_min": f_len,
            "occ_min": f_occ_min,
            "stemming": do_stem,
            "lemmatisation": do_lemm
        }
        
        # 1. Filtrage
        tokens_filtered = TP3.pipeline_filtrage(tokens_flat, config, langue)
        st.write("Après filtrage :", tokens_filtered)
        
        # 2. Morphologie
        tokens_final = TP3.pipeline_morphologique(tokens_filtered, config, langue)
        st.success(f"Résultat final ({len(tokens_final)} tokens) :")
        st.write(tokens_final)

# --- Tab 5: Comparaison Pipelines ---
with tab5:
    st.subheader("Comparaison de Configurations (A, B, C, D, E)")
    st.markdown("""
    - **A**: Brut
    - **B**: Filtré (Stopwords, Len>3, Alpha)
    - **C**: B + Stemming
    - **D**: B + Lemmatisation
    - **E**: B + Stemming + Lemmatisation
    """)
    
    if st.button("Lancer Comparaison"):
        # Création corpus
        corpus_demo = {"doc1": TP3.tokeniser_document(raw_text, abreviations=abbr_list, option=options_seg)}
        
        # On doit adapter légèrement car 'analyser_configurations' attend un corpus brut ou tokenisé ?
        # Regardons le code de TP3.py : 'pipeline_pretraitement' prend un corpus.
        # 'analyser_configurations' appelle 'pipeline_pretraitement'.
        # 'pipeline_pretraitement' appelle 'aplatir_tokens' si c'est une liste de listes.
        # Donc on peut passer notre corpus_demo tokenisé.
        
        # Cependant, 'analyser_configurations' n'est pas complètement implémentée dans l'extrait lu (il manque la fin).
        # Je vais implémenter la logique ici manuellement pour être sûr.
        
        configs = {
            "A (Brut)": {"stopwords": False, "longueur_min": 0, "non_alphabetiques": False, "stemming": False, "lemmatisation": False},
            "B (Filtré)": {"stopwords": True, "longueur_min": 3, "non_alphabetiques": True, "stemming": False, "lemmatisation": False},
            "C (Filtré+Stem)": {"stopwords": True, "longueur_min": 3, "non_alphabetiques": True, "stemming": True, "lemmatisation": False},
            "D (Filtré+Lem)": {"stopwords": True, "longueur_min": 3, "non_alphabetiques": True, "stemming": False, "lemmatisation": True},
            "E (Complet)": {"stopwords": True, "longueur_min": 3, "non_alphabetiques": True, "stemming": True, "lemmatisation": True},
        }
        
        results = []
        
        for name, conf in configs.items():
            # Traitement
            corpus_traite = TP3.pipeline_pretraitement(corpus_demo, conf, langue)
            tokens = corpus_traite["doc1"]
            
            # Stats
            vocab = TP3.distribution_occurrences_tokens(corpus_traite)
            stats_lex = TP3.indicateurs_lexicaux(corpus_traite)
            
            res_row = {
                "Config": name,
                "Tokens": len(tokens),
                "Vocabulaire": len(vocab),
                "Richesse": f"{stats_lex['richesse_lexicale']:.2f}",
                "Hapax": f"{stats_lex['taux_hapax']:.2f}"
            }
            results.append(res_row)
            
        st.table(pd.DataFrame(results))
