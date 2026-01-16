import streamlit as st
import TP6
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shutil
import os

st.set_page_config(page_title="TP6 - Moteur de Recherche", layout="wide")

st.title("TP6 - Moteur de Recherche et Analyse de Corpus")
st.markdown("""
Cette interface permet de tester les fonctionnalités du TP6 : Indexation, Recherche, Évaluation et Analyse Structurelle.
""")

# --- Sidebar: Configuration ---
st.sidebar.header("Configuration")

# --- 1. Chargement du Corpus ---
st.sidebar.subheader("1. Corpus")
corpus_path_input = st.sidebar.text_input("Chemin du corpus", value="Textes")

# Fonction pour charger le corpus (mise en cache pour éviter de recharger à chaque interaction)
@st.cache_data
def load_corpus(path_str):
    path = Path(path_str)
    if not path.exists():
        return None, None, None
    
    # 1. Métadonnées
    meta = TP6.construire_meta_corpus(path)
    
    # 2. Chargement du texte brut
    corpus_texte = {}
    for id_doc, info in meta.items():
        try:
            with open(info['chemin'], 'r', encoding='utf-8') as f:
                # On lit tout, on nettoie un peu et on split en phrases/mots basique
                # Pour simplifier ici, on considère chaque ligne comme une phrase
                contenu = f.read()
                phrases = [line.strip().split() for line in contenu.split('\n') if line.strip()]
                corpus_texte[id_doc] = phrases
        except Exception as e:
            st.warning(f"Erreur lecture {info['chemin']}: {e}")
            
    # 3. Construction Vocabulaire
    vocab_set = set()
    for doc in corpus_texte.values():
        for phrase in doc:
            vocab_set.update([m.lower() for m in phrase]) # Minuscule
    
    vocab = {mot: i for i, mot in enumerate(sorted(list(vocab_set)))}
    
    return meta, corpus_texte, vocab

if st.sidebar.button("Charger / Recharger Corpus"):
    meta, corpus_texte, vocab = load_corpus(corpus_path_input)
    if meta:
        st.session_state['meta'] = meta
        st.session_state['corpus_texte'] = corpus_texte
        st.session_state['vocab'] = vocab
        st.sidebar.success(f"Corpus chargé : {len(meta)} documents, {len(vocab)} mots.")
    else:
        st.sidebar.error("Chemin invalide ou corpus vide.")

# Vérification si corpus chargé
if 'meta' not in st.session_state:
    st.info("Veuillez charger un corpus dans la barre latérale pour commencer.")
    st.stop()

meta = st.session_state['meta']
corpus_texte = st.session_state['corpus_texte']
vocab = st.session_state['vocab']

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Moteur de Recherche", 
    "Analyse Structurelle", 
    "Expérimentations",
    "Statistiques Corpus"
])

# --- Tab 1: Moteur de Recherche ---
with tab1:
    st.subheader("Recherche d'Information")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Votre requête :", value="etudiant")
    with col2:
        top_k = st.number_input("Nombre de résultats", 1, 20, 5)
        
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        methode_vect = st.selectbox("Vectorisation", ["tf", "tfidf", "bow"])
    with col_opt2:
        mesure_sim = st.selectbox("Mesure Similarité", ["cosinus", "euclidienne", "manhattan", "jaccard"])
    with col_opt3:
        niveau = st.selectbox("Niveau Granularité", ["document_mots", "document_agrege", "phrase"])
        
    if st.button("Rechercher"):
        # 1. Vectorisation Corpus (On le fait à la volée pour la démo, idéalement pré-calculé)
        # Pour TF-IDF, on a besoin d'un IDF global. On va simuler un IDF plat ou calculer vite fait si besoin.
        # TP6.vectoriser_corpus attend un param 'vecteur_idf' pour tfidf.
        
        params_vect = {}
        if methode_vect == "tfidf":
            # Calcul IDF basique
            idf_vec = [1.0] * len(vocab) # Simplification pour la démo
            # Pour faire mieux, il faudrait implémenter calcul_idf du TP4 ici ou l'importer
            params_vect["vecteur_idf"] = idf_vec
            
        with st.spinner("Vectorisation du corpus en cours..."):
            corpus_vecs = TP6.vectoriser_corpus(corpus_texte, vocab, methode=methode_vect, niveau=niveau, **params_vect)
            
        # 2. Vectorisation Requête
        # Prétraitement requête
        req_tokens = TP6.pretraiter_requete(query)
        
        # On adapte le type_requete pour vectoriser_requete
        type_req_map = {
            "document_mots": "document_mots",
            "document_agrege": "document_agrege",
            "phrase": "phrase"
        }
        
        vec_req = TP6.vectoriser_requete(req_tokens, vocab, methode=methode_vect, type_requete=type_req_map[niveau], **params_vect)
        
        # 3. Calcul Scores
        scores = TP6.calculer_scores_requete(vec_req, corpus_vecs, mesure=mesure_sim, niveau=niveau)
        
        # 4. Top K
        top_results = TP6.extraire_top_k(scores, k=top_k, niveau=niveau)
        
        # 5. Affichage
        st.markdown(f"### Résultats ({len(top_results)})")
        
        for identifiant, score in top_results.items():
            # Gestion ID Phrase vs Doc
            real_id = identifiant[0] if isinstance(identifiant, tuple) else identifiant
            suffixe = f" (Phrase {identifiant[1]})" if isinstance(identifiant, tuple) else ""
            
            doc_meta = meta.get(real_id, {})
            
            with st.expander(f"{doc_meta.get('titre', real_id)}{suffixe} - Score: {score:.4f}"):
                st.write(f"**Source:** {doc_meta.get('sous_corpus')} | **Langue:** {doc_meta.get('langue')}")
                
                # Affichage contenu
                texte_brut = TP6.afficher_contenu_document(identifiant, corpus_texte, niveau)
                # Highlight
                texte_high = TP6.highlight_mots_pertinents(texte_brut, req_tokens, vocab)
                
                # Streamlit ne supporte pas les codes ANSI de couleur directement, on fait un replace simple pour markdown
                # Le code TP6 utilise \033[91m...
                # On va faire un highlight manuel pour Streamlit
                for mot in req_tokens:
                    if mot in vocab:
                        texte_brut = texte_brut.replace(mot, f"**:red[{mot}]**")
                        texte_brut = texte_brut.replace(mot.capitalize(), f"**:red[{mot.capitalize()}]**")
                        
                st.markdown(texte_brut)

# --- Tab 2: Analyse Structurelle ---
with tab2:
    st.subheader("Analyse de Similarité Inter-Documents")
    
    if st.button("Générer Matrice de Similarité"):
        # On utilise une vectorisation TF simple niveau document pour l'analyse
        with st.spinner("Calcul de la matrice..."):
            corpus_vecs_struct = TP6.vectoriser_corpus(corpus_texte, vocab, methode="tf", niveau="document_mots")
            matrice, ids = TP6.matrice_similarite(corpus_vecs_struct, mesure="cosinus")
            
        st.write("Matrice calculée.")
        
        # Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(matrice, xticklabels=ids, yticklabels=ids, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top Paires Similaires**")
            pairs_sim = TP6.top_paires_similaires(matrice, ids, top=5)
            st.table(pd.DataFrame(pairs_sim, columns=["Doc A", "Doc B", "Score"]))
            
        with col2:
            st.markdown("**Top Paires Différentes**")
            pairs_diff = TP6.top_paires_differentes(matrice, ids, top=5)
            st.table(pd.DataFrame(pairs_diff, columns=["Doc A", "Doc B", "Score"]))

# --- Tab 3: Expérimentations ---
with tab3:
    st.subheader("Comparaisons et Expérimentations")
    
    exp_type = st.selectbox("Type d'expérience", ["Impact Prétraitement", "Comparaison Distances", "Local vs Global"])
    
    if exp_type == "Impact Prétraitement":
        st.info("Compare les résultats d'une requête selon différentes configurations de nettoyage.")
        req_exp = st.text_input("Requête test", value="etudiant")
        
        if st.button("Lancer Expérience Prétraitement"):
            configs = {
                "Minimal": {"stopwords": False, "stemming": False},
                "Stopwords": {"stopwords": True, "stemming": False},
                "Complet": {"stopwords": True, "stemming": True}
            }
            # Attention: TP6.tester_pretraitements attend un corpus brut (dict de listes de listes)
            # C'est ce qu'on a dans corpus_texte
            res = TP6.tester_pretraitements(req_exp, corpus_texte, configs)
            st.json(res)
            
    elif exp_type == "Comparaison Distances":
        st.info("Compare le classement Top-5 selon la métrique utilisée.")
        req_dist = st.text_input("Requête distance", value="etudiant")
        
        if st.button("Comparer Distances"):
            # Vectorisation préalable
            corpus_vecs_dist = TP6.vectoriser_corpus(corpus_texte, vocab, methode="tf", niveau="document_mots")
            req_tokens = TP6.pretraiter_requete(req_dist)
            vec_req = TP6.vectoriser_requete(req_tokens, vocab, methode="tf", type_requete="document_mots")
            
            res_dist = TP6.comparer_distances(corpus_vecs_dist, vec_req, mesures=["cosinus", "euclidienne", "manhattan", "jaccard"])
            
            df_dist = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in res_dist.items()]))
            st.dataframe(df_dist)

    elif exp_type == "Local vs Global":
        st.info("Compare les résultats sur un sous-corpus (ex: UFR) vs corpus entier.")
        
        sous_corpus_list = list(set(m['sous_corpus'] for m in meta.values()))
        target_sc = st.selectbox("Sous-corpus cible", sous_corpus_list)
        req_loc = st.text_input("Requête", value="etudiant")
        
        if st.button("Comparer Local / Global"):
            # Vectorisation
            corpus_vecs_all = TP6.vectoriser_corpus(corpus_texte, vocab, methode="tf", niveau="document_mots")
            
            # Extraction sous-corpus vecteurs
            corpus_vecs_loc = TP6.extraire_sous_corpus(corpus_vecs_all, meta, critere_valeur=target_sc)
            
            # Recherche
            top_loc = TP6.top_k_local(req_loc, corpus_vecs_loc, vocab, niveau_corpus="document_mots", top_k=5)
            top_glob = TP6.top_k_global(req_loc, corpus_vecs_all, vocab, niveau_corpus="document_mots", top_k=5)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Top Local")
                st.write(top_loc)
            with col2:
                st.write("Top Global")
                st.write(top_glob)
                
            # Analyse
            diff = TP6.comparer_local_vs_global(top_loc, top_glob)
            st.write("### Analyse Différentielle")
            st.json(diff)

# --- Tab 4: Statistiques Corpus ---
with tab4:
    st.subheader("Statistiques du Corpus")
    
    st.write(f"**Nombre total de documents :** {len(meta)}")
    st.write(f"**Taille du vocabulaire :** {len(vocab)}")
    
    # Répartition Langues
    langs = [m['langue'] for m in meta.values()]
    counts_lang = pd.Series(langs).value_counts()
    
    # Répartition Sources
    sources = [m['sous_corpus'] for m in meta.values()]
    counts_source = pd.Series(sources).value_counts()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Répartition par Langue**")
        st.bar_chart(counts_lang)
    with col2:
        st.write("**Répartition par Source**")
        st.bar_chart(counts_source)
        
    st.write("**Aperçu Métadonnées**")
    st.dataframe(pd.DataFrame.from_dict(meta, orient='index'))
