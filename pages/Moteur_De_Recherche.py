import streamlit as st
import sys
from pathlib import Path
import time

# Ajout du chemin vers le dossier parent pour importer les modules App_TP*
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

import App_TP7 as TP7

st.set_page_config(page_title="TP7 - Moteur de Recherche", layout="wide")

# --- Initialisation de l'état (Session State) ---
if "corpus_indexe" not in st.session_state:
    st.session_state["corpus_indexe"] = False
if "meta_corpus" not in st.session_state:
    st.session_state["meta_corpus"] = {}
if "corpus_texte" not in st.session_state:
    st.session_state["corpus_texte"] = {}
if "vocabulaire" not in st.session_state:
    st.session_state["vocabulaire"] = {}
if "corpus_vecteurs" not in st.session_state:
    st.session_state["corpus_vecteurs"] = {}

# --- Titre et Description ---
st.title("🔍 Moteur de Recherche Vectoriel (TP7)")
st.markdown("""
Cette interface implémente un moteur de recherche d'information basé sur le modèle vectoriel (TF-IDF, Cosinus).
""")

# --- Sidebar : Indexation ---
with st.sidebar:
    st.header("⚙️ Configuration Indexation")
    
    reindex = st.button("Indexation du Corpus")
    
    st.divider()
    
    st.subheader("Options de Prétraitement")
    opt_stopwords = st.checkbox("Supprimer Stopwords", value=True)
    opt_stemming = st.checkbox("Stemming (racérisation)", value=False)
    opt_min_len = st.checkbox("Longueur min > 2", value=True)
    
    config_pretraitement = {
        "stopwords": opt_stopwords,
        "stemming": opt_stemming,
        "longueur_min": opt_min_len,
        "non_alphabetiques": True
    }
    
    st.divider()
    
    type_vect = st.selectbox("Niveau de Granularité", ["document_agrege", "document_mots", "phrase"], index=0)
    methode_ponderation = st.selectbox("Pondération", ["tfidf", "tf", "bow"], index=0)

# --- Fonction de Chargement et Indexation ---
def charger_et_indexer():
    chemin_textes = parent_dir / "Textes"
    
    if not chemin_textes.exists():
        st.error(f"Le dossier Textes n'a pas été trouvé à : {chemin_textes}")
        return

    with st.spinner("Analyse des fichiers..."):
        # 1. Méta-données
        meta = TP7.construire_meta_corpus(chemin_textes)
        st.session_state["meta_corpus"] = meta
        
        # 2. Chargement du contenu textuel
        corpus_brut = {}
        for id_doc, info in meta.items():
            chemin_fichier = Path(info["chemin"])
            try:
                txt = chemin_fichier.read_text(encoding="utf-8", errors="ignore")
                # Découpage basique en phrases (car corpus structure = liste de phrases)
                # On utilise une heuristique simple (. ? !)
                phrases_raw = txt.replace("!", ".").replace("?", ".").split(".")
                doc_contenu = []
                for p in phrases_raw:
                    mots = p.split()
                    if mots:
                        doc_contenu.append(mots)
                corpus_brut[id_doc] = doc_contenu
            except Exception as e:
                print(f"Erreur lecture {id_doc}: {e}")

        # 3. Prétraitement
        st.text("Prétraitement du corpus...")
        corpus_clean = TP7.pretraiter_corpus(corpus_brut, config_pretraitement)
        st.session_state["corpus_texte"] = corpus_clean
        
        # 4. Construction Vocabulaire
        st.text("Construction du vocabulaire...")
        vocab_set = set()
        for doc in corpus_clean.values():
            for phrase in doc:
                vocab_set.update(phrase)
        vocab = {mot: i for i, mot in enumerate(sorted(list(vocab_set)))}
        st.session_state["vocabulaire"] = vocab
        
        # 5. Vectorisation
        st.text("Vectorisation...")
        
        # Pour TF-IDF, on a besoin d'un vecteur IDF global (simulé ici ou calculé si dispo)
        # TP7.vectoriser_corpus attend vecteur_idf si methode="tfidf"
        # On va faire un calcul IDF simple si nécessaire
        vector_idf = None
        if methode_ponderation == "tfidf":
            # Calcul IDF simple : log(N / df)
            N = len(corpus_clean)
            df = {mot: 0 for mot in vocab}
            for doc in corpus_clean.values():
                mots_doc = set([m for p in doc for m in p])
                for m in mots_doc:
                    if m in df:
                        df[m] += 1
            import math
            vector_idf = [0.0] * len(vocab)
            for mot, idx in vocab.items():
                if df[mot] > 0:
                    vector_idf[idx] = math.log10(N / df[mot])
        
        corpus_vec = TP7.vectoriser_corpus(
            corpus_clean, 
            vocab, 
            methode=methode_ponderation, 
            niveau=type_vect,
            vecteur_idf=vector_idf
        )
        st.session_state["corpus_vecteurs"] = corpus_vec
        st.session_state["vecteur_idf"] = vector_idf # Stockage pour la requête
        st.session_state["corpus_indexe"] = True
        
    st.success(f"Indexation terminée ! {len(corpus_clean)} documents traités.")
    st.info(f"Taille du vocabulaire : {len(vocab)} mots.")

# Déclenchement Indexation
if reindex:
    charger_et_indexer()

# --- Interface de Recherche ---

st.header("🔎 Rechercher")

if not st.session_state["corpus_indexe"]:
    st.warning("Veuillez d'abord indexer le corpus via la barre latérale.")
else:
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Saisissez votre requête :", "intelligence artificielle")
    with col2:
        mesure_sim = st.selectbox("Mesure", ["cosinus", "euclidienne", "jaccard", "manhattan"])

    top_k_val = st.slider("Nombre de résultats (Top K)", 1, 20, 5)

    if st.button("Lancer la recherche"):
        # 1. Prétraitement requête (similaire au corpus)
        # On simule un mini corpus pour utiliser pretraiter_corpus
        dummy_corpus = {"req": [query.split()]} # Une phrase, split simple initial
        req_clean_dict = TP7.pretraiter_corpus(dummy_corpus, config_pretraitement)
        
        if not req_clean_dict or not req_clean_dict.get("req"):
            st.error("La requête est vide après prétraitement (stop words ?).")
        else:
            req_tokens = req_clean_dict["req"][0] # Liste de mots
            st.write(f"Tokens requête : `{req_tokens}`")
            
            # 2. Vectorisation Requête
            # Attention : pour TFIDF, il faut passer le vecteur IDF calculé sur le corpus
            try:
                vec_req = TP7.vectoriser_requete(
                    req_tokens, 
                    st.session_state["vocabulaire"], 
                    methode=methode_ponderation, 
                    type_requete="phrase", # La requête est vue comme une phrase
                    vecteur_idf=st.session_state.get("vecteur_idf")
                )
                
                # 3. Calcul Similarité
                scores = TP7.calculer_scores_requete(
                    vec_req, 
                    st.session_state["corpus_vecteurs"], 
                    mesure=mesure_sim, 
                    niveau=type_vect
                )
                
                # 4. Extraction Top K
                top_k = TP7.extraire_top_k(scores, k=top_k_val, niveau=type_vect)
                
                # 5. Affichage
                st.subheader(f"Résultats ({len(top_k)})")
                
                for identifiant, score in top_k.items():
                    # Gestion ID tuple (phrase) vs ID str (document)
                    if isinstance(identifiant, tuple):
                        real_id = identifiant[0]
                        suffixe = f" (Phrase n°{identifiant[1]})"
                    else:
                        real_id = identifiant
                        suffixe = ""
                        
                    meta = st.session_state["meta_corpus"].get(real_id, {})
                    
                    with st.expander(f"{meta.get('titre', real_id)}{suffixe} - Score: {score:.4f}"):
                        row1_col1, row1_col2 = st.columns(2)
                        with row1_col1:
                             st.caption(f"**Source**: {meta.get('sous_corpus')} | **Langue**: {meta.get('langue')}")
                        with row1_col2:
                             st.metric("Score", f"{score:.4f}")
                        
                        # Affichage extrait textuel
                        # On récupère le texte brut initial (non traité) ou traité ? 
                        # TP7.afficher_contenu_document utilise corpus_texte (qui est traité dans notre state)
                        # Pour l'affichage c'est mieux d'avoir le vrai texte, mais on ne l'a pas stocké brut-brut facile d'accès par phrase
                        # On va utiliser le corpus traité pour l'instant
                        extrait = TP7.afficher_contenu_document(identifiant, st.session_state["corpus_texte"], niveau=type_vect)
                        
                        # Highlight
                        if extrait:
                            st.text_area("Contenu (prétraité)", value=extrait, height=100, disabled=True)
                        else:
                            st.text("Pas de contenu disponible.")
                            
            except Exception as e:
                st.error(f"Erreur durant la recherche : {e}")

# --- Footer ---
st.markdown("---")
st.caption("TP7 - Master 1 IA - INFO0708")
