import streamlit as st
import TP2
import os
import io
from pathlib import Path
import tempfile

st.set_page_config(page_title="TP1 - Traitement de Texte", layout="wide")

st.title("TP1 - Interface de Test")
st.markdown("""
Cette interface permet de tester les fonctions du TP1 pour le traitement de texte.
Entrez un texte ou téléchargez un fichier pour commencer.
""")


# Input section
st.header("1. Entrée")
input_method = st.radio("Choisir la méthode d'entrée :", ["Texte direct", "Fichier"])

user_text = ""
uploaded_file = None

if input_method == "Texte direct":
    user_text = st.text_area("Entrez votre texte ici :", height=150, value="Ceci est un texte d'exemple avec des accents (é, à), du HTML <b>gras</b> et de la ponctuation !!!")
else:
    uploaded_file = st.file_uploader("Télécharger un fichier texte", type=["txt"])
    if uploaded_file is not None:
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        user_text = stringio.read()
        st.text_area("Contenu du fichier :", value=user_text, height=150, disabled=True)

if user_text:
    st.header("2. Traitements")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Standardisation", "Accents", "Ponctuation", "Langue"])

    with tab1:
        st.subheader("Standardisation")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Minuscules"):
                res = TP2.convertir_vers_minuscule(user_text)
                st.success(res)
        
        with col2:
            if st.button("Supprimer HTML/XML"):
                res = TP2.supprimer_balises_html_xml(user_text)
                st.success(res)
                
        with col3:
            if st.button("Normaliser Unicode (NFC)"):
                res = TP2.normaliser_unicode(user_text)
                st.success(res)

    with tab2:
        st.subheader("Accents")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Corriger Accents"):
                res = TP2.corriger_accents(user_text)
                st.success(res)
        
        with col2:
            if st.button("Uniformiser Accents"):
                res = TP2.uniformiser_accents(user_text)
                st.success(res)
        
        st.markdown("---")
        st.write("Pipeline complet :")
        opt_corriger = st.checkbox("Corriger erreurs", value=True)
        opt_uniformiser = st.checkbox("Uniformiser", value=True)
        if st.button("Traiter Accents (Options)"):
            options = {"corriger_erreurs": opt_corriger, "uniformiser": opt_uniformiser}
            res = TP2.traiter_accents(user_text, options)
            st.success(res)

    with tab3:
        st.subheader("Ponctuation")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Supprimer toute ponctuation"):
                st.success(TP2.supprimer_ponctuation(user_text))
            if st.button("Remplacer par <PONCT>"):
                st.success(TP2.remplacer_ponctuation(user_text))
            if st.button("Espacer ponctuation"):
                st.success(TP2.espacer_ponctuation(user_text))
            if st.button("Normaliser (guillemets, tirets...)"):
                st.success(TP2.normaliser_ponctuation(user_text))

        with col2:
            if st.button("Réduire répétitions (!!! -> !)"):
                st.success(TP2.reduire_ponctuation_multiple(user_text))
            if st.button("Remplacement contextuel (! -> <EXCLAMATION>)"):
                st.success(TP2.remplacer_ponctuation_contextuelle(user_text))
            
            st.markdown("**Supprimer sauf...**")
            keep_list = st.text_input("Caractères à garder (ex: . ?)", value=". ?")
            if st.button("Supprimer sauf liste"):
                chars = [c.strip() for c in keep_list.split()]
                st.success(TP2.supprimer_ponctuation_sauf(user_text, chars))

    with tab4:
        st.subheader("Langue")
        
        st.write("Analyse du contenu :")
        if st.button("Vérifier langue contenu"):
            lang = TP2.verifier_langue_contenu(user_text)
            st.info(f"Langue détectée : {lang}")

        if uploaded_file:
            st.markdown("---")
            st.write("Analyse basée sur le fichier :")
            # Save to temp file to use Path functions
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = Path(tmp.name)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Détecter langue (Nom fichier)"):
                    # We use the original filename for detection logic if possible, 
                    # but the function takes a Path object. 
                    # Let's mock a Path with the original name.
                    mock_path = Path(uploaded_file.name)
                    lang = TP2.detecter_langue_nom_fichier(mock_path)
                    st.info(f"Langue (Nom) : {lang}")
            
            with col2:
                if st.button("Vérifier cohérence"):
                    # For coherence, we need the file to exist and have the name.
                    # The temp file has a mangled name. 
                    # We can't easily use the existing function 'verifier_coherence_langue_contenu' 
                    # as is because it reads from the path provided.
                    # We will manually call the logic here for display.
                    
                    lang_nom = TP2.detecter_langue_nom_fichier(Path(uploaded_file.name))
                    lang_content = TP2.verifier_langue_contenu(user_text)
                    is_coherent = (lang_nom == lang_content) if (lang_nom != "inconnu" and lang_content != "indetermine") else True
                    
                    st.metric("Cohérence", "Oui" if is_coherent else "Non", delta=f"Nom: {lang_nom} / Contenu: {lang_content}")
            
            os.unlink(tmp_path)

        st.markdown("---")
        st.subheader("Analyse de dossier")
        dir_path = st.text_input("Chemin du dossier à analyser :", value=".")
        if st.button("Signaler incohérences (Dossier)"):
            p = Path(dir_path)
            if p.exists() and p.is_dir():
                anomalies = TP2.signaler_incoherences_langue(p)
                if anomalies:
                    st.error(f"{len(anomalies)} anomalie(s) détectée(s) :")
                    for a in anomalies:
                        st.write(a)
                else:
                    st.success("Aucune incohérence détectée.")
            else:
                st.error("Le chemin spécifié n'existe pas ou n'est pas un dossier.")

