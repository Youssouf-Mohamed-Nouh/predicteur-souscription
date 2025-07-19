import pandas as pd
import streamlit as st
import joblib
from datetime import datetime
import numpy as np

st.set_page_config(initial_sidebar_state='expanded',page_title='Prédicteur Souscription Bancaire - Youssouf',page_icon='🎗️',layout='wide')

# source
@st.cache_resource
def charger_model():
    try:
        model = joblib.load('modelGDB.pkl')
        scaler = joblib.load('scaler.pkl')
        features = joblib.load('features.pkl')
        return model,scaler,features
    except FileNotFoundError as e:
        st.error(f'Erreur: Fichier manquant - {e}')
        st.stop()
    except Exception as e:
        st.error(f'Erreur lors de chargement :{e}')
        st.stop()
        
model,scaler,features = charger_model()

# en tete
st.markdown('''
            <style>
            .main-header{
               background: linear-gradient(135deg, #91BDF2 0%, #91BDF2 100%);
               padding:2.2rem;
               border-radius:50px;
               margin-bottom:2rem;
               text-align:center;
               border-shadow: 0 20px 50px rgba(0,0,0,0.1);
                }
            </style>
            ''',unsafe_allow_html=True)
st.markdown('''
            <div class='main-header'>
            <h1>🏦 Prédicteur Souscription Bancaire</h1>
            <p style='font-size:20px;'>Développé par - <strong>Youssouf</strong> Assistant Intelligent</p>
            </div>
            
            ''',unsafe_allow_html=True)
# sidbar
st.markdown('''
            <style>
            .friendly-info {
                background: #e3f2fd;
                padding: 2rem;
                border-radius: 15px;
                border-left: 5px solid #2196F3;
                margin: 1.5rem 0;
            }
            .encouragement {
            background: linear-gradient(135deg, #fff3e0, #ffecb3);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            border-left: 5px solid #ff9800;
        }
            </style>
            ''',unsafe_allow_html=True)
with st.sidebar:
    st.markdown("## 🤖 À propos de votre assistant")
    st.markdown("""
    <div class="friendly-info">
        <h4>Comment je fonctionne ?</h4>
        <p>• J'utilise un modèle d'IA entraîné sur des milliers de cas</p>
        <p>• Ma précision est d'environ 97%</p>
        <p>• Je respecte votre vie privée</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 💡 Rappel important")
    st.markdown("""
    <div class="encouragement">
        <p><strong>Gardez en tête :</strong></p>
        <p>✨ Je suis un outil d'aide, pas un agent de la banque</p>
    </div>
    """, unsafe_allow_html=True)

# formulaire
st.markdown('''
            <h2 style='color:#343a40;text-align:center;margin-bottom:25px'> 📋 Informations du clients</h2>
            
            ''',unsafe_allow_html=True)
with st.form('Formulaire-client'):
    st.markdown('### 👤 Informations personnelles')
    col1,col2 = st.columns(2)
    with col1:
        with col1:
            nom_client = st.text_input(
                '📝 Votre nom complet du client', 
                placeholder="Ex: Youssouf Mohamed ",
                help="Saisissez le nom complet du client"
            ).strip()
            
            age_client = st.slider(
                '🎂 Votre âge', 
                18, 100, 30,
                help="L'âge du client"
            )
            solde_compte = st.slider(
                '💲 Solde du compte ($)', min_value=-1800.0,
                max_value=5170.0,value=1200.0,step=1.0,
                help="Solde actuel du compte bancaire "
            )
        with col2:
            profession = st.selectbox('💼 Prefession',options=['Sans emploi','Employe de service',
              'Cadre/Manager','Ouvrir','Travailleur independant','Technicien','Entrepreneur','Administratif',
              'Etudiant','Femme de menage','Retraite','Inconnu'],help='Profession principal du client')
            etat_civil = st.selectbox('💑 État civil',options=['Marie',
                'Celibataire','Divorce'],help='Situation matrimoniale actuelle'
                
                )
            Education = st.selectbox('🎓 Education',options=['Primaire',
                'Secondaire','Superieur','Inconnu'],help='Niveau plus haut d\'éducation atteint'
                
                )
        st.markdown('### 💳 Historique financier')
        col3,col4 = st.columns(2)
        with col3:
            Defaut_credit = st.radio('❌ Défaut crédit ?',options=['Non','Oui'],
                                     help='Le client a-t-il déja eu un défaut de paiment ?')
            
            Credit_immobilier = st.radio('🏠 Crédit immobilier en cours.. ?',options=['Non','Oui'],
                                     help='Le client Crédit immobilier actifs ?')
        with col4:
            Pret_personnel = st.radio('💼 Prêt personne en cours ?',options=['Non','Oui'],
                                     help='Le client a-t-il  un prêt personne actifs ?')
            
        st.markdown('### 📞 Détails de la campagne')   
        col5,col6 = st.columns(2)
        with col5:
            jour_contact = st.slider('Jour du contact',min_value=1,max_value=31,value=5,step=1,
                                     help='Jour de mois où le contact a été établi')
            duree_contact = st.slider('⏰ Durée du dernier contact téléphonique',min_value=3.0,max_value=1470.0,value=5.0,step=0.5,
                                     help='Durée du contact téléphonique')
            campagne_contact = st.slider('📊 Nombre de contact cette campagne',min_value=1,max_value=7,value=5,step=1,
                                     help='Nombre de fois que le client a été contacté durant cette campagne')
        with col6:
            type_contact = st.selectbox('📱 Type de contact',options=['Cellulaire',
            'Inconnu','téléphone fixe'],help='Moyen de communication utilisé')
            mois_contact = st.selectbox(' 🗓️ Mois du  contact',options=['oct', 'may', 'apr', 'jun', 'feb', 'aug', 'jan', 'jul', 'nov',
                   'sep', 'mar', 'dec'],help='Mois où le contact a été établi')
            Jours_dernier_contact = st.slider('📅 Jour depuis le dernier contact ',min_value=-1.0,max_value=250.0,value=5.0,step=1.0,
                                 help='Nombre de jour depuis le dernier contact (-1 si jamais contacté)')
        st.markdown('### 📈 Historique des campagnes')
        col7,col8 = st.columns(2)
        with col7:
            Contacts_precedents = st.slider('📞 Contact lors de campagnes précédentes ',min_value=0.0,max_value=5.0,value=2.0,step=1.0,
                                 help='Nombre total de contact lors des campagnes précédentes')
        with col8:
            Resultat_campagne = st.selectbox('📊  Résultat de la campagne précédente',options=['Inconnu',
                'Echec','Autre','Succes'],help='Résultat de la dernière campagne marketing'
                
                )
    st.markdown('---')
    col_center = st.columns([1,2,1])
    with col_center[1]:
        submit = st.form_submit_button(
            'Prédire la probabilité de souscription', 
            type="primary", 
            use_container_width=True
        )

if submit:
    if not nom_client:
        st.warning('Veuillez renseigner le nom complet du client !')
    else:
        donnees_client = {colonne: 0 for colonne in features}
       # donnee numerique
        donnees_client['Age'] = age_client
        donnees_client['solde'] = solde_compte
        donnees_client['jour'] = jour_contact
        donnees_client['Duree'] = duree_contact
        donnees_client['campagne'] = campagne_contact
        donnees_client['Jours_dernier_contact'] = jour_contact
        donnees_client['Contacts_precedents'] = Contacts_precedents
        # encodage
        colonne_prof = f'Profession_{profession}'
        if colonne_prof in donnees_client:
            donnees_client[colonne_prof] = 1
            
        colonne_civil = f'Etat_civil_{etat_civil}'
        if colonne_prof in donnees_client:
            donnees_client[colonne_civil] = 1
            
        colonne_edu = f'Education_{Education}'
        if colonne_prof in donnees_client:
            donnees_client[colonne_edu] = 1
        
       # binaire
        if Defaut_credit == 'Oui':
            donnees_client['Defaut_credit_Oui'] = 1
            
        if Credit_immobilier == 'Oui':
            donnees_client['Credit_immobilier_Oui'] = 1
        
        if Pret_personnel == 'Oui':
            donnees_client['Pret_personnel_Oui'] = 1
        # type de contact
        colonne_contact = f'Contact_{type_contact}'
        if colonne_contact in donnees_client:
            donnees_client[colonne_contact] = 1
        colonne_mois = f'Mois_{mois_contact}'   
        if colonne_contact in donnees_client:
            donnees_client[colonne_mois] = 1
        colonne_resuls = f'Resultat_campagne_{Resultat_campagne}'   
        if colonne_resuls in donnees_client:
            donnees_client[colonne_resuls] = 1
            
        # creation data
        nouvelle_donnee = pd.DataFrame([
            [donnees_client[col] for col in features]
            ],columns=features)    
        try:
            donnee_normalise = scaler.transform(nouvelle_donnee)
            prediction = model.predict(donnee_normalise)[0]
            proba = model.predict_proba(donnee_normalise)[0][1]
            st.markdown('---')
            st.markdown(f"""
            <div class="friendly-info">
                <h2>Résultat de l\'analyse pour {nom_client}</h2>
            </div>
            """, unsafe_allow_html=True)
            if prediction > 0.5:
                st.success(f'✅ **Prédiction positive** :{nom_client} a de forte chnaces de souscrire au produit bancaire ! ')
                conseil = '**Recommadation** : Contactez ce client rapidement , il présente un profil trés  favorable'
            else:
                st.warning(f'❌ **Prédiction Négative** :{nom_client} a peu  de chnaces de souscrire au produit bancaire ! ')
                conseil = '**Recommadation** : ce client ne néccesite un approche commercial adapté ou cible'
            col_prob1,col_prob2 = st.columns([1,2])
            with col_prob1:
                delta_proba = float(np.round(proba - 0.5, 2))
                st.metric(
                    label='🎯 Probabilité de souscription',
                    value =f'{proba:.1%}',
                    delta=delta_proba
                    )
                st.caption('Différence par rapport à une moyenne de 50 %')
            with col_prob2:
                couleur_barre = "#28a745" if proba > 0.5 else "#dc3545"
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px;">
                    <h5 style="margin-bottom: 10px; color: #495057;">Niveau de confiance</h5>
                    <div style="background-color: #e9ecef; border-radius: 25px; height: 20px; overflow: hidden;">
                        <div style="width: {proba*100}%; height: 100%; background-color: {couleur_barre}; 
                                   border-radius: 25px; transition: width 0.3s ease;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.info(conseil)
        except Exception as e:
             st.error(f"❌ Erreur lors de la prédiction : {str(e)}")
             st.info("💡 Veuillez vérifier que tous les champs sont correctement remplis.")     
         
# Message de conclusion plus chaleureux
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2.5rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 20px; margin-top: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <h4 style="color: #495057; margin-bottom: 1rem;">🏦' Votre Assistant Intelligente</h4>
    <p style="font-size: 1em; color: #6c757d; margin-bottom: 0.5rem;">
        Créé avec passion par <strong>Youssouf</strong> pour vous accompagner dans votre parcours santé
    </p>
    <p style="font-size: 0.9em; color: #6c757d; margin-bottom: 1rem;">
        Version 2024 - Mis à jour régulièrement pour votre bien-être
    </p>
    <div style="border-top: 1px solid #dee2e6; padding-top: 1rem;">
        <p style="font-size: 0.85em; color: #6c757d; font-style: italic;">
            ⚠️ Rappel important : Cet outil d'aide à la décision complète mais ne remplace jamais 
            l'expertise de votre agent
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

    
