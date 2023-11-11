import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import streamlit as st




# ******************** Affichage de la page de l'application ******************** #


# Titre page d'accueil :
st.write(''' Application pour classer des fleurs d'Iris ''')


# Barre de menu :
st.sidebar.header("Paramètres d'entrées")


# Définition les paramètres d'entrées :
def input():
    # inputs de la sidebar :
    sepal_length=st.sidebar.slider('longueur du Sépal', 4.3, 7.9, 5.3)
    sepal_width=st.sidebar.slider('largeur du Sépal', 2.0, 4.4, 3.3)
    petal_length=st.sidebar.slider('longueur du Pétale', 1.0, 6.9, 2.3)
    petal_width=st.sidebar.slider('largeur du Pétale', 0.1, 2.5, 1.3)
    # Récupération des valeurs sous forme de dictionnaire :
    data={
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    parametres_iris=pd.DataFrame(data, index=[0])
    return parametres_iris


# Afficher les paramètres :
df=input()
st.subheader('Paramètres de la fleur à catégoriser :')
st.write(df)




# ******************** Entrainement du modèle de Machine Learning ******************** #

# Chargement du set de données :
iris = datasets.load_iris()


# Entrainement du modèle :
clf=RandomForestClassifier()
clf.fit(iris.data, iris.target)




# ******************** Générer des prédictions ******************** #


# Calcul des prédictions :
prediction=clf.predict(df)


# Mise en pas de la prédiction :
st.subheader('Catégorie de la fleur :')
st.write(iris.target_names[prediction])


