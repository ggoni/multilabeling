import pandas as pd
import numpy as np
import streamlit as st
import pickle
import nltk
import re


etiquetas = ['Agilidad', 'Amabilidad/Trato/Disposicion', 'Asesoramiento', 'Caja',
             'Cobertura', 'Ofertas/Promociones/Canjes', 'Precio',
             'Producto(Atributos/Estado/Display)',
             'Variedad/Surtido/Stock/Disponibilidad', 'Pandemia',
             'Orden/Aseo/Seguridad(Tienda)', 'Otros']


def corrige(texto):

    traductor = {"palabras": "editada",
                 "acesoria": "asesoria",
                 "amavilidad": "amabilidad",
                 "ambilidad": "amabilidad",
                 "aser": "hacer",
                 "asesiria": "asesoria",
                 "atencio": "atencion",
                 "atencionn": "atencion",
                 "atension": "atencion",
                 "cariedad": "variedad",
                 "clinte": "cliente",
                 "cocid19": "COVID",
                 "corona viru": "COVID",
                 "cosiltar": "consultar",
                 "desrdenado": "desordenado",
                 "dispocion": "disposicion",
                 "elejir": "elegir",
                 "embarasadas": "embarazadas",
                 "espedito": "expedito",
                 "esxpedito": "expedito",
                 "excaso": "escaso",
                 "expredito": "expedito",
                 "exusas": "excusas",
                 "faborable": "favorable",
                 "fluides": "fluidez",
                 "hamabilidad": "amabilidad",
                 "lastencion": "atencion",
                 "limpiesa": "limpieza",
                 "mabilidsd": "amabilidad",
                 "nesecitaba": "necesitaba",
                 "nesecite": "necesite",
                 "ocacion": "ocasión",
                 "otden": "orden",
                 "pacillos": "pasillos",
                 "pandeoma": "pandemia",
                 "pasillod": "pasillos",
                 "pecio": "precio",
                 "personl": "personal",
                 "prasios": "precios",
                 "presio": "precio",
                 "presios": "precios",
                 "prevesion": "prevencion",
                 "productod": "productos",
                 "profuctos": "productos",
                 "rapide": "rapidez",
                 "rapidea": "rapidez",
                 "rapides": "rapidez",
                 "seguridas": "seguridad",
                 "seramicas": "ceramica",
                 "sircular": "circular",
                 "stok": "stock",
                 "valerio": "valoro",
                 "varato": "barato",
                 "variedady": "variedad",
                 "variwdad": "variedad",
                 "vendedotes": "vendedores",
                 "votados": "botados"}

    if not texto:
        return texto

    for key, value in traductor.items():
        texto = texto.replace(key, value)
    return texto


def reemplaza_letra(palabra):

    palabra = palabra.replace('&#225;', 'á')
    palabra = palabra.replace('&#193;', 'á')
    palabra = palabra.replace('&#233;', 'é')
    palabra = palabra.replace('&#201;', 'é')
    palabra = palabra.replace('&#237;', 'í')
    palabra = palabra.replace('&#205;', 'í')
    palabra = palabra.replace('&#243;', 'ó')
    palabra = palabra.replace('&#242;', 'ó')
    palabra = palabra.replace('&#211;', 'ó')
    palabra = palabra.replace('&#218;', 'ú')
    palabra = palabra.replace('&#249;', 'ú')
    palabra = palabra.replace('&#250;', 'ú')
    palabra = palabra.replace('&#241;', 'ñ')
    palabra = palabra.replace('&#209;', 'ñ')
    palabra = palabra.replace('&#252;', 'u')
    palabra = palabra.replace('&#176;', 'º')
    return palabra


def edita_texto(texto):
    texto = corrige(reemplaza_letra(texto))
    return texto


with open("multi_vectorizador.pckl", 'rb') as archivo_in:
    vectorizador = pickle.load(archivo_in)

with open("multi_clasificador.pckl", 'rb') as archivo_in:
    clasificador = pickle.load(archivo_in)


def predice_etiquetas_comentario(comentario):

    comentario = edita_texto(comentario)
    prediccion = clasificador.predict(vectorizador.transform([comentario]))
    prediccion = prediccion.reshape(-1).tolist()

    categorias = [etiquetas[i]
                  for i in range(len(prediccion)) if prediccion[i] == 1]

    return categorias


st.title('Clasificador de Feedback de Clientes')

st.markdown(
    '## Viene de la pregunta: _Cuéntanos, ¿por qué calificas con esa nota la experiencia en X?_')

st.subheader('1) Clasificar sólo Un comentario')

comentario_input = st.text_area('Comentario:')


y = ''

if st.button("Clasifica"):
    y = predice_etiquetas_comentario(comentario_input)
    st.markdown('_Las categorías más apropiadas son:_')
    
else:
    st.write("En espera")

  

for _, label in enumerate(y):

    st.markdown('\t\t\t\t\t\t\t\t\t'+'-_' + label + '_')
