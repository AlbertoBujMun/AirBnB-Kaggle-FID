#Numpy es la librería que utilizamos para la mayoría de operaciones y gestión de matrices
import numpy as np
#Pandas se utiliza para manipular y analizar datos. 
import pandas as pd
#Usamos el módulo pyplot de matplotlib para producir gráficos
import matplotlib.pyplot as pyplot
#Seaborn permite construir gráficos más visuales que pyplot, para los datos que podamos extraer para la presentación
import seaborn as sns
#Para lanzar ciertos warnings de código.
import warnings

warnings.filterwarnings('ignore')

#almacenamos dataset (No está dividido aún dado que es para estudio previo)
dataset = pd.read_csv('C:/Users/Alberto/Desktop/Master/Int. Datos/Proyecto/data/train_users_2.csv')


#Comprobamos algunos datos sobre el dataset
"""
print(dataset.head(5))
print(dataset.shape)
print(dataset.index)
print(dataset.columns)
"""


#Comprobamos valores vacíos en el dataset
print(dataset.isnull().sum())
print(dataset.shape)

#resultados
"""
date_first_booking         124543
age                         87990
first_affiliate_tracked      6065
"""

#Método dropna
#Si hay más de un 75% de valores nulos para un atributo, borra el atributo entero. 
#En otro caso, borraría la fila entera a la que perteneciese ese atributo.
dataset.dropna(inplace=True)
print(dataset.isnull().sum())
print(dataset.shape)
#Observamos que varía el tamaño del dataset (Se reducen muchísimos datos)
"""
Viendo que la columna que da más problemas es la fecha de la primera reserva, y que tiene más 
de un 50% por ciento de nulos pero no llega al 75% por lo que está haciendo perder demasiados datos,
yo la eliminaría igualmente. No me parece lo suficientemente relevante como para justificar la pérdida de tantos datos.
O eso, o se le hace un tratamiento distinto, como sustituir los valores.
"""
#Reiniciamos el dataset para más pruebas
dataset = pd.read_csv('C:/Users/Alberto/Desktop/Master/Int. Datos/Proyecto/data/train_users_2.csv')