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


#Comprobamos algunos datos
"""
print(dataset.head(5))
print(dataset.shape)
print(dataset.index)
print(dataset.columns)
"""


