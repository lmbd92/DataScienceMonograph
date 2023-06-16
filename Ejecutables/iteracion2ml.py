# -*- coding: utf-8 -*-
"""iteracion2ML.ipynb

# Iteración 2

**Clasificación y regresión con técnicas ML:**


*   Árboles
*   Random Forest
*   SVM
"""
print("Inicio Ejecución iteración 2")

# Tratamiento de datos
# ==============================================================================
import warnings
import pandas as pd
import numpy as np


# Matemáticas y estadísticas
# ==============================================================================
import math
from sklearn.svm import LinearSVC
from sklearn import svm

# Preparación de datos
# ==============================================================================
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_validate
import time
from sklearn.metrics import mean_squared_error, r2_score
from pylab import *
from pandas import DataFrame
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_validate
from statistics import mean
from sklearn.linear_model import LinearRegression

#Creación de modelo
from sklearn.svm import SVR


# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Configuración matplotlib
# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
# plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

# Configuración warnings
# ==============================================================================
warnings.filterwarnings('ignore')


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.svm import SVR

"""## 1. Carga de datos

Se usará el DF limpio y transformado en la primer iteración
"""

url = 'https://raw.githubusercontent.com/lmbd92/DataScienceMonograph/main/Data/raw-files/online_retail_II_limpio.csv'
df = pd.read_csv(url)
df.head()

df.info()

df.describe()

"""Generación de nueva variable numérica "Tipo_cliente":


*   1: Mayorista
*   0: Minorista

La condición estará dada por la variable "TotalQuantity", donde los registros menores e iguales a 100 unidades se etiquetará como minorista y mayores a 100 como mayoristas.


"""

# Distriución de la variable TotalQuantity

df_TotalQuantity = df.groupby('Invoice').TotalQuantity.min().sort_values(ascending=False)

# Generar el histograma
plt.hist(df_TotalQuantity, bins=10)  # Puedes ajustar el número de 'bins' según tus necesidades

# Configurar etiquetas y título
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Histograma de la variable')

# Mostrar el histograma
plt.show()

# Agrupación arbitraria

# Definir una función que asigna el valor de 'tipo_cliente' según la condición
def asignar_tipo_cliente(row):
    if row['TotalQuantity'] <= 100:
        return 0
    else:
        return 1

# Aplicar la función a cada fila del DataFrame y crear la nueva variable 'tipo_cliente'
df['tipo_cliente'] = df.apply(asignar_tipo_cliente, axis=1)

# Verificar los resultados
print(df['tipo_cliente'])

# Distriución de la variable tipo_cliente

df.groupby('tipo_cliente').tipo_cliente.count().sort_values(ascending=False)

# Obtener los valores únicos de tipo_cliente y su frecuencia
tipo_cliente = df['tipo_cliente'].unique()
frecuencia = df['tipo_cliente'].value_counts().values

# Etiquetas para los sectores de la torta
labels = tipo_cliente

# Crear la gráfica de torta
plt.pie(frecuencia, labels=labels, autopct='%1.1f%%', labeldistance=1.1)

# Título de la gráfica
plt.title('Distribución de la variable tipo_cliente')

print(f"[tipo cliente: {tipo_cliente[0]} cantidad: {frecuencia[0]}] - [tipo cliente: {tipo_cliente[1]} cantidad: {frecuencia[1]}]")

# Mostrar la gráfica
plt.show()

df.head(20)

"""## 2. Visualización de datos

### Variables de entrada

Visualización de una muestra representativa de los datos, dada las limitaciones de procesamiento
"""

# DF de muestreo con todos los tipos de clientes
df_sample_all = df.sample(n=10000)  # Change 'n' to the desired sample size

df_sample_all.head()

# DF de muestreo solo con clientes de tipo mayorista
df_sample_mayorista = df.loc[df['tipo_cliente'] == 1].sample(n=10000)  # Change 'n' to the desired sample size

df_sample_mayorista.head()

# DF de muestreo solo con clientes de tipo minorista
df_sample_minorista = df.loc[df['tipo_cliente'] == 0].sample(n=10000)  # Change 'n' to the desired sample size

df_sample_minorista.head()

# Definición variables de entrada y variable de salida

df_inputs= df.drop(columns=['TotalTransaction'])

df_output = df[['TotalTransaction']].copy()

df_inputs.head(10)

df_output.head(10)

"""### Variable de salida"""

# Distriución de la variable de salida

df.groupby('TotalTransaction').TotalTransaction.count().sort_values(ascending=False)

# Se visualiza la variable de salida
plt.hist(df['TotalTransaction'], bins=50)
plt.xlabel('TotalTransaction')
plt.ylabel('Count')
plt.title('TotalTransaction distribution histogram')
plt.show()

# Visualización de variables temporales versus variable objetivo "TotalTransaction"
sns.pairplot(df_sample_all[['Year','Months', 'Month_day','Time_hour','wk_day','year_month','TotalTransaction']])

# Visualización de variables geograficas versus variable objetivo "TotalTransaction"
sns.pairplot(df_sample_all[['Quantity','Price','TotalSpent','TotalProductosUnicos','TotalQuantity','TotalTransaction']])

#Histográma y caja de bigotes del TotalTransaction

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.title('TotalTransaction Distribution Plot')
sns.distplot(df_sample_all.Price)

plt.subplot(1,2,2)
plt.title('TotalTransaction Spread')
sns.boxplot(x=df_sample_all.Price)

plt.show()

"""## 3. Creación del modelo (Todos los clientes)"""

# Definición variables de entrada y variable de salida

df_inputs_all= df_sample_all.drop(columns=['TotalTransaction'])

df_output_all = df_sample_all[['TotalTransaction']].copy()

"""### 3.1 Arboles Simples Variables"""

# Creación de variables para el test y el train

TiempoEntrenamientoTRAIN=[]
TiempoEvaluacionTRAIN=[]
RMSETrain=[]
R2Train=[]
MSETrain=[]
MAETrain=[]
nameColum=[]

TiempoEntrenamientoTEST=[]
TiempoEvaluacionTEST=[]
RMSETest=[]
R2Test=[]
MSETest=[]
MAETest=[]

import time
start_time = time.time()

print('-------------------INICIO PROCESAMIENTO-----------------')

Cantidad = 20

for i in range(Cantidad):

  # Hyperparameters.
  max_depth = 10
  minsamplesplit = 10
  minsampleleaf = 10

  param_grid = {"model__max_depth": list(range(1, max_depth + 1)),
                "model__min_samples_split": list(range(1, minsamplesplit + 1)),
                "model__min_samples_leaf": list(range(1, minsampleleaf + 1))}

  #param_grid = {"model__min_samples_split": list(range(1, minsamplesplit + 1)),
  #              "model__min_samples_leaf": list(range(1, minsampleleaf + 1))}

  # Loading the data.
  train_features = df_inputs_all.iloc[:,i]
  targets = df_output_all.iloc[:,0]

  NameVariable = df_inputs_all.columns[i]

  # Train, test split.
  data_split = train_test_split(train_features, targets, test_size=0.3, random_state=4444)
  X_train, X_test, targets_train, targets_test = data_split

  # Getting the target.
  y_train = targets_train
  y_test = targets_test

  # Model definition.
  pipe = Pipeline([("minmax", MinMaxScaler((-1, 1))),
                    ("model", DecisionTreeRegressor(random_state=4444))])
  # By default GridSearchCV uses a 5-kfold validation strategy.
  search = GridSearchCV(pipe, param_grid, cv=7, n_jobs=-1)
  search.fit(X_train.values.reshape(-1, 1), y_train.values)

  # Getting the test score.
  y_hat_test = search.predict(X_test.values.reshape(-1, 1))
  test_score = r2_score(y_test.values, y_hat_test)

  # Printing stats.
  print(f"Columna: {NameVariable}")
  print(f"Best CV score: {search.best_score_:0.3f}")
  print(f"Best Parameters:\n {search.best_params_}")
  print(f"Test score: {test_score:0.3f}")
  print("")

print('--------------------------------------------------------------')
print('PROCESAMIENTO FINALIZADO EXITOSAMENTE!!!')
print('--------------------------------------------------------------')

TIEMPO = (time.time() - start_time)/60

# print("--- %s Segundos ---" % (time.time() - start_time))
print("--- %s Minutos ---" % (TIEMPO))

"""### 3.1 Arboles Multiples Variables"""

import time
start_time = time.time()

print('-------------------INICIO PROCESAMIENTO-----------------')

# Hyperparameters.
max_depth = 10
minsamplesplit = 10
minsampleleaf = 10

param_grid = {"model__max_depth": list(range(1, max_depth + 1)),
               "model__min_samples_split": list(range(1, minsamplesplit + 1)),
               "model__min_samples_leaf": list(range(1, minsampleleaf + 1))}

#param_grid = {"model__min_samples_split": list(range(1, minsamplesplit + 1)),
#              "model__min_samples_leaf": list(range(1, minsampleleaf + 1))}

# Loading the data.
train_features = df_inputs_all
targets = df_output_all

# Train, test split.
data_split = train_test_split(train_features, targets, test_size=0.3, random_state=4444)
X_train, X_test, targets_train, targets_test = data_split

# Find the best model per target.
for target in range(targets.shape[1]):
    # Getting the target.
    y_train = targets_train.iloc[:, target]
    y_test = targets_test.iloc[:, target]

    # Model definition.
    pipe = Pipeline([("minmax", MinMaxScaler((-1, 1))),
                      ("model", DecisionTreeRegressor(random_state=4444))])
    # By default GridSearchCV uses a 5-kfold validation strategy.
    search = GridSearchCV(pipe, param_grid, cv=7, n_jobs=-1)
    search.fit(X_train.values, y_train.values)

    # Getting the test score.
    y_hat_test = search.predict(X_test.values)
    test_score = r2_score(y_test.values, y_hat_test)

    # Printing stats.
    print(f"Columna: {targets.columns[target]}")
    print(f"Best CV score: {search.best_score_:0.3f}")
    print(f"Best Parameters:\n {search.best_params_}")
    print(f"Test score: {test_score:0.3f}")
    print("")

print('--------------------------------------------------------------')
print('PROCESAMIENTO FINALIZADO EXITOSAMENTE!!!')
print('--------------------------------------------------------------')

TIEMPO = (time.time() - start_time)/60

# print("--- %s Segundos ---" % (time.time() - start_time))
print("--- %s Minutos ---" % (TIEMPO))

"""### 3.2 Bosques Simples Variables"""

import time
start_time = time.time()

print('-------------------INICIO PROCESAMIENTO-----------------')

Cantidad = 20

 # Hyperparameters.
param_gride = {"model__max_depth": [2,3,4],
              "model__min_samples_split": [2,4],
              "model__min_samples_leaf": [2,4],
              "model__n_estimators": [150]}

for i in range(Cantidad):

   # Loading the data.
  train_features = df_inputs_all.iloc[:,i]
  targets = df_output_all.iloc[:,0]

  NameVariable = df_inputs_all.columns[i]

  # Train, test split.
  data_split = train_test_split(train_features, targets, test_size=0.3, random_state=4444)
  X_train, X_test, targets_train, targets_test = data_split

  # Getting the target.
  y_train = targets_train
  y_test = targets_test

  # Model definition.
  pipe = Pipeline([("minmax", MinMaxScaler((-1, 1))),
                    ("model", RandomForestRegressor(random_state=4444))])
  # By default GridSearchCV uses a 5-kfold validation strategy.
  search = GridSearchCV(pipe, param_gride, cv=7, n_jobs=-1)
  search.fit(X_train.values.reshape(-1, 1), y_train.values)

  # Getting the test score.
  y_hat_test = search.predict(X_test.values.reshape(-1, 1))
  test_score = r2_score(y_test.values, y_hat_test)

  # Printing stats.
  print(f"Columna: {NameVariable}")
  print(f"Best CV score: {search.best_score_:0.3f}")
  print(f"Best Parameters:\n {search.best_params_}")
  print(f"Test score: {test_score:0.3f}")
  print("")

print('--------------------------------------------------------------')
print('PROCESAMIENTO FINALIZADO EXITOSAMENTE!!!')
print('--------------------------------------------------------------')

TIEMPO = (time.time() - start_time)/60

# print("--- %s Segundos ---" % (time.time() - start_time))
print("--- %s Minutos ---" % (TIEMPO))

"""### 3.2 Bosques Multiples Variables"""

import time
start_time = time.time()

print('-------------------INICIO PROCESAMIENTO-----------------')

# Hyperparameters.

#param_gride = {"model__max_depth": [2,3],
#              "model__min_samples_split": [2,3],
#              "model__min_samples_leaf": [2,3],
#              "model__n_estimators": [50]}

param_gride = {"model__max_depth": [5,10],
              "model__min_samples_split": [2,3,4,5],
              "model__min_samples_leaf": [2,3,4,5],
              "model__n_estimators": [150]}


# Loading the data.
train_features = df_inputs_all
targets = df_output_all

# Train, test split.
data_split = train_test_split(train_features, targets, test_size=0.3, random_state=4444)
X_train, X_test, targets_train, targets_test = data_split

# Find the best model per target.
for target in range(targets.shape[1]):
    # Getting the target.
    y_train = targets_train.iloc[:, target]
    y_test = targets_test.iloc[:, target]

    # Model definition.
    pipe = Pipeline([("minmax", MinMaxScaler((-1, 1))),
                      ("model", RandomForestRegressor(random_state=4444,n_jobs=-1))])
    # By default GridSearchCV uses a 5-kfold validation strategy.
    search = GridSearchCV(estimator=pipe, param_grid = param_gride, cv=7, n_jobs=-1)
    search.fit(X_train.values, y_train.values)

    # Getting the test score.
    y_hat_test = search.predict(X_test.values)
    test_score = r2_score(y_test.values, y_hat_test)

    # Printing stats.
    print(f"Columna: {targets.columns[target]}")
    print(f"Best CV score: {search.best_score_:0.3f}")
    print(f"Test score: {test_score:0.3f}")
    print(f"Best Parameters:\n {search.best_params_}")
    print("")

print('--------------------------------------------------------------')
print('PROCESAMIENTO FINALIZADO EXITOSAMENTE!!!')
print('--------------------------------------------------------------')

TIEMPO = (time.time() - start_time)/60

# print("--- %s Segundos ---" % (time.time() - start_time))
print("--- %s Minutos ---" % (TIEMPO))

"""### 3.3 SVM Simples Variables"""

import time
start_time = time.time()

print('-------------------INICIO PROCESAMIENTO-----------------')

Cantidad = 20

# Hyperparameters.

#param_gride = {"model__kernel": ['poly'],
#              "model__gamma": ['scale'],
#              "model__degree": [1],
#              "model__epsilon": [0.0001],
#              "model__C":  [1]}

param_gride = {"model__kernel": ['poly','rbf'],
              "model__gamma": ['scale','auto'],
              "model__degree": [1,2],
              "model__epsilon": [0.01, 0.1],
              "model__C":  [1]}

for i in range(Cantidad):

   # Loading the data.
  train_features = df_inputs_all.iloc[:,i]
  targets = df_output_all.iloc[:,0]

  NameVariable = df_inputs_all.columns[i]

  # Train, test split.
  data_split = train_test_split(train_features, targets, test_size=0.3, random_state=4444)
  X_train, X_test, targets_train, targets_test = data_split

  # Getting the target.
  y_train = targets_train
  y_test = targets_test

  # Model definition.
  pipe = Pipeline([("minmax", MinMaxScaler((-1, 1))),
                    ("model", SVR())])

  # By default GridSearchCV uses a 5-kfold validation strategy.
  search = GridSearchCV(pipe, param_gride, cv=7, n_jobs=-1)
  search.fit(X_train.values.reshape(-1, 1), y_train.values)

  # Getting the test score.
  y_hat_test = search.predict(X_test.values.reshape(-1, 1))
  test_score = r2_score(y_test.values, y_hat_test)

  # Printing stats.
  print(f"Columna: {NameVariable}")
  print(f"Best CV score: {search.best_score_:0.3f}")
  print(f"Best Parameters:\n {search.best_params_}")
  print(f"Test score: {test_score:0.3f}")
  print("")

print('--------------------------------------------------------------')
print('PROCESAMIENTO FINALIZADO EXITOSAMENTE!!!')
print('--------------------------------------------------------------')

TIEMPO = (time.time() - start_time)/60

# print("--- %s Segundos ---" % (time.time() - start_time))
print("--- %s Minutos ---" % (TIEMPO))

"""### 3.3 SVM Multiples Variables"""

import time
start_time = time.time()

print('-------------------INICIO PROCESAMIENTO-----------------')

# Hyperparameters.

#param_gride = {"model__kernel": ['poly'],
#              "model__gamma": ['scale'],
#              "model__degree": [1],
#              "model__epsilon": [0.0001],
#              "model__C":  [1]}

param_gride = {"model__kernel": ['poly','rbf'],
              "model__gamma": ['scale','auto'],
              "model__degree": [1,2,3,4,5],
              "model__epsilon": [0.001,0.01, 0.1],
              "model__C":  [1, 3, 5]}

# Loading the data.
train_features = df_inputs_all
targets = df_output_all

# Train, test split.
data_split = train_test_split(train_features, targets, test_size=0.3, random_state=4444)
X_train, X_test, targets_train, targets_test = data_split

# Find the best model per target.
for target in range(targets.shape[1]):
    # Getting the target.
    y_train = targets_train.iloc[:, target]
    y_test = targets_test.iloc[:, target]

    # Model definition.
    pipe = Pipeline([("minmax", MinMaxScaler((-1, 1))),
                      ("model", SVR())])
    # By default GridSearchCV uses a 5-kfold validation strategy.
    search = GridSearchCV(estimator=pipe, param_grid = param_gride, cv=7, n_jobs=-1)
    search.fit(X_train.values, y_train.values)

    # Getting the test score.
    y_hat_test = search.predict(X_test.values)
    test_score = r2_score(y_test.values, y_hat_test)

    # Printing stats.
    print(f"Columna: {targets.columns[target]}")
    print(f"Best CV score: {search.best_score_:0.3f}")
    print(f"Test score: {test_score:0.3f}")
    print(f"Best Parameters:\n {search.best_params_}")
    print("")

print('--------------------------------------------------------------')
print('PROCESAMIENTO FINALIZADO EXITOSAMENTE!!!')
print('--------------------------------------------------------------')

TIEMPO = (time.time() - start_time)/60

# print("--- %s Segundos ---" % (time.time() - start_time))
print("--- %s Minutos ---" % (TIEMPO))

"""### 3. Creación del modelo (Mayoristas)"""

# Definición variables de entrada y variable de salida

df_inputs_all= df_sample_mayorista.drop(columns=['TotalTransaction'])

df_output_all = df_sample_mayorista[['TotalTransaction']].copy()

"""### 3.1 Arboles Simples Variables"""

import time
start_time = time.time()

print('-------------------INICIO PROCESAMIENTO-----------------')

Cantidad = 20

for i in range(Cantidad):

  # Hyperparameters.
  max_depth = 10
  minsamplesplit = 10
  minsampleleaf = 10

  param_grid = {"model__max_depth": list(range(1, max_depth + 1)),
                "model__min_samples_split": list(range(1, minsamplesplit + 1)),
                "model__min_samples_leaf": list(range(1, minsampleleaf + 1))}

  #param_grid = {"model__min_samples_split": list(range(1, minsamplesplit + 1)),
  #              "model__min_samples_leaf": list(range(1, minsampleleaf + 1))}

  # Loading the data.
  train_features = df_inputs_all.iloc[:,i]
  targets = df_output_all.iloc[:,0]

  NameVariable = df_inputs_all.columns[i]

  # Train, test split.
  data_split = train_test_split(train_features, targets, test_size=0.3, random_state=4444)
  X_train, X_test, targets_train, targets_test = data_split

  # Getting the target.
  y_train = targets_train
  y_test = targets_test

  # Model definition.
  pipe = Pipeline([("minmax", MinMaxScaler((-1, 1))),
                    ("model", DecisionTreeRegressor(random_state=4444))])
  # By default GridSearchCV uses a 5-kfold validation strategy.
  search = GridSearchCV(pipe, param_grid, cv=7, n_jobs=-1)
  search.fit(X_train.values.reshape(-1, 1), y_train.values)

  # Getting the test score.
  y_hat_test = search.predict(X_test.values.reshape(-1, 1))
  test_score = r2_score(y_test.values, y_hat_test)

  # Printing stats.
  print(f"Columna: {NameVariable}")
  print(f"Best CV score: {search.best_score_:0.3f}")
  print(f"Best Parameters:\n {search.best_params_}")
  print(f"Test score: {test_score:0.3f}")
  print("")

print('--------------------------------------------------------------')
print('PROCESAMIENTO FINALIZADO EXITOSAMENTE!!!')
print('--------------------------------------------------------------')

TIEMPO = (time.time() - start_time)/60

# print("--- %s Segundos ---" % (time.time() - start_time))
print("--- %s Minutos ---" % (TIEMPO))

"""### 3.1 Arboles Multiples Variables"""

import time
start_time = time.time()

print('-------------------INICIO PROCESAMIENTO-----------------')

# Hyperparameters.
max_depth = 10
minsamplesplit = 10
minsampleleaf = 10

param_grid = {"model__max_depth": list(range(1, max_depth + 1)),
               "model__min_samples_split": list(range(1, minsamplesplit + 1)),
               "model__min_samples_leaf": list(range(1, minsampleleaf + 1))}

#param_grid = {"model__min_samples_split": list(range(1, minsamplesplit + 1)),
#              "model__min_samples_leaf": list(range(1, minsampleleaf + 1))}

# Loading the data.
train_features = df_inputs_all
targets = df_output_all

# Train, test split.
data_split = train_test_split(train_features, targets, test_size=0.3, random_state=4444)
X_train, X_test, targets_train, targets_test = data_split

# Find the best model per target.
for target in range(targets.shape[1]):
    # Getting the target.
    y_train = targets_train.iloc[:, target]
    y_test = targets_test.iloc[:, target]

    # Model definition.
    pipe = Pipeline([("minmax", MinMaxScaler((-1, 1))),
                      ("model", DecisionTreeRegressor(random_state=4444))])
    # By default GridSearchCV uses a 5-kfold validation strategy.
    search = GridSearchCV(pipe, param_grid, cv=7, n_jobs=-1)
    search.fit(X_train.values, y_train.values)

    # Getting the test score.
    y_hat_test = search.predict(X_test.values)
    test_score = r2_score(y_test.values, y_hat_test)

    # Printing stats.
    print(f"Columna: {targets.columns[target]}")
    print(f"Best CV score: {search.best_score_:0.3f}")
    print(f"Best Parameters:\n {search.best_params_}")
    print(f"Test score: {test_score:0.3f}")
    print("")

print('--------------------------------------------------------------')
print('PROCESAMIENTO FINALIZADO EXITOSAMENTE!!!')
print('--------------------------------------------------------------')

TIEMPO = (time.time() - start_time)/60

# print("--- %s Segundos ---" % (time.time() - start_time))
print("--- %s Minutos ---" % (TIEMPO))

"""### 3.2 Bosques Simples Variables"""

import time
start_time = time.time()

print('-------------------INICIO PROCESAMIENTO-----------------')

Cantidad = 20

 # Hyperparameters.
param_gride = {"model__max_depth": [2,3,4],
              "model__min_samples_split": [2,4],
              "model__min_samples_leaf": [2,4],
              "model__n_estimators": [150]}

for i in range(Cantidad):

   # Loading the data.
  train_features = df_inputs_all.iloc[:,i]
  targets = df_output_all.iloc[:,0]

  NameVariable = df_inputs_all.columns[i]

  # Train, test split.
  data_split = train_test_split(train_features, targets, test_size=0.3, random_state=4444)
  X_train, X_test, targets_train, targets_test = data_split

  # Getting the target.
  y_train = targets_train
  y_test = targets_test

  # Model definition.
  pipe = Pipeline([("minmax", MinMaxScaler((-1, 1))),
                    ("model", RandomForestRegressor(random_state=4444))])
  # By default GridSearchCV uses a 5-kfold validation strategy.
  search = GridSearchCV(pipe, param_gride, cv=7, n_jobs=-1)
  search.fit(X_train.values.reshape(-1, 1), y_train.values)

  # Getting the test score.
  y_hat_test = search.predict(X_test.values.reshape(-1, 1))
  test_score = r2_score(y_test.values, y_hat_test)

  # Printing stats.
  print(f"Columna: {NameVariable}")
  print(f"Best CV score: {search.best_score_:0.3f}")
  print(f"Best Parameters:\n {search.best_params_}")
  print(f"Test score: {test_score:0.3f}")
  print("")

print('--------------------------------------------------------------')
print('PROCESAMIENTO FINALIZADO EXITOSAMENTE!!!')
print('--------------------------------------------------------------')

TIEMPO = (time.time() - start_time)/60

# print("--- %s Segundos ---" % (time.time() - start_time))
print("--- %s Minutos ---" % (TIEMPO))

"""### 3.2 Bosques Multiples Variables"""

import time
start_time = time.time()

print('-------------------INICIO PROCESAMIENTO-----------------')

# Hyperparameters.

#param_gride = {"model__max_depth": [2,3],
#              "model__min_samples_split": [2,3],
#              "model__min_samples_leaf": [2,3],
#              "model__n_estimators": [50]}

param_gride = {"model__max_depth": [5,10],
              "model__min_samples_split": [2,3,4,5],
              "model__min_samples_leaf": [2,3,4,5],
              "model__n_estimators": [150]}


# Loading the data.
train_features = df_inputs_all
targets = df_output_all

# Train, test split.
data_split = train_test_split(train_features, targets, test_size=0.3, random_state=4444)
X_train, X_test, targets_train, targets_test = data_split

# Find the best model per target.
for target in range(targets.shape[1]):
    # Getting the target.
    y_train = targets_train.iloc[:, target]
    y_test = targets_test.iloc[:, target]

    # Model definition.
    pipe = Pipeline([("minmax", MinMaxScaler((-1, 1))),
                      ("model", RandomForestRegressor(random_state=4444,n_jobs=-1))])
    # By default GridSearchCV uses a 5-kfold validation strategy.
    search = GridSearchCV(estimator=pipe, param_grid = param_gride, cv=7, n_jobs=-1)
    search.fit(X_train.values, y_train.values)

    # Getting the test score.
    y_hat_test = search.predict(X_test.values)
    test_score = r2_score(y_test.values, y_hat_test)

    # Printing stats.
    print(f"Columna: {targets.columns[target]}")
    print(f"Best CV score: {search.best_score_:0.3f}")
    print(f"Test score: {test_score:0.3f}")
    print(f"Best Parameters:\n {search.best_params_}")
    print("")

print('--------------------------------------------------------------')
print('PROCESAMIENTO FINALIZADO EXITOSAMENTE!!!')
print('--------------------------------------------------------------')

TIEMPO = (time.time() - start_time)/60

# print("--- %s Segundos ---" % (time.time() - start_time))
print("--- %s Minutos ---" % (TIEMPO))

"""### 3.3 SVM Simples Variables"""

import time
start_time = time.time()

print('-------------------INICIO PROCESAMIENTO-----------------')

Cantidad = 20

# Hyperparameters.

#param_gride = {"model__kernel": ['poly'],
#              "model__gamma": ['scale'],
#              "model__degree": [1],
#              "model__epsilon": [0.0001],
#              "model__C":  [1]}

param_gride = {"model__kernel": ['poly','rbf'],
              "model__gamma": ['scale','auto'],
              "model__degree": [1,2],
              "model__epsilon": [0.01, 0.1],
              "model__C":  [1]}

for i in range(Cantidad):

   # Loading the data.
  train_features = df_inputs_all.iloc[:,i]
  targets = df_output_all.iloc[:,0]

  NameVariable = df_inputs_all.columns[i]

  # Train, test split.
  data_split = train_test_split(train_features, targets, test_size=0.3, random_state=4444)
  X_train, X_test, targets_train, targets_test = data_split

  # Getting the target.
  y_train = targets_train
  y_test = targets_test

  # Model definition.
  pipe = Pipeline([("minmax", MinMaxScaler((-1, 1))),
                    ("model", SVR())])

  # By default GridSearchCV uses a 5-kfold validation strategy.
  search = GridSearchCV(pipe, param_gride, cv=7, n_jobs=-1)
  search.fit(X_train.values.reshape(-1, 1), y_train.values)

  # Getting the test score.
  y_hat_test = search.predict(X_test.values.reshape(-1, 1))
  test_score = r2_score(y_test.values, y_hat_test)

  # Printing stats.
  print(f"Columna: {NameVariable}")
  print(f"Best CV score: {search.best_score_:0.3f}")
  print(f"Best Parameters:\n {search.best_params_}")
  print(f"Test score: {test_score:0.3f}")
  print("")

print('--------------------------------------------------------------')
print('PROCESAMIENTO FINALIZADO EXITOSAMENTE!!!')
print('--------------------------------------------------------------')

TIEMPO = (time.time() - start_time)/60

# print("--- %s Segundos ---" % (time.time() - start_time))
print("--- %s Minutos ---" % (TIEMPO))

"""### 3.3 SVM Multiples Variables"""

import time
start_time = time.time()

print('-------------------INICIO PROCESAMIENTO-----------------')

# Hyperparameters.

#param_gride = {"model__kernel": ['poly'],
#              "model__gamma": ['scale'],
#              "model__degree": [1],
#              "model__epsilon": [0.0001],
#              "model__C":  [1]}

param_gride = {"model__kernel": ['poly','rbf'],
              "model__gamma": ['scale','auto'],
              "model__degree": [1,2,3,4,5],
              "model__epsilon": [0.001,0.01, 0.1],
              "model__C":  [1, 3, 5]}

# Loading the data.
train_features = df_inputs_all
targets = df_output_all

# Train, test split.
data_split = train_test_split(train_features, targets, test_size=0.3, random_state=4444)
X_train, X_test, targets_train, targets_test = data_split

# Find the best model per target.
for target in range(targets.shape[1]):
    # Getting the target.
    y_train = targets_train.iloc[:, target]
    y_test = targets_test.iloc[:, target]

    # Model definition.
    pipe = Pipeline([("minmax", MinMaxScaler((-1, 1))),
                      ("model", SVR())])
    # By default GridSearchCV uses a 5-kfold validation strategy.
    search = GridSearchCV(estimator=pipe, param_grid = param_gride, cv=7, n_jobs=-1)
    search.fit(X_train.values, y_train.values)

    # Getting the test score.
    y_hat_test = search.predict(X_test.values)
    test_score = r2_score(y_test.values, y_hat_test)

    # Printing stats.
    print(f"Columna: {targets.columns[target]}")
    print(f"Best CV score: {search.best_score_:0.3f}")
    print(f"Test score: {test_score:0.3f}")
    print(f"Best Parameters:\n {search.best_params_}")
    print("")

print('--------------------------------------------------------------')
print('PROCESAMIENTO FINALIZADO EXITOSAMENTE!!!')
print('--------------------------------------------------------------')

TIEMPO = (time.time() - start_time)/60

# print("--- %s Segundos ---" % (time.time() - start_time))
print("--- %s Minutos ---" % (TIEMPO))

"""### 3. Creación del modelo (Minoristas)"""

# Definición variables de entrada y variable de salida

df_inputs_all= df_sample_minorista.drop(columns=['TotalTransaction'])

df_output_all = df_sample_minorista[['TotalTransaction']].copy()

"""### 3.1 Arboles Simples Variables"""

# Creación de variables para el test y el train

TiempoEntrenamientoTRAIN=[]
TiempoEvaluacionTRAIN=[]
RMSETrain=[]
R2Train=[]
MSETrain=[]
MAETrain=[]
nameColum=[]

TiempoEntrenamientoTEST=[]
TiempoEvaluacionTEST=[]
RMSETest=[]
R2Test=[]
MSETest=[]
MAETest=[]

import time
start_time = time.time()

print('-------------------INICIO PROCESAMIENTO-----------------')

Cantidad = 20

for i in range(Cantidad):

  # Hyperparameters.
  max_depth = 10
  minsamplesplit = 10
  minsampleleaf = 10

  param_grid = {"model__max_depth": list(range(1, max_depth + 1)),
                "model__min_samples_split": list(range(1, minsamplesplit + 1)),
                "model__min_samples_leaf": list(range(1, minsampleleaf + 1))}

  #param_grid = {"model__min_samples_split": list(range(1, minsamplesplit + 1)),
  #              "model__min_samples_leaf": list(range(1, minsampleleaf + 1))}

  # Loading the data.
  train_features = df_inputs_all.iloc[:,i]
  targets = df_output_all.iloc[:,0]

  NameVariable = df_inputs_all.columns[i]

  # Train, test split.
  data_split = train_test_split(train_features, targets, test_size=0.3, random_state=4444)
  X_train, X_test, targets_train, targets_test = data_split

  # Getting the target.
  y_train = targets_train
  y_test = targets_test

  # Model definition.
  pipe = Pipeline([("minmax", MinMaxScaler((-1, 1))),
                    ("model", DecisionTreeRegressor(random_state=4444))])
  # By default GridSearchCV uses a 5-kfold validation strategy.
  search = GridSearchCV(pipe, param_grid, cv=7, n_jobs=-1)
  search.fit(X_train.values.reshape(-1, 1), y_train.values)

  # Getting the test score.
  y_hat_test = search.predict(X_test.values.reshape(-1, 1))
  test_score = r2_score(y_test.values, y_hat_test)

  # Printing stats.
  print(f"Columna: {NameVariable}")
  print(f"Best CV score: {search.best_score_:0.3f}")
  print(f"Best Parameters:\n {search.best_params_}")
  print(f"Test score: {test_score:0.3f}")
  print("")

print('--------------------------------------------------------------')
print('PROCESAMIENTO FINALIZADO EXITOSAMENTE!!!')
print('--------------------------------------------------------------')

TIEMPO = (time.time() - start_time)/60

# print("--- %s Segundos ---" % (time.time() - start_time))
print("--- %s Minutos ---" % (TIEMPO))

"""### 3.1 Arboles Multiples Variables"""

import time
start_time = time.time()

print('-------------------INICIO PROCESAMIENTO-----------------')

# Hyperparameters.
max_depth = 10
minsamplesplit = 10
minsampleleaf = 10

param_grid = {"model__max_depth": list(range(1, max_depth + 1)),
               "model__min_samples_split": list(range(1, minsamplesplit + 1)),
               "model__min_samples_leaf": list(range(1, minsampleleaf + 1))}

#param_grid = {"model__min_samples_split": list(range(1, minsamplesplit + 1)),
#              "model__min_samples_leaf": list(range(1, minsampleleaf + 1))}

# Loading the data.
train_features = df_inputs_all
targets = df_output_all

# Train, test split.
data_split = train_test_split(train_features, targets, test_size=0.3, random_state=4444)
X_train, X_test, targets_train, targets_test = data_split

# Find the best model per target.
for target in range(targets.shape[1]):
    # Getting the target.
    y_train = targets_train.iloc[:, target]
    y_test = targets_test.iloc[:, target]

    # Model definition.
    pipe = Pipeline([("minmax", MinMaxScaler((-1, 1))),
                      ("model", DecisionTreeRegressor(random_state=4444))])
    # By default GridSearchCV uses a 5-kfold validation strategy.
    search = GridSearchCV(pipe, param_grid, cv=7, n_jobs=-1)
    search.fit(X_train.values, y_train.values)

    # Getting the test score.
    y_hat_test = search.predict(X_test.values)
    test_score = r2_score(y_test.values, y_hat_test)

    # Printing stats.
    print(f"Columna: {targets.columns[target]}")
    print(f"Best CV score: {search.best_score_:0.3f}")
    print(f"Best Parameters:\n {search.best_params_}")
    print(f"Test score: {test_score:0.3f}")
    print("")

print('--------------------------------------------------------------')
print('PROCESAMIENTO FINALIZADO EXITOSAMENTE!!!')
print('--------------------------------------------------------------')

TIEMPO = (time.time() - start_time)/60

# print("--- %s Segundos ---" % (time.time() - start_time))
print("--- %s Minutos ---" % (TIEMPO))

"""### 3.2 Bosques Simples Variables"""

import time
start_time = time.time()

print('-------------------INICIO PROCESAMIENTO-----------------')

Cantidad = 20

 # Hyperparameters.
param_gride = {"model__max_depth": [2,3,4],
              "model__min_samples_split": [2,4],
              "model__min_samples_leaf": [2,4],
              "model__n_estimators": [150]}

for i in range(Cantidad):

   # Loading the data.
  train_features = df_inputs_all.iloc[:,i]
  targets = df_output_all.iloc[:,0]

  NameVariable = df_inputs_all.columns[i]

  # Train, test split.
  data_split = train_test_split(train_features, targets, test_size=0.3, random_state=4444)
  X_train, X_test, targets_train, targets_test = data_split

  # Getting the target.
  y_train = targets_train
  y_test = targets_test

  # Model definition.
  pipe = Pipeline([("minmax", MinMaxScaler((-1, 1))),
                    ("model", RandomForestRegressor(random_state=4444))])
  # By default GridSearchCV uses a 5-kfold validation strategy.
  search = GridSearchCV(pipe, param_gride, cv=7, n_jobs=-1)
  search.fit(X_train.values.reshape(-1, 1), y_train.values)

  # Getting the test score.
  y_hat_test = search.predict(X_test.values.reshape(-1, 1))
  test_score = r2_score(y_test.values, y_hat_test)

  # Printing stats.
  print(f"Columna: {NameVariable}")
  print(f"Best CV score: {search.best_score_:0.3f}")
  print(f"Best Parameters:\n {search.best_params_}")
  print(f"Test score: {test_score:0.3f}")
  print("")

print('--------------------------------------------------------------')
print('PROCESAMIENTO FINALIZADO EXITOSAMENTE!!!')
print('--------------------------------------------------------------')

TIEMPO = (time.time() - start_time)/60

# print("--- %s Segundos ---" % (time.time() - start_time))
print("--- %s Minutos ---" % (TIEMPO))

"""### 3.2 Bosques Multiples Variables"""

import time
start_time = time.time()

print('-------------------INICIO PROCESAMIENTO-----------------')

# Hyperparameters.

#param_gride = {"model__max_depth": [2,3],
#              "model__min_samples_split": [2,3],
#              "model__min_samples_leaf": [2,3],
#              "model__n_estimators": [50]}

param_gride = {"model__max_depth": [5,10],
              "model__min_samples_split": [2,3,4,5],
              "model__min_samples_leaf": [2,3,4,5],
              "model__n_estimators": [150]}


# Loading the data.
train_features = df_inputs_all
targets = df_output_all

# Train, test split.
data_split = train_test_split(train_features, targets, test_size=0.3, random_state=4444)
X_train, X_test, targets_train, targets_test = data_split

# Find the best model per target.
for target in range(targets.shape[1]):
    # Getting the target.
    y_train = targets_train.iloc[:, target]
    y_test = targets_test.iloc[:, target]

    # Model definition.
    pipe = Pipeline([("minmax", MinMaxScaler((-1, 1))),
                      ("model", RandomForestRegressor(random_state=4444,n_jobs=-1))])
    # By default GridSearchCV uses a 5-kfold validation strategy.
    search = GridSearchCV(estimator=pipe, param_grid = param_gride, cv=7, n_jobs=-1)
    search.fit(X_train.values, y_train.values)

    # Getting the test score.
    y_hat_test = search.predict(X_test.values)
    test_score = r2_score(y_test.values, y_hat_test)

    # Printing stats.
    print(f"Columna: {targets.columns[target]}")
    print(f"Best CV score: {search.best_score_:0.3f}")
    print(f"Test score: {test_score:0.3f}")
    print(f"Best Parameters:\n {search.best_params_}")
    print("")

print('--------------------------------------------------------------')
print('PROCESAMIENTO FINALIZADO EXITOSAMENTE!!!')
print('--------------------------------------------------------------')

TIEMPO = (time.time() - start_time)/60

# print("--- %s Segundos ---" % (time.time() - start_time))
print("--- %s Minutos ---" % (TIEMPO))

"""### 3.3 SVM Simples Variables"""

import time
start_time = time.time()

print('-------------------INICIO PROCESAMIENTO-----------------')

Cantidad = 20

# Hyperparameters.

#param_gride = {"model__kernel": ['poly'],
#              "model__gamma": ['scale'],
#              "model__degree": [1],
#              "model__epsilon": [0.0001],
#              "model__C":  [1]}

param_gride = {"model__kernel": ['poly','rbf'],
              "model__gamma": ['scale','auto'],
              "model__degree": [1,2],
              "model__epsilon": [0.01, 0.1],
              "model__C":  [1]}

for i in range(Cantidad):

   # Loading the data.
  train_features = df_inputs_all.iloc[:,i]
  targets = df_output_all.iloc[:,0]

  NameVariable = df_inputs_all.columns[i]

  # Train, test split.
  data_split = train_test_split(train_features, targets, test_size=0.3, random_state=4444)
  X_train, X_test, targets_train, targets_test = data_split

  # Getting the target.
  y_train = targets_train
  y_test = targets_test

  # Model definition.
  pipe = Pipeline([("minmax", MinMaxScaler((-1, 1))),
                    ("model", SVR())])

  # By default GridSearchCV uses a 5-kfold validation strategy.
  search = GridSearchCV(pipe, param_gride, cv=7, n_jobs=-1)
  search.fit(X_train.values.reshape(-1, 1), y_train.values)

  # Getting the test score.
  y_hat_test = search.predict(X_test.values.reshape(-1, 1))
  test_score = r2_score(y_test.values, y_hat_test)

  # Printing stats.
  print(f"Columna: {NameVariable}")
  print(f"Best CV score: {search.best_score_:0.3f}")
  print(f"Best Parameters:\n {search.best_params_}")
  print(f"Test score: {test_score:0.3f}")
  print("")

print('--------------------------------------------------------------')
print('PROCESAMIENTO FINALIZADO EXITOSAMENTE!!!')
print('--------------------------------------------------------------')

TIEMPO = (time.time() - start_time)/60

# print("--- %s Segundos ---" % (time.time() - start_time))
print("--- %s Minutos ---" % (TIEMPO))

"""### 3.3 SVM Multiples Variables"""

import time
start_time = time.time()

print('-------------------INICIO PROCESAMIENTO-----------------')

# Hyperparameters.

#param_gride = {"model__kernel": ['poly'],
#              "model__gamma": ['scale'],
#              "model__degree": [1],
#              "model__epsilon": [0.0001],
#              "model__C":  [1]}

param_gride = {"model__kernel": ['poly','rbf'],
              "model__gamma": ['scale','auto'],
              "model__degree": [1,2,3,4,5],
              "model__epsilon": [0.001,0.01, 0.1],
              "model__C":  [1, 3, 5]}

# Loading the data.
train_features = df_inputs_all
targets = df_output_all

# Train, test split.
data_split = train_test_split(train_features, targets, test_size=0.3, random_state=4444)
X_train, X_test, targets_train, targets_test = data_split

# Find the best model per target.
for target in range(targets.shape[1]):
    # Getting the target.
    y_train = targets_train.iloc[:, target]
    y_test = targets_test.iloc[:, target]

    # Model definition.
    pipe = Pipeline([("minmax", MinMaxScaler((-1, 1))),
                      ("model", SVR())])
    # By default GridSearchCV uses a 5-kfold validation strategy.
    search = GridSearchCV(estimator=pipe, param_grid = param_gride, cv=7, n_jobs=-1)
    search.fit(X_train.values, y_train.values)

    # Getting the test score.
    y_hat_test = search.predict(X_test.values)
    test_score = r2_score(y_test.values, y_hat_test)

    # Printing stats.
    print(f"Columna: {targets.columns[target]}")
    print(f"Best CV score: {search.best_score_:0.3f}")
    print(f"Test score: {test_score:0.3f}")
    print(f"Best Parameters:\n {search.best_params_}")
    print("")

print('--------------------------------------------------------------')
print('PROCESAMIENTO FINALIZADO EXITOSAMENTE!!!')
print('--------------------------------------------------------------')

TIEMPO = (time.time() - start_time)/60

# print("--- %s Segundos ---" % (time.time() - start_time))
print("--- %s Minutos ---" % (TIEMPO))
