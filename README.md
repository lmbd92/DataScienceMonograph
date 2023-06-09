# README - Proyecto de Ciencia en Datos

Este repositorio contiene el proyecto del curso de Seminario de la Especialización en Analítica y Ciencia de Datos de la Universidad de Antioquia, sede Medellín, correspondiente a la Cohorte 5 del año 2023.

## Integrantes
- Lina Beltrán
- Mario Otero

## Objetivo del curso

El objetivo de este curso es desarrollar habilidades en la aplicación de fundamentos metodológicos y conceptuales para la identificación y formulación de proyectos de investigación aplicada en el campo de la analítica y ciencia de datos. Además, se busca explorar consideraciones prácticas de conceptos de analítica de datos en proyectos aplicados, así como promover la elaboración y presentación de una propuesta monográfica.

## Caso de estudio

El caso de estudio seleccionado para este proyecto es el conjunto de datos de venta minorista y mayorista en línea II. Este conjunto de datos contiene todas las transacciones que se produjeron en una tienda minorista en línea registrada y con sede en el Reino Unido entre el 12/01/2009 y el 12/09/2011. La empresa vende principalmente artículos de regalo únicos para toda ocasión, y muchos de sus clientes son mayoristas.

Este fue tomado de: [Online Retail II UCI](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci/code)


## Información de atributos

El conjunto de datos proporciona la siguiente información de atributos:

- Número de factura: Número de factura asignado de manera única a cada transacción. Si el código comienza con la letra 'c', indica una cancelación.
- StockCode: Código del producto (artículo) asignado de forma única a cada producto distinto.
- Descripción: Nombre del producto (artículo).
- Cantidad: Cantidades de cada producto (artículo) por transacción.
- InvoiceDate: Fecha y hora de la factura, indicando el día y la hora en que se generó una transacción.
- PrecioUnidad: Precio unitario del producto en libras esterlinas (£).
- CustomerID: Número de cliente asignado de manera única a cada cliente.
- País: Nombre del país donde reside un cliente.

## Caracteríticas del dataset

- RangeIndex: 1067371 entries, 0 to 1067370
- Data columns (total 8 columns):
- dtypes: float64(2), int64(1), object(5)
- shape: (1067371, 8)

| Columna       | No Nulo | Tipo    |
|---------------|---------|---------|
| Invoice       | 1,067,371 | Objeto  |
| StockCode     | 1,067,371 | Objeto  |
| Description   | 1,062,989 | Objeto  |
| Quantity      | 1,067,371 | Entero  |
| InvoiceDate   | 1,067,371 | Objeto  |
| Price         | 1,067,371 | Flotante |
| Customer ID   |   824,364 | Flotante |
| Country       | 1,067,371 | Objeto  |


## Instrucciones de uso

**Para utilizar el código y los datos de este proyecto DE MANERA LOCAL, siga los siguientes pasos:**

1. Clone este repositorio en su máquina local.
2. Asegúrese de tener instalado Python 3 y las bibliotecas necesarias mencionadas en el archivo de requisitos "requirements.txt" ubicado en el folder "Ejecutables". Puede usar este archivo para instalar los recursos en su maquina local, con el siguiente comando `pip install -r requirements.txt`
3. Ejecute el archivo principal del proyecto ubicado en el folder "Ejecutables" utilizando el intérprete de Python.

```bash

python main.py

```

4. Siga las instrucciones presentadas en la interfaz para interactuar con el proyecto y analizar los datos.

**Para utilizar el código y los datos de este proyecto DE MANERA ONLINE, siga los siguientes pasos:**

1. Para ejecutar el script/notebook 1 ingrese a este link: [iteracion1ML.ipynb](https://github.com/lmbd92/DataScienceMonograph/blob/main/Notebooks/iteracion1ML.ipynb) para el script/notebook 2 ingrese a este link: [iteracion2ML.ipynb](https://github.com/lmbd92/DataScienceMonograph/blob/main/Notebooks/iteracion2ML.ipynb)
2. Haga click en el ícono "open in Colab" ubicado en la cabecera del archivo
3. En el menú de Colab despliegue la opción "Entorno de ejecución" y seguidamente seleccione la opción "Ejecutar todas" / ctrl +F9 (ver imagen)

![Ejecutar Todas](https://github.com/lmbd92/DataScienceMonograph/blob/main/Assets/ejecutarTodasColab.png)


## Contribución

Si desea contribuir a este proyecto, siga los siguientes pasos:

1. Realice un fork de este repositorio.
2. Cree una rama para su nueva característica (`git checkout -b feature/nueva-caracteristica`).
3. Realice los cambios necesarios y agregue los archivos modificados (`git add`).
4. Realice un commit de sus cambios (`git commit -m 'Agregada nueva característica'`).
5. Haga push de los cambios a la rama (`git push origin feature/nueva-caracteristica`).
6. Abra una solicitud de extracción en este repositorio.

Agradecemos cualquier contribución y estaremos encantados de revisar y fusionar las solicitudes de mejoras


## Licencia

Este proyecto se encuentra bajo la Licencia MIT.
