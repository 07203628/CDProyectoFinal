# Proyecto - Final (Semana 7)

Proyecto actualizado con el dataset de Kaggle **Airbnb price prediction**:

https://www.kaggle.com/datasets/stevezhenghp/airbnb-price-prediction

## Estructura

- `Final/download_data.py`: descarga `train.csv` desde KaggleHub y lo copia a `Datos/raw.csv`.
- `Final/Limpieza.py`: limpia el dataset, crea variables auxiliares y genera `Datos/cleaned.csv`.
- `Final/Modelo.py`: entrena una regresión lineal múltiple con `log_price` como objetivo, usando variables numéricas y categóricas.
- `Final/Visualizaciones/plot_predictions.py`: crea la comparación entre precio real y estimado, más el histograma de residuos.
- `Final/Visualizaciones/generate_presentation_plots.py`: crea las gráficas de presentación del análisis.
- `Final/Presentacion_Notebook.ipynb`: notebook narrativo para exponer el proyecto.
- `Final/analysis_notebook.ipynb`: notebook técnico de apoyo.

## Flujo recomendado

1. Descargar y actualizar los datos:

```bash
python Semana7/Proyecto/Final/download_data.py
```

2. Limpiar los datos:

```bash
python Semana7/Proyecto/Final/Limpieza.py --input Semana7/Proyecto/Final/Datos/raw.csv --output Semana7/Proyecto/Final/Datos/cleaned.csv
```

3. Entrenar el modelo:

```bash
python Semana7/Proyecto/Final/Modelo.py --input Semana7/Proyecto/Final/Datos/cleaned.csv --output-model Semana7/Proyecto/Final/model.joblib
```

4. Generar visualizaciones:

```bash
python Semana7/Proyecto/Final/Visualizaciones/generate_presentation_plots.py
python Semana7/Proyecto/Final/Visualizaciones/plot_predictions.py --pred Semana7/Proyecto/Final/predictions.csv --out Semana7/Proyecto/Final/Visualizaciones
```

## Notas

- El objetivo del modelo es `log_price`.
- Para mostrar resultados más fáciles de leer, las predicciones también se convierten a precio real usando la exponencial.
- Si el dataset no existe o quieres refrescarlo, usa `download_data.py`.
