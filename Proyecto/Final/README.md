# Proyecto Final - Airbnb Price Prediction

Este directorio contiene la versión actualizada del proyecto con el dataset de Kaggle:

https://www.kaggle.com/datasets/stevezhenghp/airbnb-price-prediction

## Flujo de trabajo

1. Descargar los datos:

```bash
python download_data.py
```

2. Limpiar el CSV:

```bash
python Limpieza.py --input Datos/raw.csv --output Datos/cleaned.csv
```

3. Entrenar el modelo:

```bash
python Modelo.py --input Datos/cleaned.csv --output-model model.joblib
```

4. Generar visualizaciones:

```bash
python Visualizaciones/generate_presentation_plots.py
python Visualizaciones/plot_predictions.py --pred predictions.csv --out Visualizaciones
```

## Qué hace el proyecto

- Usa `log_price` como variable objetivo.
- Conserva variables numéricas y categóricas útiles para predecir precios.
- Convierte el precio a escala real solo para explicar mejor los resultados.
- Genera gráficas y notebooks narrativos para presentar el análisis.
