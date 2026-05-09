# Semana 7 - Consolidado

Resumen de actividades y decisiones:

Proyecto: Predicción de precio de alojamientos en Airbnb usando el dataset de Kaggle.

1. Estructura del proyecto
- `Proyecto/Final/` — código y artefactos finales.
	- `Datos/raw.csv` — datos originales.
	- `Datos/cleaned.csv` — datos procesados usados para el análisis.
	- `Limpieza.py` — funciones `load_data()` y `clean_data()` para preparar el dataset.
	- `Modelo.py` — función `train_model()` que entrena el modelo, calcula métricas y guarda `model.joblib`.
	- `run_clean_noninteractive.py` — script para correr la limpieza desde línea de comandos.
	- `Presentacion_Notebook.ipynb` — notebook con exploración, visualizaciones y resultados.
	- `requirements.txt` — dependencias del proyecto.

2. Datos y limpieza
- Fuente: dataset público de Kaggle (Airbnb Price Prediction). El CSV original se guarda en `Datos/raw.csv`.
- Pasos principales de limpieza (implementados en `Limpieza.py`):
	- Normalización de nombres y tipos de columnas.
	- Imputación o tratamiento de valores faltantes en `bedrooms`, `bathrooms`, `beds`, `review_scores_rating`, etc.
	- Conversión de la variable objetivo: se calcula `log_price = log(price)` para modelar en escala logarítmica.
	- Guardado del resultado en `Datos/cleaned.csv`.

3. Ingeniería de variables y selección
- Variable objetivo: `log_price` (modelo) y `price` (para interpretación en escala real).
- Variables explicativas principales usadas: `accommodates`, `bathrooms`, `bedrooms`, `beds`, `city`, `room_type`, `property_type`, `review_scores_rating`, y algunas tasas (`cleaning_fee`, `security_deposit`, `extra_people`) si están presentes.
- Se revisó la multicolinealidad con VIF; variables con VIF alto fueron consideradas para exclusión o agrupación.

4. Modelado
- Implementación: `train_model(clean_path, target='log_price', model_out='model.joblib')` en `Modelo.py`.
- Métricas reportadas en el notebook:
	- R² (sobre `log_price`): ~0.56
	- RMSE en precio real: ~79
	- MAE en precio real: ~48
	- Se calculan además MSE, RMSE sobre `log_price`, VIF y coeficientes del modelo.
- Artefactos: `model.joblib` (modelo entrenado) y `predictions.csv` (predicciones guardadas: `price_true`, `price_pred`, `log_price_true`, `log_price_pred`).

5. Visualizaciones generadas
- Pairplot de `price`, `accommodates`, `bedrooms`, `beds`, `bathrooms`.
- Barra de correlaciones absolutas con la variable objetivo.
- Heatmap de correlación entre las principales variables.
- Scatter/regplot de las top3 variables vs `price`/`log_price`.
- Histogramas de la distribución objetivo.
- Gráficos de `predicted_vs_actual.png` y `residuals.png` (guardados en `Visualizaciones/`).

6. Interpretación y conclusiones
- Las variables de mayor peso son de tamaño/capacidad (`accommodates`, `bedrooms`, `beds`, `bathrooms`) y, sobre todo, variables de ubicación (`city`) y tipo de alojamiento.
- El modelo explica una parte considerable de la variación (R² ≈ 0.56 en `log_price`) pero no captura toda la complejidad; el error típico en precio real es de ~79 (RMSE).
- Recomendaciones: probar modelos no lineales o de ensamblado, mayor ingeniería de variables (interacciones, distancias a puntos de interés), y validación por tiempo/ubicación si se dispone de datos temporales.

7. Cómo reproducir
- Instalar dependencias:

	pip install -r Proyecto/Final/requirements.txt

- Limpieza (genera `Datos/cleaned.csv`):

	python Proyecto/Final/run_clean_noninteractive.py

- Entrenar modelo desde Python (usa `Datos/cleaned.csv`):

	python -c "from Limpieza import load_data, clean_data; from Modelo import train_model; train_model('Proyecto/Final/Datos/cleaned.csv', target='log_price', model_out='Proyecto/Final/model.joblib')"

- Alternativamente, abrir y ejecutar `Proyecto/Final/Presentacion_Notebook.ipynb`.

8. Artefactos útiles
- `Proyecto/Final/model.joblib` — modelo final entrenado.
- `Proyecto/Final/predictions.csv` — predicciones y valores reales usados para las gráficas.
- `Proyecto/Final/Visualizaciones/` — imágenes generadas por el notebook.

9. Siguientes pasos sugeridos
- Evaluar modelos alternativos (Random Forest, XGBoost) y comparar con validación cruzada.
- Añadir validación espacial o por segmentos de ciudad.
- Crear script de evaluación automatizada y un README con instrucciones de despliegue.

Si quieres, puedo: 1) commitear y pushear esta actualización, 2) abrir un pull request en el repo remoto, o 3) generar un README con comandos de reproducción.
