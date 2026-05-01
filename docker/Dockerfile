FROM jupyter/pyspark-notebook:spark-3.5.0

USER root

# Instalar librerías adicionales (sklearn, lightgbm, xgboost, catboost, etc.)
RUN pip install --no-cache-dir \
    scikit-learn==1.4.0 \
    lightgbm==4.3.0 \
    xgboost==2.0.3 \
    catboost==1.2.5 \
    matplotlib==3.8.2 \
    seaborn==0.13.2 \
    pandas==2.2.0 \
    numpy==1.26.3

USER ${NB_UID}

# Configurar directorio de trabajo
WORKDIR /home/jovyan/work
