ENTRENAMIENTO CON MLFLOW
Martha Alicia Villalobos Facundo | Matrícula: A01840063

Esta carpeta contiene el código de entrenamiento del notebook
TC5061_MLflow_Colab_MarthaVillalobos.ipynb.

Archivos necesarios:
train.py: carga los datos, prepara las variables, entrena y evalúa el modelo.
requirements.txt: contiene las librerías y sus versiones.
run_experiments.py: ejecuta las cinco configuraciones y repite el baseline.

EJECUCIÓN EN GOOGLE COLAB

1. Abre el notebook y ejecuta las celdas hasta terminar el paso 4.
   Ahí se crean los archivos, se prepara Python 3.12.10 y se instalan
   las librerías. Los archivos quedan en /content/tc5061_mlflow.

2. Antes de entrenar, ejecuta esta celda:

   import os
   os.chdir('/content/tc5061_mlflow')
   os.environ['MPLBACKEND'] = 'Agg'

   Esta configuración permite guardar los gráficos como imágenes.

3. Para ejecutar el modelo base, copia este comando en una celda de código:

!MPLBACKEND=Agg /content/tc5061_mlflow/.venv312/bin/python train.py --C 1.0 --max-iter 1000 --class-weight balanced --tol 0.0001 --seed 42 --run-name 01_baseline

   Para ejecutar la configuración 05_tol_0_01, usa:

!MPLBACKEND=Agg /content/tc5061_mlflow/.venv312/bin/python train.py --C 1.0 --max-iter 1000 --class-weight balanced --tol 0.01 --seed 42 --run-name 05_tol_0_01

4. Para ejecutar las cinco configuraciones y la réplica, usa este comando:

!MPLBACKEND=Agg /content/tc5061_mlflow/.venv312/bin/python run_experiments.py

   Este comando ya incluye las dos configuraciones anteriores. Si solo
   necesitas la comparación completa, puedes ejecutar directamente el lote.
   También comprueba la repetición del baseline y el modelo guardado.

5. Para abrir MLflow, ejecuta la celda que inicia el servidor en el paso 9
   del notebook. Después ejecuta:

   from google.colab import output
   output.serve_kernel_port_as_iframe(5000, height=800)

   La interfaz aparecerá dentro de Colab. Selecciona el experimento:
   TC5061_Online_Shoppers_Regresion_Logistica_Comparacion

   Usa el batch_id que imprime el notebook para identificar tu lote.
   Para comparar solo las cinco configuraciones, filtra también por
   tags.role = 'comparacion'. La sexta corrida es la réplica del baseline.

EJECUCIÓN EN UNA COMPUTADORA

Usa Python 3.12.10 y abre una terminal en la carpeta de los archivos.

1. Crea un entorno:

   python -m venv .venv

2. Actívalo. En Windows PowerShell:

   .\.venv\Scripts\Activate.ps1

   En Linux o macOS:

   source .venv/bin/activate

3. Instala las librerías:

   python -m pip install -r requirements.txt

4. Configura los gráficos. En Windows PowerShell:

   $env:MPLBACKEND = 'Agg'

   En Linux o macOS:

   export MPLBACKEND=Agg

5. Ejecuta el modelo base:

   python train.py --C 1.0 --max-iter 1000 --class-weight balanced --tol 0.0001 --seed 42 --run-name 01_baseline

   Para consultar los argumentos: python train.py --help
   Para entrenar con los valores por defecto: python train.py
   Para ejecutar las cinco configuraciones y la réplica: python run_experiments.py

6. Abre MLflow con este comando:

   python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

   Después abre http://127.0.0.1:5000 en el navegador de tu computadora.

RESULTADOS Y CAPTURAS

Si el CSV no está disponible, el script lo descarga de UCI; necesitas
conexión a internet. Cada corrida registra parámetros, cinco métricas,
archivos de evaluación y el modelo entrenado.

Conserva mlflow.db junto con las carpetas mlruns y outputs. Consulta los
artefactos desde la carpeta donde ejecutaste el entrenamiento, porque
MLflow puede guardar sus rutas completas. En Colab descarga los resultados
antes de cerrar la sesión.

Las capturas incluidas están en la carpeta capturas:
tabla_mlflow.png: cinco configuraciones y una réplica del baseline.
detalle_corrida_artefactos.png: corrida 05_tol_0_01 y archivos del modelo.

La semilla es 42. Para repetir los resultados, conserva los mismos datos,
argumentos y versiones. Al ejecutar de nuevo se crean corridas nuevas.

REFERENCIAS

Dataset UCI:
https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

MLflow Tracking:
https://mlflow.org/docs/latest/ml/tracking/

Argparse:
https://docs.python.org/3/library/argparse.html
