"""Entrenamiento reproducible del baseline Online Shoppers. Ejecutar: python train.py."""
import os

# Limitar hilos antes de importar las librerías numéricas.
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ.setdefault('MPLCONFIGDIR', os.path.abspath('.matplotlib'))

import argparse
import hashlib
import io
import json
import platform
import random
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import requests
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

EXPERIMENT = 'TC5061_Online_Shoppers_Regresion_Logistica_Comparacion'
DATA_URL = 'https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip'
FEATURES = ['Administrative', 'Administrative_Duration', 'Informational',
            'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
            'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Month',
            'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--C', type=float, default=1.0, help='Inverso de la regularización; debe ser positivo')
    parser.add_argument('--max-iter', type=int, default=1000, help='Máximo de iteraciones del optimizador')
    parser.add_argument('--class-weight', choices=['balanced', 'none'], default='balanced', help='Ponderación de clases')
    parser.add_argument('--tol', type=float, default=1e-4, help='Tolerancia de convergencia positiva')
    parser.add_argument('--seed', type=int, default=42, help='Semilla para Python, NumPy y scikit-learn')
    parser.add_argument('--data-path', default='data/online_shoppers_intention.csv', help='CSV de UCI; se descarga si falta')
    parser.add_argument('--tracking-uri', default='sqlite:///mlflow.db', help='Almacén de MLflow')
    parser.add_argument('--experiment', default=EXPERIMENT, help='Nombre del experimento')
    parser.add_argument('--run-name', default='baseline', help='Nombre de la corrida')
    parser.add_argument('--batch-id', default='manual', help='Etiqueta para separar grupos de comparación')
    parser.add_argument('--role', default='comparacion', choices=['comparacion', 'replica'], help='Finalidad de la corrida')
    args = parser.parse_args()
    if args.C <= 0 or args.tol <= 0 or args.max_iter < 1 or not 0 <= args.seed < 2**32:
        parser.error('C y tol deben ser positivos, max-iter >= 1 y 0 <= seed < 2**32.')
    return args


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(path):
    """Descargar el CSV original, validar columnas y aplicar la limpieza del baseline."""
    path = Path(path)
    if not path.exists():
        response = requests.get(DATA_URL, timeout=90)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            matches = [n for n in archive.namelist() if Path(n).name == 'online_shoppers_intention.csv']
            if len(matches) != 1:
                raise ValueError('No se encontró un único CSV de Online Shoppers en UCI.')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(matches[0]))
    raw = pd.read_csv(path)
    if set(raw.columns) != set(FEATURES + ['Revenue']):
        raise ValueError('El CSV no contiene las 17 características y Revenue del baseline.')
    clean = raw.drop_duplicates().copy()
    for column in ['Weekend', 'Revenue']:
        # Evitar convertir silenciosamente texto desconocido en True.
        converted = clean[column].astype(str).str.strip().str.lower().map(
            {'true': 1, 'false': 0, '1': 1, '0': 0})
        if converted.isna().any():
            raise ValueError(f'Valores booleanos no reconocidos en {column}.')
        clean[column] = converted.astype(int)
    metadata = {'raw_rows': len(raw), 'clean_rows': len(clean),
                'duplicates_removed': len(raw) - len(clean), 'data_sha256': sha256(path)}
    return clean[FEATURES], clean['Revenue'], metadata


def split_data(X, y, seed):
    """Mantener el holdout estratificado 80/20 del notebook original."""
    return train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)


def build_pipeline(X_train, args):
    """Ajustar imputación, escalado y codificación exclusivamente en entrenamiento."""
    numeric = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical = X_train.select_dtypes(exclude=np.number).columns.tolist()
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                         ('scaler', RobustScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                         ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocess = ColumnTransformer([('num', num_pipe, numeric), ('cat', cat_pipe, categorical)])
    model = LogisticRegression(C=args.C, max_iter=args.max_iter,
                               class_weight=None if args.class_weight == 'none' else 'balanced',
                               tol=args.tol, random_state=args.seed, solver='lbfgs', penalty='l2')
    return Pipeline([('preprocess', preprocess), ('model', model)])


def fit_model(pipeline, X_train, y_train):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        pipeline.fit(X_train, y_train)
    messages = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    return pipeline, messages


def evaluate_model(model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = {'accuracy': accuracy_score(y_test, pred),
               'precision': precision_score(y_test, pred, zero_division=0),
               'recall': recall_score(y_test, pred, zero_division=0),
               'f1': f1_score(y_test, pred, zero_division=0),
               'roc_auc': roc_auc_score(y_test, prob)}
    return metrics, pred, prob


def save_artifacts(folder, metrics, model, X_train, X_test, y_test, pred, prob, messages):
    (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    report = classification_report(y_test, pred, digits=5, zero_division=0)
    (folder / 'classification_report.txt').write_text(report, encoding='utf-8')
    pd.DataFrame({'row_index': X_test.index, 'real': y_test.to_numpy(),
                  'prediction': pred, 'probability': prob}).to_csv(folder / 'predictions.csv', index=False)
    (folder / 'split.json').write_text(json.dumps({'train': X_train.index.tolist(),
                                                 'test': X_test.index.tolist()}), encoding='utf-8')
    (folder / 'convergence.json').write_text(json.dumps({'warnings': messages,
         'n_iter': model.named_steps['model'].n_iter_.tolist()}, indent=2), encoding='utf-8')
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=['No compra', 'Compra'], cmap='Blues')
    plt.title('Matriz de confusión — Online Shoppers')
    plt.tight_layout()
    plt.savefig(folder / 'confusion_matrix.png', dpi=160)
    plt.close()
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (folder / 'environment_freeze.txt').write_text(freeze, encoding='utf-8')
    (folder / 'python_version.txt').write_text(platform.python_version(), encoding='utf-8')


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.set_tags({'batch_id': args.batch_id, 'role': args.role,
                         'baseline_source': 'TC5061_Notebook_Baseline_VillalobosMartha.ipynb'})
        for name, value in vars(args).items():
            mlflow.log_param(name, value)
        for name, value in {'model': 'LogisticRegression', 'solver': 'lbfgs', 'penalty': 'l2',
                            'test_size': 0.20, 'scaler': 'RobustScaler', 'encoding': 'OneHotEncoder',
                            'train_sha256': sha256(__file__)}.items():
            mlflow.log_param(name, value)
        X, y, metadata = load_data(args.data_path)
        for name, value in metadata.items():
            mlflow.log_param(name, value)
        X_train, X_test, y_train, y_test = split_data(X, y, args.seed)
        model, messages = fit_model(build_pipeline(X_train, args), X_train, y_train)
        metrics, pred, prob = evaluate_model(model, X_test, y_test)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        mlflow.set_tag('converged', str(not messages).lower())
        folder = Path('outputs') / run.info.run_id
        folder.mkdir(parents=True, exist_ok=True)
        save_artifacts(folder, metrics, model, X_train, X_test, y_test, pred, prob, messages)
        for path in folder.iterdir():
            mlflow.log_artifact(str(path), artifact_path='evaluation')
        mlflow.log_artifact(__file__, artifact_path='code')
        mlflow.log_artifact('requirements.txt', artifact_path='code')
        mlflow.sklearn.log_model(model, artifact_path='model',
                                signature=infer_signature(X_train, model.predict(X_train)),
                                input_example=X_train.head(3),
                                pip_requirements=Path('requirements.txt').read_text().splitlines())
        print(json.dumps({'run_id': run.info.run_id, 'metrics': metrics, 'converged': not messages}, indent=2))


if __name__ == '__main__':
    main()
