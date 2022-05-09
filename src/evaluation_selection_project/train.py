from pathlib import Path
from joblib import dump

import click
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score

from .data import get_dataset
from .pipeline import create_pipeline_Logistic_Regression

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)
@click.option(
    "--with-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--max-iter",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True,
)
@click.option(
    "--with-feature-selection", 
    default=2, 
    type=int)
@click.option(
    "--with-grid",
    default=True,
    type=bool)
@click.option(
    "--model-selector", 
    default=1, 
    type=int)

def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    max_iter: int,
    logreg_c: float,
    with_scaler: bool,
    with_feature_selection: int,
    with_grid: bool,
    model_selector: int,
) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
    )

    with mlflow.start_run():
        if model_selector == 1:
            pipeline = create_pipeline_Logistic_Regression(
                with_scaler, 
                max_iter, 
                logreg_c,
                random_state,
                with_feature_selection,
                with_grid)

            mlflow.log_param("Selected Model", "Logistic Regression")

            pipeline.fit(features_train, target_train)

            accuracy = accuracy_score(target_val, pipeline.predict(features_val))
            
            mlflow.log_param("with_scaler", with_scaler)
            mlflow.log_param("max_iter", max_iter)
            mlflow.log_param("logreg_c", logreg_c)
            mlflow.log_metric("accuracy", accuracy)

            click.echo(f"Accuracy: {accuracy}.")

            dump(pipeline, save_model_path)

            click.echo(f"Model is saved to {save_model_path}.")