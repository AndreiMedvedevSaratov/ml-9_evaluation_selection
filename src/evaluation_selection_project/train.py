from pathlib import Path
from joblib import dump

import click
import mlflow  # type: ignore[unused-ignore]
import mlflow.sklearn  # type: ignore[unused-ignore]

from sklearn.model_selection import KFold  # type: ignore[unused-ignore]
from sklearn.model_selection import GridSearchCV  # type: ignore[unused-ignore]
from sklearn.model_selection import cross_val_score  # type: ignore[unused-ignore]

from .data import get_dataset
from .pipeline import create_pipeline_Logistic_Regression
from .pipeline import create_pipeline_RandomForest

@click.command()

@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)

@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)

@click.option(
    "--random-state",
    default=42,
    type=int,
)

@click.option(
    "--test-split-ratio",
    default=10,
    type=int,
)

@click.option(
    "--with-scaler",
    default=True,
    type=bool,
)

@click.option(
    "--max-iter",
    default=1000,
    type=int,
)

@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
)

@click.option(
    "--with-feature-selection", 
    default=0, 
    type=int)

@click.option(
    "--with-grid",
    default=False,
    type=bool)

@click.option(
    "--model-selector", 
    default=2, 
    type=int)

@click.option(
    "--n-estimators", 
    default=100, 
    type=int
    )

def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: int,
    with_scaler: bool,
    max_iter: int,
    logreg_c: float,
    with_feature_selection: int,
    model_selector: int,
    with_grid: bool,
    n_estimators: int,
) -> None:
    features, target = get_dataset(
        dataset_path
    )

    with mlflow.start_run():
        if model_selector == 1:
            pipeline = create_pipeline_Logistic_Regression(
                with_scaler, 
                max_iter, 
                logreg_c,
                with_feature_selection,
                with_grid,
                random_state,
            )

            mlflow.log_param("Selected Model", "Logistic Regression")

            if with_grid is False:
                mlflow.log_param("max_iter", max_iter)
                mlflow.log_param("logreg_c", logreg_c)

            if with_grid is True:
                cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
                param_grid = {
                    "penalty": ["l1", "l2"],
                    "C": [1, 10, 100],
                    "solver": ["newton-cg", "lbfgs", "liblinear"],
                    "max_iter": [100, 200, 300, 500],
                }
                scoring = [
                    "accuracy",
                    "f1_weighted",
                    "precision_weighted",
                    "recall_weighted",
                ]
                search = GridSearchCV(
                    pipeline,
                    param_grid=param_grid,
                    n_jobs=1,
                    cv=cv_inner,
                    scoring=scoring,
                    refit="f1_weighted",
                )

            cv = KFold(n_splits=test_split_ratio,
                    shuffle=True, random_state=random_state)

            if with_grid is True:
                f1 = cross_val_score(
                search, features, target, scoring="f1_weighted", cv=cv, n_jobs=1).mean()
                mlflow.log_param("max_iter", search.best_params_["max_iter"])
                mlflow.log_param("logreg_c", search.best_params_["C"])
                search.fit(features, target)
            else:
                f1 = cross_val_score(
                    pipeline, features, target, scoring="f1_weighted", cv=cv, n_jobs=1).mean()
                pipeline.fit(features, target)
            
            if with_feature_selection == 0:
                mlflow.log_param("with_feature_selection", "None")

            if with_feature_selection == 1:
                mlflow.log_param("with_feature_selection", "SelectFromModel")

            if with_feature_selection == 2:
                mlflow.log_param("with_feature_selection", "VarianceThreshold")

            mlflow.log_param("with_scaler", with_scaler)
            mlflow.log_param("splits", test_split_ratio)
            
            mlflow.log_metric("f1-score", f1)
            click.echo(f"f1-score: {f1}.")

            dump(pipeline, save_model_path)

            click.echo(f"Model is saved to {save_model_path}.")
        
        if model_selector == 2:
            pipeline = create_pipeline_RandomForest(
                with_scaler,
                n_estimators,
                with_feature_selection,
                with_grid,
                random_state,
            )

            mlflow.log_param("Selected Model", "Random Forest Classifier")

            if with_grid is False:
                mlflow.log_param("n_estimators", n_estimators)

            if with_grid is True:
                cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
                param_grid = {
                    "classifier__n_estimators": [10, 100, 150, 200],
                    "classifier__max_features": ["auto", "sqrt", "log2"],
                    "classifier__max_depth": [2, 4, 5, 6, None],
                    "classifier__criterion": ["gini", "entropy"],
                }
                scoring = [
                    "accuracy",
                    "f1_weighted",
                    "precision_weighted",
                    "recall_weighted",
                ]
                search = GridSearchCV(
                    pipeline,
                    param_grid=param_grid,
                    n_jobs=1,
                    cv=cv_inner,
                    scoring=scoring,
                    refit="f1_weighted",
                )

            cv = KFold(n_splits=test_split_ratio, shuffle=True, random_state=random_state)

            if with_grid is True:
                f1 = cross_val_score(search, features, target, scoring="f1_weighted", cv=cv, n_jobs=1).mean()
                mlflow.log_param("n_estimators", search.best_params_["n_estimators"])
                search.fit(features, target)
            else:
                f1 = cross_val_score(pipeline, features, target, scoring="f1_weighted", cv=cv, n_jobs=1).mean()
                pipeline.fit(features, target)

            if with_feature_selection == 0:
                mlflow.log_param("with_feature_selection", "None")

            if with_feature_selection == 1:
                mlflow.log_param("with_feature_selection", "SelectFromModel")

            if with_feature_selection == 2:
                mlflow.log_param("with_feature_selection", "VarianceThreshold")

            mlflow.log_param("with_scaler", with_scaler)
            mlflow.log_param("splits", test_split_ratio)

            mlflow.log_metric("f1-score", f1)
            click.echo(f"f1-score: {f1}.")

            dump(pipeline, save_model_path)

            click.echo(f"Model is saved to {save_model_path}.")
