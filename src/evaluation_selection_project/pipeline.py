from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, VarianceThreshold  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore


def create_pipeline_Logistic_Regression(
    with_scaler: bool, 
    max_iter: int, 
    logreg_C: float,
    with_feature_selection: int,
    with_grid: bool,
    random_state: int,
) -> Pipeline:
    pipeline_steps = []

    if with_scaler:
        pipeline_steps.append(
            (
                "scaler", 
                StandardScaler()
            )
        )

    if with_feature_selection == 1:
        pipeline_steps.append(
            (
                "with_feature_selection",
                SelectFromModel(RandomForestClassifier(random_state=2022)),
            )
        )

    if with_feature_selection == 2:
        pipeline_steps.append(
            (
                "with_feature_selection", 
            VarianceThreshold(threshold=0.20)
            )
        )

    if with_grid is True:
        pipeline_steps.append(
            (
                "classifier",
                LogisticRegression(random_state=random_state),
            )
        )

    if with_grid is False:
        pipeline_steps.append(
            (
                "classifier",
                LogisticRegression(random_state=random_state, max_iter=max_iter, C=logreg_C),
            )
        )

    return Pipeline(steps=pipeline_steps)


def create_pipeline_RandomForest(
    with_scaler: bool,
    n_estimators: int,
    with_feature_selection: int,
    grid_search: bool,
    random_state: int,
) -> Pipeline:
    pipeline_steps = []

    if with_scaler:
        pipeline_steps.append(
            (
                "scaler", 
                StandardScaler()
            )
        )

    if with_feature_selection == 1:
        pipeline_steps.append(
            (
                "with_feature_selection",
                SelectFromModel(RandomForestClassifier(random_state=2022)),
            )
        )

    if with_feature_selection == 2:
        pipeline_steps.append(
            (
                "with_feature_selection", 
                VarianceThreshold(threshold=0.20)
            )
        )

    if grid_search is False:
        pipeline_steps.append(
            (
                "classifier",
                RandomForestClassifier(n_estimators=n_estimators, random_state=random_state),
            )
        )
        
    if grid_search is True:
        pipeline_steps.append(
            (
                "classifier",
                RandomForestClassifier(random_state=random_state),
            )
        )
    return Pipeline(steps=pipeline_steps)
