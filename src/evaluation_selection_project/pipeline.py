from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, VarianceThreshold  # type: ignore

def create_pipeline(
    use_scaler: bool, 
    max_iter: int, 
    logreg_C: float,
    random_state: int,
    with_feature_selection: int
) -> Pipeline:
    pipeline_steps = []

    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))

    if with_feature_selection == 1:
        pipeline_steps.append(
            (
                "with_feature_selection",
            )
        )

    pipeline_steps.append(
        (
            "classifier",
            LogisticRegression(
                random_state=random_state, max_iter=max_iter, C=logreg_C
            ),
        )
    )

    return Pipeline(steps=pipeline_steps)
