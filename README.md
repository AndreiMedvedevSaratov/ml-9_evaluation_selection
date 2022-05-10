# 9 Evaluation Selection Project for RS School Machine Learning course.

I used [Forest Cover Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction/) dataset.

## Installation process:

This project allows you to train models for forest categories classification.

1. Clone this repository to your computer.

2. Download [Forest Cover Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction/) dataset, save csv locally (default path is *data/train.csv* in repository's root).

3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine.

4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. To generate dataset profiling report (EDA), use the following command:
```sh
poetry run profiling -d <path to csv with data>
```
or if you just want to visualize dataset:
```sh
poetry run dataset_gui -d <path to csv with data>
```

## Training process:

6. Run training process with the following commands:

- Logistic Regression:
```sh
poetry run train --model-selector 1 --max-iter 100 --logreg-c 10 --with-feature-selection 0
```
You can use different parameters.

- Random Forest Classifier:
```sh
poetry run train --model-selector 2 --n-estimators 100 --with-feature-selection 0
```

You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```

7. I tried to make --with-grid method to find best hyperparameters. But unfortunately it fails ((.
You can check possible methods in remarks.

## Making reports:

8. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```

Here is the results for experiments for Logistic Regression and Random Forest Classifier:
![results](info/experiments.png)

## Development

The code in this repository is tested, formatted with black, and pass mypy typechecking before being commited to the repository.

Install all requirements (including dev requirements) to poetry environment:
```sh
poetry install
```
Now you can use developer instruments, e.g. pytest:
```sh
poetry run pytest
```
![tests](info/tests.png)

Check your code annotation with mypy by using poetry:
```sh
poetry run mypy .
```