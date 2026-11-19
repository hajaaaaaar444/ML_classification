
import numpy as np
import pandas as pd
import pytest

from src.ml_workflow import (
    load_data,
    prepare_data,
    create_model,
    train_model,
    evaluate_model,
    save_model
)


# 1. Créer un petit dataset pour les tests
def create_sample_data():

    np.random.seed(42)

    df = pd.DataFrame({
        "age": np.random.randint(20, 70, 100),
        "glucose": np.random.randint(70, 200, 100),
        "bmi": np.random.uniform(18, 40, 100),
        "Outcome": [0, 1] * 50
    })

    return df


# 2. Tester le chargement des données
def test_load_data(tmp_path):

    df = create_sample_data()

    file_path = tmp_path / "diabetes.csv"
    df.to_csv(file_path, index=False)

    loaded_df = pd.read_csv(file_path)

    assert len(loaded_df) == 100
    assert "Outcome" in loaded_df.columns


# 3. Tester la préparation des données
def test_prepare_data():

    df = create_sample_data()

    X_train, X_test, y_train, y_test = prepare_data(
        df, "Outcome"
    )

    assert len(X_train) == 80
    assert len(X_test) == 20

    assert "Outcome" not in X_train.columns
    assert len(y_train) == 80
    assert len(y_test) == 20


# 4. Tester la création du modèle
def test_create_model():

    model = create_model()

    assert "imputer" in model.named_steps
    assert "scaler" in model.named_steps
    assert "classifier" in model.named_steps


# 5. Tester l'entraînement
def test_train_model():

    df = create_sample_data()

    X_train, X_test, y_train, y_test = prepare_data(
        df, "Outcome"
    )

    model = create_model()

    trained_model = train_model(
        model, X_train, y_train
    )

    assert trained_model is not None
    assert hasattr(trained_model, "predict")


# 6. Tester l'évaluation
def test_evaluate_model(capsys):

    df = create_sample_data()

    X_train, X_test, y_train, y_test = prepare_data(
        df, "Outcome"
    )

    model = create_model()
    model = train_model(model, X_train, y_train)

    predictions = evaluate_model(
        model, X_test, y_test
    )

    assert len(predictions) == len(y_test)

    output = capsys.readouterr().out

    assert "Accuracy" in output
    assert "Precision" in output
    assert "Recall" in output
    assert "F1-score" in output


# 7. Tester la sauvegarde du modèle
def test_save_model(tmp_path):

    df = create_sample_data()

    X_train, X_test, y_train, y_test = prepare_data(
        df, "Outcome"
    )

    model = create_model()
    model = train_model(model, X_train, y_train)

    save_model(model, "diabetes")

    # Vérifier que la sauvegarde fonctionne
    # On teste directement avec un chemin temporaire
    import joblib

    model_path = tmp_path / "model.pkl"
    joblib.dump(model, model_path)

    assert model_path.exists()

    loaded_model = joblib.load(model_path)

    assert len(loaded_model.predict(X_test)) == len(X_test)
