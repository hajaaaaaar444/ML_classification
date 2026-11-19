import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# Chemins des datasets
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "diabetes": PROJECT_ROOT / "data/raw/diabetes/diabetes.csv",
    "spam": PROJECT_ROOT / "data/raw/spam/spam.csv"
}


# 1. Charger les données
def load_data(dataset_name):
    file_path = DATASETS[dataset_name]

    df = pd.read_csv(file_path)

    return df


# 2. Préparer les données
def prepare_data(df, target_column):

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# 3. Créer le modèle
def create_model():

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    return model


# 4. Entraîner le modèle
def train_model(model, X_train, y_train):

    model.fit(X_train, y_train)

    return model


# 5. Évaluer le modèle
def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision :", precision_score(y_test, y_pred, zero_division=0))
    print("Recall :", recall_score(y_test, y_pred, zero_division=0))
    print("F1-score :", f1_score(y_test, y_pred, zero_division=0))

    print("\nClassification Report :")
    print(classification_report(y_test, y_pred))

    return y_pred


# 6. Sauvegarder le modèle
def save_model(model, dataset_name):

    output_path = PROJECT_ROOT / "models" / f"{dataset_name}_logistic.pkl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path)

    print(f"Modèle sauvegardé : {output_path}")


# Workflow complet
def run_workflow(dataset_name, target_column):

    print(f"\n===== DATASET : {dataset_name.upper()} =====")

    df = load_data(dataset_name)

    X_train, X_test, y_train, y_test = prepare_data(
        df,
        target_column
    )

    model = create_model()

    model = train_model(model, X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model, dataset_name)


if __name__ == "__main__":

    run_workflow("diabetes", "Outcome")