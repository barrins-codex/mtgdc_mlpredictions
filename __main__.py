from pathlib import Path

from mtgdc_decklists import ImportDecks
from mtgdc_mlpredictions import DataPreparation, RandomForest, XGBoost

DIRECTORY = Path(__file__).parent
XGBOOST_TRAINING = DIRECTORY / "xgboost_model.pkl"
RFOREST_TRAINING = DIRECTORY / "rdmforest_model.pkl"

def train():
    print(".", "Data Preparation")
    data_preparation = DataPreparation()
    data_preparation.load_decks()
    data_preparation.build_data()
    data_preparation.dump(DIRECTORY / "data_preparation.json")
    X_train, X_test, y_train, y_test = data_preparation.train_test_split()

    print(".", "XGBoost")
    xgboost_model = XGBoost()
    xgboost_model.import_parameters(X_train, X_test, y_train, y_test)
    xgboost_model.fit()
    xgboost_model.dump(XGBOOST_TRAINING)
    xgboost_metrics = xgboost_model.evaluate_model()
    print("XGBoost Metrics:", xgboost_metrics)

    print(".", "Random Forest")
    randomforest_model = RandomForest()
    randomforest_model.import_parameters(X_train, X_test, y_train, y_test)
    randomforest_model.fit()
    randomforest_model.dump(RFOREST_TRAINING)
    randomforest_metrics = randomforest_model.evaluate_model()
    print("Random Forest Metrics:", randomforest_metrics)


def predict(links: str | list = ""):
    if not all([XGBOOST_TRAINING.is_file(), RFOREST_TRAINING.is_file()]):
        print("----------", "Entrainement des modèles")
        train()

    print(".", "Data Preparation")
    if not links:
        liste_decks = ImportDecks.from_file(DIRECTORY / "test.json")
        liste_decks.load_decks()
        data_preparation = DataPreparation()
        data_preparation.decks = liste_decks.decks

    else:
        data_preparation = DataPreparation(ImportDecks.from_moxfield(links))

    data_preparation.build_data()
    data_preparation.dump(DIRECTORY / "data_test.json")

    print(".", "XGBoost")
    xgboost_model = XGBoost.load_from_model(XGBOOST_TRAINING)
    predictions = xgboost_model.predict(data_preparation.get_X_y()[0])
    for idx, deck in enumerate(data_preparation.decks):
        print("XGBoost", deck["commander"], ":", predictions[idx], "lands")

    print(".", "Random Forest")
    randomforest_model = RandomForest.load_from_model(RFOREST_TRAINING)
    predictions = randomforest_model.predict(data_preparation.get_X_y()[0])
    for idx, deck in enumerate(data_preparation.decks):
        print("Random Forest", deck["commander"], ":", predictions[idx], "lands")


if __name__ == "__main__":
    links = [
        "https://www.moxfield.com/decks/F9I34cGN4Uu5uazoB9394w",  # Ertai Resurrected
        "https://www.moxfield.com/decks/25qC36vOw0OLxDwBOBLEiA",  # Saskia from Barrin's Codex project
    ]

    print("----------", "Prédictions sur des decks")
    predict(links=links)
