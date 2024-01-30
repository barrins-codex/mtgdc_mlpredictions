from pathlib import Path

from mtgdc_decklists import ImportDecks
from mtgdc_mlpredictions import DataPreparation, RandomForest, XGBoost

DIRECTORY = Path(__file__).parent


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
    xgboost_model.dump(DIRECTORY / "xgboost_model.pkl")
    xgboost_metrics = xgboost_model.evaluate_model()

    print(".", "Random Forest")
    randomforest_model = RandomForest()
    randomforest_model.import_parameters(X_train, X_test, y_train, y_test)
    randomforest_model.fit()
    randomforest_model.dump(DIRECTORY / "rdmforest_model.pkl")
    randomforest_metrics = randomforest_model.evaluate_model()

    print("XGBoost Metrics:", xgboost_metrics)
    print("Random Forest Metrics:", randomforest_metrics)


def predict():
    print(".", "Data Preparation")
    liste_decks = ImportDecks.from_file(DIRECTORY / "test.json")
    liste_decks.load_decks()
    data_preparation = DataPreparation()
    data_preparation.decks = liste_decks.decks
    data_preparation.build_data()
    data_preparation.dump(DIRECTORY / "data_test.json")

    print(".", "XGBoost")
    xgboost_model = XGBoost.load_from_model(DIRECTORY / "xgboost_model.pkl")
    predictions = xgboost_model.predict(data_preparation.get_X_y()[0])
    for idx, deck in enumerate(liste_decks.decks):
        print("XGBoost", deck["commander"], ":", predictions, "lands")

    print(".", "Random Forest")
    xgboost_model = RandomForest.load_from_model(DIRECTORY / "rdmforest_model.pkl")
    predictions = xgboost_model.predict(data_preparation.get_X_y()[0])
    for idx, deck in enumerate(liste_decks.decks):
        print("Random Forest", deck["commander"], ":", predictions, "lands")


if __name__ == "__main__":
    print("----------", "Entrainement des modèles")
    train()

    print("----------", "Prédictions sur des decks")
    predict()  # Needs to be revised
