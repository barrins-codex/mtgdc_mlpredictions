import time
from pathlib import Path

from mtgdc_decklists import ImportDecks
from mtgdc_mlpredictions import DataPreparation, RandomForest, XGBoost

DIRECTORY = Path(__file__).parent
XGBOOST_TRAINING = DIRECTORY / "xgboost_model.pkl"
RFOREST_TRAINING = DIRECTORY / "rforest_model.pkl"


def is_older_than_seven_days(file: Path) -> bool:
    threshold = time.time() - 7 * 24 * 60 * 60
    return file.is_file() and file.stat().st_mtime < threshold


def predict(links: str | list = ""):
    print(".", "Data Preparation")
    if not links:
        liste_decks = ImportDecks.from_file(DIRECTORY / "test.json")
        liste_decks.load_decks()
        data_preparation = DataPreparation()
        data_preparation.decks = liste_decks.decks

    else:
        data_preparation = DataPreparation(ImportDecks.from_moxfield(links))

    data_preparation.build_data()

    print(".", "XGBoost")
    if not XGBOOST_TRAINING.is_file() or is_older_than_seven_days(XGBOOST_TRAINING):
        print("----------", "Entrainement", "----------")
        xgboost_model = XGBoost.train()
        print("----------", "Fin", "----------")
    else:
        xgboost_model = XGBoost.load_from_model(XGBOOST_TRAINING)

    X, y = data_preparation.get_X_y()
    predictions = xgboost_model.predict(X)
    for idx, deck in enumerate(data_preparation.decks):
        print(
            "XGBoost",
            deck["commander"],
            ":",
            predictions[idx],
            "lands /",
            y[idx],
            "in reality",
        )

    print(".", "Random Forest")
    if not RFOREST_TRAINING.is_file() or is_older_than_seven_days(RFOREST_TRAINING):
        print("----------", "Entrainement", "----------")
        rforest_model = RandomForest.train()
        print("----------", "Fin", "----------")
    else:
        rforest_model = RandomForest.load_from_model(RFOREST_TRAINING)

    X, y = data_preparation.get_X_y()
    predictions = rforest_model.predict(X)
    for idx, deck in enumerate(data_preparation.decks):
        print(
            "Random Forest",
            deck["commander"],
            ":",
            predictions[idx],
            "lands /",
            y[idx],
            "in reality",
        )


if __name__ == "__main__":
    links = [
        # "https://www.moxfield.com/decks/fnD6JHj-I0Gb_cKausknKA",  # Ertai Resurrected
        "https://www.moxfield.com/decks/sswIvKF-9Uyiisu11z4gPw",  # Aragorn, King of Gondor
        # "https://www.moxfield.com/decks/HYtf4WoZ2EiLSok2gI1WRg",  # Ghyrson Starn, Kelermorph
        "https://www.moxfield.com/decks/qxlD4JeXxESYK5AYSlOEdg",  # Marath, Will of the Wild
    ]

    print("----------", "Prédictions sur des decks")
    predict(links=links)
