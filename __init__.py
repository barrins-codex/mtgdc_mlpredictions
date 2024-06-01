import json
from datetime import datetime
from pathlib import Path

import joblib
import optuna
import pandas as pd
from mtgdc_carddata import CardDatabase
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from mtgdc_decklists import ImportDecks

DATABASE = CardDatabase()


class DataPreparation:

    data = {
        "NombreTerrains": [],
        "CoutManaCommandant": [],
        "NombreCouleursDeck": [],
        "CoutMoyenManaDeck": [],
        "NombreCarteMana1": [],
        "NombreCarteMana2": [],
        "NombreCarteMana3": [],
        "NombreCarteMana4": [],
    }

    def __init__(self, liste: ImportDecks = ImportDecks()) -> None:
        if len(liste.decks) > 0:
            self.decks = liste.decks

    def load_decks(self) -> None:
        liste_decks = ImportDecks.from_directory("mtgdc_decklists/decklists")
        liste_decks.load_decks(datetime(2018, 1, 1))
        self.decks = liste_decks.decks

    def build_data(self) -> None:
        for deck in self.decks:
            if ("Unknown Card" in deck["cardlist"]) or (len(deck["cardlist"]) < 10):
                continue

            deck_totals = self._analyze_deck(deck)

            czone_cmc = (
                sum(deck_totals["CoutCommandant"]) / len(deck_totals["CoutCommandant"])
                if len(deck_totals["CoutCommandant"]) > 0
                else 0
            )
            deck_cmc = (
                sum(deck_totals["Cout"]) / (99 - deck_totals["Terrains"])
                if deck_totals["Terrains"] < 99
                else 0
            )

            self.data["NombreTerrains"].append(deck_totals["Terrains"])
            self.data["CoutManaCommandant"].append(czone_cmc)
            self.data["NombreCouleursDeck"].append(len(set(deck_totals["Couleurs"])))
            self.data["CoutMoyenManaDeck"].append(deck_cmc)

            for i in range(1, 4 + 1):
                self.data[f"NombreCarteMana{i}"].append(
                    max(0, len([val for val in deck_totals["Cout"] if val == i]))
                )

    def _analyze_deck(self, deck: dict) -> dict:
        tmp = {"Terrains": 0, "CoutCommandant": [], "Couleurs": [], "Cout": []}

        for card_name in deck["commander"]:
            card = DATABASE.card(card_name)
            if DATABASE.has_been_commander(card):
                tmp["CoutCommandant"].append(card["convertedManaCost"])
                tmp["Cout"].append(card["convertedManaCost"])
                tmp["Couleurs"].extend(card.get("colors", []))

        for qty, card_name in deck["decklist"]:
            if card_name in deck["commander"]:
                continue

            card = DATABASE.card(card_name)
            if "land" in card["type"].lower():
                tmp["Terrains"] += qty
            else:
                tmp["Cout"].append(card["convertedManaCost"])
                tmp["Couleurs"].extend(card.get("colors", []))

        return tmp

    def dump(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.data, file)

    def get_X_y(self) -> tuple:
        df = pd.DataFrame(self.data)

        X = df[
            [
                "CoutManaCommandant",
                "CoutMoyenManaDeck",
                "NombreCouleursDeck",
                "NombreCarteMana1",
                "NombreCarteMana2",
                "NombreCarteMana3",
                "NombreCarteMana4",
            ]
        ]
        y = df["NombreTerrains"]

        return X, y

    def train_test_split(self) -> list:
        X, y = self.get_X_y()
        return train_test_split(X, y, test_size=0.2, random_state=0)

    @staticmethod
    def load_and_build_data(liste: ImportDecks = ImportDecks()) -> tuple:
        data_preparation = DataPreparation(liste)
        data_preparation.load_decks()
        data_preparation.build_data()

        return data_preparation.train_test_split()


class MachineLearning:
    def __init__(self, modele) -> None:
        self.model = modele
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def dump(self, path: Path) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load_from_model(cls, path: Path) -> None:
        tmp = cls()
        tmp.model = joblib.load(path)
        return tmp

    def _import_parameters(self, X_train, X_test, y_train, y_test) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self) -> None:
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)

    def evaluate_model(self) -> str:
        mae = metrics.mean_absolute_error(self.y_test, self.y_pred)
        mse = metrics.mean_squared_error(self.y_test, self.y_pred)
        r2 = metrics.r2_score(self.y_test, self.y_pred)

        return [mae, mse, r2]

    def predict(self, X) -> list:
        return self.model.predict(X)


class ModelOptimization(MachineLearning):
    def objective(self, trial):
        # Unique appel à DataPreparation pour accéler l'exécution du script
        if self.X_train is None:
            self._import_parameters(*DataPreparation.load_and_build_data())

        params = {  # Paramètres à optimiser avec Optuna
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        # Paramètre inexistant pour Random Forest
        if self.model.__class__ == XGBRegressor:
            params["learning_rate"] = trial.suggest_float(
                "learning_rate", 0.01, 0.1, log=True
            )
            params["subsample"] = (trial.suggest_float("subsample", 0.5, 1.0),)

        # On réutilise le modèle utilisé lors de l'appel de la classe héritière
        self.model = self.model.__class__(**params)
        self.fit()

        # Prédictions sur l'ensemble de test
        y_pred = self.predict(self.X_test)
        metric = metrics.mean_absolute_error(self.y_test, y_pred)

        return metric

    def optimize_parameters(self):
        dir = Path(__file__).parent
        prefix = "xgboost_" if self.model.__class__ == XGBRegressor else "rforest_"

        study = optuna.create_study(
            direction="minimize", study_name=(prefix + "optimization")
        )
        study.optimize(self.objective, n_trials=50)  # Nombre d'essais à ajuster
        joblib.dump(study, dir / (prefix + "study.pkl"))

        # Meilleurs paramètres trouvés par Optuna
        best_params = study.best_params
        del study  # Libération de la mémoire

        print("Meilleurs paramètres trouvés:", best_params)
        with open(dir / (prefix + "best_params.json"), "w", encoding="utf-8") as file:
            json.dump(best_params, file, indent=4)

        # Utiliser les meilleurs paramètres pour entraîner le modèle final
        self.model = self.model.__class__(**best_params)


class XGBoost(ModelOptimization):
    def __init__(self) -> None:
        super().__init__(XGBRegressor())

    @classmethod
    def train(cls):
        xgboost_model = cls()
        xgboost_model.optimize_parameters()
        xgboost_model.fit()

        print("XGBoost Metrics:", xgboost_model.evaluate_model())

        dir = Path(__file__).parent
        xgboost_model.dump(dir / "xgboost_model.pkl")

        return xgboost_model


class RandomForest(MachineLearning):
    def __init__(self, training: bool = False) -> None:
        super().__init__(RandomForestRegressor(n_estimators=100, random_state=0))

    @classmethod
    def train(cls):
        rforest_model = cls()
        rforest_model._import_parameters(*DataPreparation.load_and_build_data())
        rforest_model.fit()

        print("Random Forest Metrics:", rforest_model.evaluate_model())

        dir = Path(__file__).parent
        rforest_model.dump(dir / "rforest_model.pkl")

        return rforest_model
