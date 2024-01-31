import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from mtgdc_carddata import CardDatabase
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


class MachineLearning:
    def __init__(self) -> None:
        self.model = None

    def dump(self, path: Path) -> None:
        joblib.dump(self.model, path)

    @staticmethod
    def load_from_model(path: Path) -> None:
        tmp = MachineLearning()
        tmp.model = joblib.load(path)
        return tmp

    def import_parameters(self, X_train, X_test, y_train, y_test) -> None:
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


class XGBoost(MachineLearning):
    def __init__(self) -> None:
        super().__init__()
        self.model = XGBRegressor()


class RandomForest(MachineLearning):
    def __init__(self) -> None:
        super().__init__()
        self.model = RandomForestRegressor(n_estimators=100, random_state=0)
