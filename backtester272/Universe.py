import requests
import pandas as pd
from typing import List, Dict, Union
from backtester272.DataBase import DataBase


class Universe:
    """
    Classe pour interagir avec l'API CoinGecko et une base de données locale pour récupérer et structurer
    des données de marché sur les cryptomonnaies et les actions.
    """

    def __init__(self, api_key: str = None, api_secret: str = None, verbose: bool = False) -> None:
        """
        Initialise l'instance de la classe Universe.

        Args:
            api_key (str, optional): Clé API pour Binance ou autres services. Par défaut None.
            api_secret (str, optional): Clé secrète API pour Binance. Par défaut None.
            verbose (bool, optional): Active les messages d'information si True. Par défaut False.
        """
        self.verbose = verbose
        self.db = DataBase(api_key, api_secret, self.verbose)

    def get_crypto_symbols(self, categories: Union[List[str], str], nb_actif: int = 10, format: str = "list") -> Union[List[str], Dict[str, List[str]]]:
        """
        Récupère les symboles de cryptomonnaies depuis l'API CoinGecko en fonction de leurs catégories.

        Args:
            categories (Union[List[str], str]): Liste ou chaîne de caractères représentant les catégories à récupérer.
            nb_actif (int, optional): Nombre maximum de cryptomonnaies à récupérer par catégorie. Par défaut 10.
            format (str, optional): Format des données renvoyées ("list" pour une liste ou "dict" pour un dictionnaire par catégorie). Par défaut "list".

        Returns:
            Union[List[str], Dict[str, List[str]]]: Liste des symboles ou dictionnaire des symboles par catégorie.
        """
        categories = categories if isinstance(categories, list) else [categories]

        # URL de l'API CoinGecko pour les marchés
        coingecko_markets_url = 'https://api.coingecko.com/api/v3/coins/markets'
        data_merged = pd.DataFrame()

        if self.verbose:
            print(f"Récupération des symboles pour les catégories : {categories}")

        for category in categories:
            # Paramètres de la requête
            params = {
                'vs_currency': 'usd',
                'category': category,
                'order': 'market_cap_desc',
                'per_page': nb_actif,
                'page': 1,
                'sparkline': False,
                'price_change_percentage': '24h',
                'locale': 'en',
                'precision': 3
            }

            response = requests.get(coingecko_markets_url, params=params)
            data_json = response.json()

            if isinstance(data_json, list) and len(data_json) > 0:
                # Convertir les données JSON en DataFrame
                data = pd.DataFrame(data_json)
                data['symbol'] = data['symbol'].str.upper() + 'USDT'  # Ajouter 'USDT' pour Binance
                data['category'] = category
                data = data[['id', 'symbol', 'name', 'current_price', 'market_cap', 'market_cap_rank', 'category']]
                data_merged = pd.concat([data_merged, data], ignore_index=True)
            else:
                print(f"Erreur ou aucune donnée pour la catégorie {category}: {data_json}")

        if format == "list":
            # Renvoie une liste unique des symboles
            return list(data_merged['symbol'].unique())

        if format == "dict":
            # Renvoie un dictionnaire par catégorie
            category_dict = {}
            for index, row in data_merged.iterrows():
                category = row['category']
                symbol = row['symbol']

                if category in category_dict:
                    category_dict[category].append(symbol)
                else:
                    category_dict[category] = [symbol]

            return category_dict

    def get_crypto_prices(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Récupère les prix historiques des cryptomonnaies à partir de la base de données ou de Binance.

        Args:
            symbols (List[str]): Liste des symboles à récupérer.
            start_date (str): Date de début au format 'YYYY-MM-DD'.
            end_date (str): Date de fin au format 'YYYY-MM-DD'.

        Returns:
            pd.DataFrame: Données historiques des prix des cryptomonnaies.
        """
        if self.verbose:
            print(f"Récupération des prix pour les symboles {symbols} de {start_date} à {end_date}.")
        self.db.update_database(symbols, start_date, end_date, 'binance')
        return self.db.get_data(symbols, start_date, end_date)

    def get_equity_prices(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Récupère les prix historiques des actions à partir de la base de données ou de Yahoo Finance.

        Args:
            symbols (List[str]): Liste des symboles à récupérer.
            start_date (str): Date de début au format 'YYYY-MM-DD'.
            end_date (str): Date de fin au format 'YYYY-MM-DD'.

        Returns:
            pd.DataFrame: Données historiques des prix des actions.
        """
        if self.verbose:
            print(f"Récupération des prix pour les actions {symbols} de {start_date} à {end_date}.")
        self.db.update_database(symbols, start_date, end_date, 'yfinance')
        return self.db.get_data(symbols, start_date, end_date)