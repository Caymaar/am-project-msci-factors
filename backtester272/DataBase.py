import pandas as pd
import os
from binance.client import Client
import yfinance as yf
from typing import List, Tuple, Optional


class DataBase:
    """
    Classe pour gérer une base de données financière avec des fonctionnalités d'intégration
    de données depuis Binance et Yahoo Finance.

    Cette classe offre des fonctionnalités de mise à jour, de suppression, et de récupération
    de données financières historiques.
    """

    def __init__(self, api_key: str = None, api_secret: str = None, verbose: bool = False) -> None:
        """
        Initialise la base de données et tente une connexion à Binance.

        Si la connexion échoue, la base de données fonctionne en mode hors-ligne.

        Args:
            api_key (str, optional): Clé API pour Binance.
            api_secret (str, optional): Secret API pour Binance.
            verbose (bool): Active les messages de débogage si True.
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.verbose = verbose
        self.is_online = False

        # Tentative de connexion à Binance
        try:
            self.client = Client(self.api_key, self.api_secret)
            self.is_online = True
            if self.verbose:
                print("Connexion à Binance établie.")
        except Exception as e:
            if self.verbose:
                print(f"Impossible de se connecter à Binance: {e}")
                print("La base de données fonctionnera en mode hors-ligne.")

        if self.verbose:
            print("Initialisation de la base de données...")

        # Chargement ou création de la base de données locale
        self.load_database()

    def load_database(self) -> None:
        """
        Charge ou initialise une base de données locale à partir d'un fichier CSV.

        Si le fichier n'existe pas, un fichier vide est créé.
        """
        directory = 'data'
        database_file = 'database.csv'

        self.file_path = os.path.join(directory, database_file)

        # Créer le répertoire s'il n'existe pas
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Créer un fichier vide si la base de données n'existe pas
        if not os.path.exists(self.file_path):
            df = pd.DataFrame(columns=['Date'])
            df.to_csv(self.file_path, index=False)

        # Charger les données dans un DataFrame
        self.database = pd.read_csv(self.file_path, index_col='Date', parse_dates=True)

    def get_historical_close(self, symbols: List[str], start_date: str, end_date: str, backend: str) -> Optional[pd.DataFrame]:
        """
        Récupère les données de clôture historiques pour les symboles donnés.

        Args:
            symbols (List[str]): Liste des symboles à récupérer.
            start_date (str): Date de début (format YYYY-MM-DD).
            end_date (str): Date de fin (format YYYY-MM-DD).
            backend (str): Source des données ('binance' ou 'yfinance').

        Returns:
            pd.DataFrame or None: Données de clôture ou None en cas d'échec.
        """
        if backend == 'binance':
            return self.get_binance_historical_close(symbols, start_date, end_date)
        elif backend == 'yfinance':
            return self.get_yfinance_historical_close(symbols, start_date, end_date)
        else:
            raise ValueError("Backend non supporté. Utilisez 'binance' ou 'yfinance'.")

    def get_binance_historical_close(self, symbols: List[str], start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Récupère les données de clôture historiques depuis Binance.

        Args:
            symbols (List[str]): Liste des symboles à récupérer.
            start_date (str): Date de début (format YYYY-MM-DD).
            end_date (str): Date de fin (format YYYY-MM-DD).

        Returns:
            pd.DataFrame or None: Données de clôture ou None en cas d'échec.
        """
        try:
            data = {}

            # Boucle sur chaque symbole pour récupérer les données
            for symbol in symbols:
                klines = self.client.get_historical_klines(
                    symbol,
                    Client.KLINE_INTERVAL_1DAY,
                    start_date,
                    end_date
                )

                # Extraire les dates et prix de clôture
                close_data = [(pd.to_datetime(kline[0], unit='ms'), float(kline[4])) for kline in klines]
                df = pd.DataFrame(close_data, columns=['date', 'close']).set_index('date')
                data[symbol] = df['close']

            # Combiner les DataFrames pour tous les symboles
            result_df = pd.concat(data.values(), axis=1, keys=data.keys())
            result_df.index.name = 'Date'

            return result_df
        except Exception as e:
            if self.verbose:
                print(f"Erreur lors de la récupération des données Binance : {e}")
            return None

    def get_yfinance_historical_close(self, symbols: List[str], start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Récupère les données de clôture historiques depuis Yahoo Finance.

        Args:
            symbols (List[str]): Liste des symboles à récupérer.
            start_date (str): Date de début (format YYYY-MM-DD).
            end_date (str): Date de fin (format YYYY-MM-DD).

        Returns:
            pd.DataFrame or None: Données de clôture ou None en cas d'échec.
        """
        data = yf.download(symbols, start=start_date, end=end_date, progress=self.verbose)
        return data['Close']

    def get_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Extrait les données des symboles spécifiés entre deux dates.

        Args:
            symbols (List[str]): Liste des symboles à récupérer.
            start_date (str): Date de début au format 'YYYY-MM-DD'.
            end_date (str): Date de fin au format 'YYYY-MM-DD'.

        Returns:
            pd.DataFrame: Données filtrées contenant les symboles valides entre les dates spécifiées.
        """
        # Listes pour les symboles valides et invalides
        valid_symbols = []
        invalid_symbols = []

        # Filtrer les symboles selon leur présence dans la base de données
        for s in symbols:
            if s not in self.notlisted:
                if s in self.database.columns:  # Vérifie si le symbole est dans la base
                    valid_symbols.append(s)
                else:  # Ajoute à la liste des invalides si absent
                    invalid_symbols.append(s)
                    if self.verbose:
                        print(f"Le symbole {s} n'est pas présent dans la base de données")

        # Vérifie s'il y a au moins un symbole valide
        if not valid_symbols:
            if self.verbose:
                print("Aucun symbole valide trouvé.")
            return pd.DataFrame()

        # Conversion des dates en objets datetime pour garantir la compatibilité
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Filtrer les données dans l'intervalle de dates pour les symboles valides
        filtered_data = self.database.loc[start_date:end_date, valid_symbols]

        # Supprimer les lignes où toutes les valeurs sont NaN
        filtered_data = filtered_data.dropna(how='all')

        # Message verbose pour confirmer l'extraction
        if self.verbose:
            print(f"Données extraites pour {len(valid_symbols)} symboles du {start_date.date()} au {end_date.date()}")

        return filtered_data

    def save_database(self) -> None:
        """
        Sauvegarde la base de données actuelle dans un fichier CSV.
        """
        self.database = self.database.sort_index()
        self.database.to_csv(self.file_path, index=True)
        if self.verbose:
            print("Base de données sauvegardée.")

    def _get_symbol_date_range(self, symbol: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Récupère la plage de dates (première et dernière) pour un symbole spécifique 
        dans la base de données.

        Args:
            symbol (str): Le symbole pour lequel récupérer la plage de dates.

        Returns:
            Tuple[Optional[str], Optional[str]]:
                - Première date valide (format 'YYYY-MM-DD').
                - Dernière date valide (format 'YYYY-MM-DD').
                - Si le symbole n'existe pas ou n'a pas de données valides, retourne (None, None).
        """
        if symbol not in self.database.columns:
            # Le symbole n'est pas présent dans la base de données
            if self.verbose:
                print(f"Le symbole '{symbol}' n'existe pas dans la base de données.")
            return None, None

        # Obtenir les premières et dernières dates valides
        first_date = self.database[symbol].first_valid_index()
        last_date = self.database[symbol].last_valid_index()

        if first_date is None or last_date is None:
            # Si aucune donnée valide n'est trouvée
            if self.verbose:
                print(f"Le symbole '{symbol}' n'a pas de données valides dans la base.")
            return None, None

        # Retourner les dates au format 'YYYY-MM-DD'
        return first_date.strftime('%Y-%m-%d'), last_date.strftime('%Y-%m-%d')

    def update_database(self, symbols: List[str], start_date: str, end_date: str, backend: str) -> List[str]:
        """
        Met à jour la base de données avec des données manquantes pour les symboles spécifiés.

        Args:
            symbols (List[str]): Liste des symboles à mettre à jour.
            start_date (str): Date de début au format 'YYYY-MM-DD'.
            end_date (str): Date de fin au format 'YYYY-MM-DD'.
            backend (str): Source des données ("binance" ou "yfinance").

        Returns:
            List[str]: Liste des symboles qui n'ont pas pu être mis à jour (notlisted).
        """
        self.notlisted = []  # Réinitialisation de la liste des symboles introuvables
        modified = False  # Indicateur de modification de la base de données

        if not self.is_online:
            # Vérification du mode hors ligne
            if self.verbose:
                print("Base de données en mode hors ligne. Mise à jour impossible.")
            return self.notlisted

        for symbol in symbols:
            if self.verbose:
                print(f"Vérification des données pour {symbol}...")

            # Obtenir la plage de dates actuelle pour le symbole
            min_date, max_date = self._get_symbol_date_range(symbol)

            # Déterminer si des données supplémentaires sont nécessaires
            if min_date is None or pd.to_datetime(max_date) < pd.to_datetime(end_date):
                # Définir la plage de dates à récupérer
                new_start_date = max_date if max_date else start_date
                if self.verbose:
                    print(f"Récupération des données pour {symbol} de {new_start_date} à {end_date}...")

                # Récupérer les données historiques manquantes
                new_data = self.get_historical_close([symbol], new_start_date, end_date, backend)

                if new_data is None:
                    # Ajouter le symbole à la liste des introuvables si les données ne sont pas disponibles
                    if self.verbose:
                        print(f"Les données pour {symbol} ne sont pas disponibles.")
                    self.notlisted.append(symbol)
                    continue

                # Ajouter les nouvelles données à la base de données
                self.database = self.database.combine_first(new_data)
                modified = True

                if self.verbose:
                    print(f"Données mises à jour pour {symbol} ({new_start_date} - {end_date}).")
            else:
                if self.verbose:
                    print(f"Les données pour {symbol} sont déjà à jour.")

        # Sauvegarder la base de données si des modifications ont été apportées
        if modified:
            self.save_database()
            if self.verbose:
                print("Base de données sauvegardée avec les nouvelles mises à jour.")
        else:
            if self.verbose:
                print("Aucune mise à jour nécessaire.")

        return self.notlisted

    @staticmethod
    def from_ohlcv_to_close(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforme un DataFrame OHLCV en un DataFrame avec les prix de clôture.

        Args:
            ohlcv_df (pd.DataFrame): DataFrame contenant les colonnes 'DATE', 'ID', et 'CLOSE'.

        Returns:
            pd.DataFrame: DataFrame pivoté avec les prix de clôture.
        """
        ohlcv_df.columns = [col.upper() for col in ohlcv_df.columns]
        ohlcv_df = ohlcv_df[['DATE', 'ID', 'CLOSE']].copy()

        ohlcv_df['DATE'] = pd.to_datetime(ohlcv_df['DATE'])

        # Éliminer les doublons pour chaque combinaison de 'DATE' et 'ID'
        ohlcv_df = ohlcv_df.sort_values('DATE').drop_duplicates(subset=['DATE', 'ID'], keep='last')

        return ohlcv_df.pivot(index='DATE', columns='ID', values='CLOSE')