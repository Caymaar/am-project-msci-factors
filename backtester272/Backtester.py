import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Union
from backtester272.Result import Result
from backtester272.Strategy import Strategy
from backtester272.Tactical import Tactical
from backtester272.MacroStrat import MacroTactical
 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Backtester:
    """
    Classe principale pour effectuer des backtests financiers.

    Cette classe permet d'exécuter des backtests sur des données de prix en utilisant différentes stratégies
    et configurations. Elle prend en charge les univers dynamiques (listes de tickers valides par date),
    la gestion des données manquantes, et le calcul de la performance du portefeuille.

    Attributes:
        data (pd.DataFrame): Données de prix pour les actifs.
        dates_universe (Dict[str, List[str]]): Univers des actifs par date, au format {date: [tickers]}.
        start_date (pd.Timestamp): Date de début du backtest.
        end_date (pd.Timestamp): Date de fin du backtest.
        freq (int): Fréquence de rééquilibrage en jours.
        window (int): Taille de la fenêtre de formation en jours.
        aum (float): Actifs sous gestion (AUM).
        transaction_cost (float): Coût de transaction en pourcentage.
        weights (pd.DataFrame): Poids des actifs calculés.
        performance (pd.Series): Performance du portefeuille sur la période du backtest.
        total_transaction_costs (float): Coût total des transactions pendant le backtest.
    """

    def __init__(self, data: pd.DataFrame, dates_universe: Dict[str, List[str]] = None, benchmark_weights: pd.DataFrame = None) -> None:
        """
        Initialise le backtester avec les données de prix et un univers d'actifs optionnel.

        Args:
            data (pd.DataFrame or pd.Series): Données de prix pour les actifs, indexées par date.
            dates_universe (Dict[str, List[str]], optional): Univers des actifs par date, au format
                {date: [tickers]}. Par défaut, aucun univers n'est défini.

        Raises:
            TypeError: Si les données ou `dates_universe` ne sont pas au bon format.
            ValueError: Si les données sont vides ou non quotidiennes.
        """
        # Gestion des données de prix
        if isinstance(data, pd.Series):
            self.data = pd.DataFrame(data)
            if self.data.columns[0] == 0:
                self.data.columns = ['Asset']
        else:
            self.data = data

        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Les données doivent être un DataFrame ou une Series.")

        if self.data.empty:
            raise ValueError("Les données ne peuvent pas être vides.")

        if (self.data.index.to_series().diff()[1:].dt.days < 1).all():
            raise ValueError("Les données doivent être quotidiennes.")
        
        if benchmark_weights is not None:
            # Vérifier si l'index est de type datetime
            if not isinstance(benchmark_weights.index, pd.DatetimeIndex):
                try:
                    benchmark_weights.index = pd.to_datetime(benchmark_weights.index)
                except ValueError:
                    raise ValueError("L'index des données doit être de type datetime.")

        self.benchmark_weights = benchmark_weights
        
        # Gestion des données manquantes, calcul des poids et performance
        self.data = self.handle_missing_data(self.data)

        # Gestion de l'univers d'actifs par date
        self.dates_universe = {}
        if dates_universe is not None:
            if not isinstance(dates_universe, dict):
                raise TypeError("dates_universe doit être un dictionnaire.")

            for date_str, tickers in dates_universe.items():
                # Validation des clés et valeurs
                try:
                    pd.to_datetime(date_str)
                except ValueError:
                    raise ValueError(f"La clé {date_str} n'est pas une date valide au format YYYY-MM-DD.")
                
                if not isinstance(tickers, list) or not all(isinstance(t, str) for t in tickers):
                    raise ValueError(f"Les tickers pour la date {date_str} doivent être une liste de chaînes.")
                
                # Vérifie la présence des tickers dans les colonnes des données
                invalid_tickers = [t for t in tickers if t not in self.data.columns]
                if invalid_tickers:
                    raise ValueError(f"Tickers non trouvés dans les données: {invalid_tickers}.")
            
            self.dates_universe = dates_universe

            

    def run(self, 
            start_date: Optional[pd.Timestamp] = None, 
            end_date: Optional[pd.Timestamp] = None, 
            freq: int = 30, 
            window: int = 365,
            freq_tactical: int = 30,
            window_tactical: int = 365, 
            freq_macro: int = 30,
            aum: float = 100, 
            transaction_cost: float = 0.0, 
            strategy: Strategy = None,
            macro: MacroTactical = None,
            tactical: Tactical = None) -> Result:
        """
        Exécute le backtest sur la période spécifiée avec les paramètres donnés.

        Args:
            start_date (pd.Timestamp, optional): Date de début du backtest. Par défaut, la première date des données.
            end_date (pd.Timestamp, optional): Date de fin du backtest. Par défaut, la dernière date des données.
            freq (int): Fréquence de rééquilibrage (en jours).
            window (int): Taille de la fenêtre de formation (en jours).
            aum (float): Actifs sous gestion (AUM).
            transaction_cost (float): Coût de transaction (en pourcentage).
            strategy (Strategy): Stratégie de trading à appliquer.

        Returns:
            Result: Objet contenant les résultats du backtest.

        Raises:
            ValueError: Si des paramètres obligatoires ou incohérents sont fournis.
        """
        if strategy is None:
            raise ValueError("Une stratégie doit être fournie pour exécuter le backtest.")

        if start_date is None:
            start_date = self.data.index[0]
        if end_date is None:
            end_date = self.data.index[-1]

        # Validation des paramètres
        if not isinstance(freq, int) or freq <= 0:
            raise ValueError("freq doit être un entier positif supérieur à 0.")
        if not isinstance(window, int) or window <= 0:
            raise ValueError("window doit être un entier positif supérieur à 0.")
        if not isinstance(aum, (int, float)) or aum <= 0:
            raise ValueError("aum doit être un float positif supérieur à 0.")
        if not isinstance(transaction_cost, (int, float)) or transaction_cost < 0:
            raise ValueError("transaction_cost doit être un float positif ou nul.")

        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.window = window
        self.freq_tactical = freq_tactical
        self.window_tactical = window_tactical
        self.freq_macro = freq_macro
        self.aum = aum
        self.transaction_cost = transaction_cost

        # Exécution de la strategie
        self.weights = self.calculate_weights(strategy)

        self.performance = self.calculate_performance(self.weights)
        total_transaction_costs = self.total_transaction_costs

        if macro is not None:
            self.macro_weights = self.calculate_weights(macro, relative_weights=self.weights)
            result_macro_weights = self.macro_weights.loc[self.start_date:self.end_date].loc[self.weights.index[1]:]
            self.macro_performance = self.calculate_performance(result_macro_weights)

            if tactical is not None:
                self.tactical_macro_weights = self.calculate_weights(tactical, relative_weights=self.macro_weights)
                result_tactical_macro_weights = self.tactical_macro_weights.loc[self.start_date:self.end_date].loc[self.weights.index[1]:]
                self.tactical_macro_performance = self.calculate_performance(result_tactical_macro_weights)
            else:
                self.tactical_macro_performance = None
                result_tactical_macro_weights = None
        else:
            self.macro_performance = None
            result_macro_weights = None
            self.tactical_macro_performance = None
            result_tactical_macro_weights = None

        # Exécution de la tactique si disponible
        if tactical is not None:
            self.tactical_weights = self.calculate_weights(tactical, relative_weights=self.weights)
            result_tactical_weights = self.tactical_weights.loc[self.start_date:self.end_date].loc[self.weights.index[1]:]
            self.tactical_performance = self.calculate_performance(result_tactical_weights)
        else:
            self.tactical_performance = None
            result_tactical_weights = None

        if self.benchmark_weights is not None:
            result_benchmark_weights = self.benchmark_weights.loc[self.start_date:self.end_date].loc[self.weights.index[1]:]
            self.benchmark = self.calculate_performance(result_benchmark_weights)

            if macro is not None:
                self.macro_benchmark_weights = self.calculate_weights(macro, relative_weights=self.benchmark_weights)
                result_macro_benchmark_weights = self.macro_benchmark_weights.loc[self.start_date:self.end_date].loc[self.weights.index[1]:]
                self.macro_benchmark = self.calculate_performance(result_macro_benchmark_weights)

                if tactical is not None:
                    self.tactical_macro_benchmark_weights = self.calculate_weights(tactical, relative_weights=self.macro_benchmark_weights)
                    result_tactical_macro_benchmark_weights = self.tactical_macro_benchmark_weights.loc[self.start_date:self.end_date].loc[self.weights.index[1]:]
                    self.tactical_macro_benchmark = self.calculate_performance(result_tactical_macro_benchmark_weights)
                else:
                    self.tactical_macro_benchmark = None
                    result_tactical_macro_benchmark_weights = None
            else:
                self.macro_benchmark = None
                result_macro_benchmark_weights = None
                self.tactical_macro_benchmark = None
                result_tactical_macro_benchmark_weights = None

            if tactical is not None:
                self.benchmark_tactical_weights = self.calculate_weights(tactical, relative_weights=self.benchmark_weights)
                result_tactical_benchmark_weights = self.benchmark_tactical_weights.loc[self.start_date:self.end_date].loc[self.weights.index[1]:]
                self.tactical_benchmark = self.calculate_performance(result_tactical_benchmark_weights)
            else:
                self.tactical_benchmark = None
                result_tactical_benchmark_weights = None
        else:
            self.benchmark = None
            self.tactical_benchmark = None
            self.macro_benchmark = None
            self.tactical_macro_benchmark = None
            result_benchmark_weights = None
            result_tactical_benchmark_weights = None
            result_macro_benchmark_weights = None
            result_tactical_macro_benchmark_weights = None
        
        self.performance, self.benchmark, self.tactical_performance, self.tactical_benchmark, self.macro_performance, self.macro_benchmark, self.tactical_macro_performance, self.tactical_macro_benchmark = self.rebase_performances(self.performance, self.benchmark, self.tactical_performance, self.tactical_benchmark, self.macro_performance, self.macro_benchmark, self.tactical_macro_performance, self.tactical_macro_benchmark)

        # Nommer la stratégie si elle n'est pas déjà nommée
        if not hasattr(strategy, 'name'):
            strategy.name = strategy.__class__.__name__

        return Result(self.performance, 
                      self.weights, 
                      total_transaction_costs, 
                      strategy.name, 
                      self.benchmark, 
                      result_benchmark_weights,
                      self.tactical_performance,
                      result_tactical_weights,
                      self.tactical_benchmark,
                      result_tactical_benchmark_weights,
                      self.macro_performance,
                      result_macro_weights,
                      self.macro_benchmark,
                      result_macro_benchmark_weights,
                      self.tactical_macro_performance,
                      result_tactical_macro_weights,
                      self.tactical_macro_benchmark,
                      result_tactical_macro_benchmark_weights)
    

    @staticmethod
    def handle_missing_data(data) -> None:
        """
        Gère les données manquantes en remplissant les valeurs dans les colonnes valides.
        """
        # Supprime les colonnes entièrement vides et conserve les données numériques
        data = data.dropna(axis=1, how='all').select_dtypes(include=[np.number])

        # Remplit les valeurs manquantes entre le premier et le dernier index valides
        for col in data.columns:
            data[col] = data[col].loc[
                data[col].first_valid_index():data[col].last_valid_index()
            ].ffill()

        if data.empty:
            raise ValueError("Aucune donnée disponible après le traitement des valeurs manquantes.")
        
        return data

    def calculate_weights(self, strategy: Union[Strategy, Tactical], relative_weights: pd.DataFrame = None) -> None:
        """
        Calcule les poids optimaux pour chaque date de rééquilibrage en fonction de la stratégie.

        Args
            strategy (Strategy): Stratégie de trading utilisée.
        """

        # Définir la fréquence de rééquilibrage et la fenêtre de formation en jours
        if isinstance(strategy, Strategy):
            freq_dt = pd.DateOffset(days=self.freq)
            window_dt = pd.DateOffset(days=self.window)
        elif isinstance(strategy, Tactical):
            freq_dt = pd.DateOffset(days=self.freq_tactical)
            window_dt = pd.DateOffset(days=self.window_tactical)
        elif isinstance(strategy, MacroTactical):
            freq_dt = pd.DateOffset(days=self.freq_macro)
            window_dt = pd.DateOffset(days=self.window)

        # Calculer la date de début en tenant compte de la fenêtre
        start_date_with_window = pd.to_datetime(self.start_date) - window_dt

        # Obtenir les données de prix sur la période pertinente
        prices = self.data[start_date_with_window:self.end_date]

        # Générer les dates de rééquilibrage en ordre décroissant
        rebalancing_dates = []
        current_date = prices.index[-1]
        while current_date >= prices.index[0] + window_dt:
            rebalancing_dates.append(current_date)
            current_date -= freq_dt

        # Si relative_weights est fourni, on récupère ses dates (pour la tactique)
        if relative_weights is not None:
            # Les dates tactiques sont celles présentes dans l'index de relative_weights
            strategic_dates = pd.DatetimeIndex(relative_weights.index)
            tactical_dates = pd.DatetimeIndex(rebalancing_dates)
            
            # Filtrer strategic_dates pour ne conserver que celles postérieures ou égales à la date minimum des tactical_dates
            min_tactical = tactical_dates.min()
            strategic_dates = strategic_dates[strategic_dates >= min_tactical]

            # On fait l'union des deux ensembles
            combined_dates = strategic_dates.union(tactical_dates)
            
            # Pour éviter les doublons ou rééquilibrages trop rapprochés,
            # on remplace une date tactique proche d'une date stratégique par la date stratégique.
            threshold = pd.Timedelta(days=3)  # seuil d'un jour par exemple
            final_dates = []
            for date in combined_dates:
                # Si une date stratégique est proche (différence inférieure au seuil)
                close_strategic = strategic_dates[abs(strategic_dates - date) <= threshold]
                if not close_strategic.empty:
                    final_dates.append(close_strategic.min())  # On prend la première date stratégique proche
                else:
                    final_dates.append(date)
            # On remet les dates en ordre chronologique
            rebalancing_dates = sorted(set(final_dates))
        else:

            # Inverser la liste pour avoir les dates en ordre croissant
            rebalancing_dates.reverse()

        # Initialiser les poids précédents à zéro pour tous les actifs
        last_weights = pd.Series(0.0, index=prices.columns)

        # Initialiser les listes pour collecter les poids et les dates
        weights_list = [last_weights]
        dates_list = [(current_date - pd.DateOffset(days=1))]

        # Calculer les poids pour chaque date de rééquilibrage
        for current_date in rebalancing_dates:
            # Définir la période de formation
            train_start = current_date - window_dt
            train_end = current_date - pd.DateOffset(days=1)

            # Obtenir les données de prix pour la période de formation
            price_window = prices[train_start:train_end]

            # Filtrer les données en fonction de l'univers défini
            if self.dates_universe:
                # Convertir les dates du dictionnaire en format datetime
                universe_dates = [pd.to_datetime(date) for date in self.dates_universe.keys()]

                # Trouver la date d'univers la plus récente avant la date courante
                available_dates = [date for date in universe_dates if date <= current_date]

                if available_dates:
                    reference_date = max(available_dates)
                    active_tickers = self.dates_universe[reference_date.strftime('%Y-%m-%d')]
                    price_window = price_window[active_tickers]
                else:
                    print(f"Pas d'univers défini avant {current_date}")
                    price_window = pd.DataFrame()  # Renvoie un DataFrame vide si aucune date valide

            # Supprimer les colonnes avec des valeurs manquantes
            price_window_filtered = price_window.dropna(axis=1)
            if price_window_filtered.empty:
                print(f"Aucune donnée disponible pour {current_date}. Passage...")
                continue

            # Calculer les nouveaux poids en fonction de la stratégie
            if isinstance(strategy, Strategy):
                if self.benchmark_weights is not None:
                    benchmark_weights = self.get_drifted_weights(current_date, self.benchmark_weights)
                    final_optimal_weights = strategy.get_position(price_window_filtered, last_weights, benchmark_weights)
                else:
                    final_optimal_weights = strategy.get_position(price_window_filtered, last_weights)
                last_weights = final_optimal_weights

            elif isinstance(strategy, Tactical):
                try:
                    actual_weights = self.get_drifted_weights(current_date, relative_weights)
                except:
                    actual_weights = pd.Series(0.0, index=prices.columns)
                final_optimal_weights = strategy.get_position(price_window_filtered, actual_weights)

            elif isinstance(strategy, MacroTactical):
                try:
                    actual_weights = self.get_drifted_weights(current_date, relative_weights)
                except:
                    actual_weights = pd.Series(0.0, index=prices.columns)
                final_optimal_weights = strategy.get_position(current_date, actual_weights)

            # Enregistrer les poids et la date
            weights_list.append(final_optimal_weights)
            dates_list.append(current_date)

        # Créer un DataFrame à partir des poids collectés
        optimal_weights_df = pd.DataFrame(weights_list, index=dates_list)

        # Assigner les poids calculés à l'attribut de la classe
        return optimal_weights_df.fillna(0.0)

    def calculate_performance(self, weights: pd.Series) -> pd.Series:
        """
        Calcule la performance du portefeuille en utilisant les poids calculés et en tenant compte du drift des poids en l'absence de rebalancement.
        """
        
        # Initialiser le solde du portefeuille avec les actifs sous gestion (AUM)
        balance = self.aum

        # Obtenir la première date où des poids sont disponibles
        first_valid_date = weights.first_valid_index()

        # Filtrer les données de prix dans la plage de dates spécifiée
        df = self.data[self.start_date:self.end_date]

        # Calculer les rendements quotidiens
        returns = df.pct_change()[1:]

        # Initialiser les coûts totaux de transaction et les poids précédents
        self.total_transaction_costs = 0
        previous_weights = pd.Series(0.0, index=weights.columns)

        # Initialiser les listes pour stocker les valeurs du portefeuille et les dates
        portfolio_values = [self.aum]
        dates = [first_valid_date - pd.DateOffset(days=1)]

        # Obtenir la liste des dates à traiter
        date_range = returns.loc[first_valid_date:].index

        for date in date_range:
            daily_returns = returns.loc[date]

            # Récupérer la date suivante dans date_range s'il y en a une
            current_idx = date_range.get_loc(date)
            next_date = date_range[current_idx + 1] if current_idx + 1 < len(date_range) else None

            # Chercher dans weights une date de rebalancement comprise entre la date courante (incluse)
            # et la prochaine date dans date_range (non incluse)
            rebalancing_dates = [
                wd for wd in weights.index 
                if wd >= date and (next_date is None or wd < next_date)
            ]
            if rebalancing_dates:
                # On prend la date la plus proche correspondant au rebalancement
                reb_date = min(rebalancing_dates)
                current_weights = weights.loc[reb_date]

                # Calculer les changements de positions
                changes = (current_weights - previous_weights) * balance

                # Calculer les coûts de transaction
                transaction_costs = changes.abs().sum() * (self.transaction_cost / 100)

                # Mettre à jour les coûts totaux et réduire le solde
                self.total_transaction_costs += transaction_costs
                balance -= transaction_costs
            else:
                # Sans rebalancement, on conserve les poids de la veille
                current_weights = previous_weights.copy()
                #print(self.get_drifted_weights(date, self.weights) - current_weights)
            # Calculer le rendement du portefeuille pour la journée
            portfolio_return = (current_weights * daily_returns).sum()

            # Mettre à jour le solde du portefeuille
            balance *= (1 + portfolio_return)

            # Actualiser les poids pour refléter le drift dû aux rendements journaliers
            # Chaque actif évolue selon son rendement, ce qui modifie la composition relative

            drifted_weights = (current_weights * (1 + daily_returns)) / (1 + portfolio_return)
   

            previous_weights = drifted_weights.copy()

            # Enregistrer la valeur du portefeuille et la date
            portfolio_values.append(balance)
            dates.append(date)

        # Retourner la performance sous forme de Series
        return pd.Series(portfolio_values, index=dates)

    def get_drifted_weights(self, date: pd.Timestamp, weights: pd.Series) -> pd.Series:
        """
        Retourne les poids driftés pour une date donnée.
        Si la date n'est pas directement présente, on utilise la dernière date de poids disponible 
        et on calcule le drift jour par jour jusqu'à la date demandée.
        """
        if weights is None:
            raise ValueError("Aucun poids de stratégie n'a été fourni.")

        if not isinstance(date, pd.Timestamp):
            date = pd.to_datetime(date)

        # Récupérer la dernière date de poids disponible inférieure ou égale à la date demandée
        if date in weights.index:
            base_date = date
            base_weights = weights.loc[date]
        else:
            valid_dates = weights.index[weights.index <= date]
            if valid_dates.empty:
                raise ValueError(f"Pas de poids de stratégie disponibles pour la date {date}.")
            base_date = valid_dates.max()
            base_weights = weights.loc[base_date]

        # Si la date demandée correspond à la date de base, retourner directement les poids
        if base_date == date:
            return base_weights

        # Extraire les données de prix entre la date de base et la date demandée
        returns = self.data.pct_change()[1:].loc[base_date:date]

        # Vérifier que le DataFrame n'est pas vide
        if returns.empty:
            return base_weights

        # Si la date de base n'est pas exactement présente, on ajuste la date de base
        if returns.index[0] != base_date:
            # Ici, on choisit de prendre le premier jour disponible dans les données de prix comme nouvelle base.
            base_date = returns.index[0]
            # Vous pouvez également logguer un avertissement pour signaler ce changement.

        # Initialiser les poids courants avec les poids de base
        current_weights = base_weights.copy()

        # Appliquer le drift jour par jour
        for day in returns.index[:-1]:
            r = returns.loc[day]
            portfolio_return = (current_weights * r).sum()
            current_weights = (current_weights * (1 + r)) / (1 + portfolio_return)
        
        return current_weights
            

    def rebase_performances(self, *performances: pd.Series) -> list:
        """
        Rebase plusieurs séries de performance de sorte qu'elles commencent toutes à 100
        à partir de la date commune, déterminée comme le plus tardif des "départs" effectifs.
        
        Pour chaque série non None, le "départ" est défini comme la date juste AVANT le premier mouvement 
        (c'est-à-dire, le premier rendement différent de zéro). Si la série reste constante, on prend la première date.
        Les entrées qui sont None sont renvoyées telles quelles.
        
        Args:
            *performances (pd.Series): Plusieurs séries de performance avec des index temporels,
                                        qui peuvent débuter à des dates différentes. Certaines peuvent être None.
        
        Returns:
            list: Liste de pd.Series rebased ou None pour celles qui étaient None.
        """
        
        def find_start_date(perf: pd.Series) -> pd.Timestamp:
            """
            Pour une série de performance, identifie la date juste AVANT le premier mouvement.
            Si la série ne présente aucun mouvement, retourne la première date.
            """
            if len(perf) < 2:
                return perf.index[0]
            diff = perf.diff()
            # On ignore le NaN initial et on cherche le premier rendement non nul
            non_zero = diff.dropna()
            movement_dates = non_zero[non_zero != 0].index
            if not movement_dates.empty:
                first_movement = movement_dates[0]
                loc = perf.index.get_loc(first_movement)
                # S'il existe une valeur précédente, on la prend
                if loc > 0:
                    return perf.index[loc - 1]
                else:
                    return first_movement
            else:
                return perf.index[0]
        
        # Filtrer les séries non None
        valid_perfs = [perf for perf in performances if perf is not None]
        
        if not valid_perfs:
            # Si aucune série valide, renvoyer la liste d'inputs (qui seront tous None)
            return [None for _ in performances]
        
        # Pour chaque série non None, déterminer sa date de départ effective
        start_dates = [find_start_date(perf) for perf in valid_perfs]
        # La date commune sera la plus tardive parmi ces dates
        common_start = max(start_dates)
        
        rebased_series = []
        for perf in performances:
            if perf is None:
                rebased_series.append(None)
                continue
            
            # Sélectionner les données à partir de common_start
            perf_common = perf[perf.index >= common_start]
            if perf_common.empty:
                # S'il n'y a pas de données après common_start, renvoyer une série vide
                rebased_series.append(pd.Series(dtype=perf.dtype))
            else:
                # Utiliser la première valeur après common_start pour rebaser la série à 100
                base_value = perf_common.iloc[0]
                rebased = (perf_common / base_value) * 100
                rebased_series.append(rebased)
        
        return rebased_series