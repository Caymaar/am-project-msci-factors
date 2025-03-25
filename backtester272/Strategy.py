from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import List, Dict
from functools import wraps

def filter_with_signals(func):
    """
    Décorateur pour filtrer les colonnes de données historiques en fonction
    des signaux définis par la stratégie (méthode `signals`).

    Args:
        func: Méthode de la classe à décorer.

    Returns:
        Callable: Méthode décorée avec filtrage des colonnes.
    """
    @wraps(func)
    def wrapper(self, historical_data: pd.DataFrame, current_position: pd.Series, *args, **kwargs):
        if hasattr(self, "signals") and callable(getattr(self, "signals")):
            columns_to_keep = self.signals(historical_data)
            historical_data = historical_data[columns_to_keep]
        return func(self, historical_data, current_position, *args, **kwargs)
    return wrapper

class Strategy(ABC):
    """
    Classe abstraite pour définir une stratégie d'investissement.

    Chaque stratégie doit implémenter la méthode `get_position`, qui détermine
    les poids des actifs dans le portefeuille à partir des données historiques.
    """
    def __init__(self) -> None:
        """
        Initialise la stratégie avec le nom de la classe fille.
        """
        self.name: str = self.__class__.__name__

    @abstractmethod
    def get_position(self, historical_data: pd.DataFrame, current_position: pd.Series) -> pd.Series:
        """
        Méthode obligatoire pour déterminer la position actuelle.

        Args:
            historical_data (pd.DataFrame): Les données historiques.
            current_position (pd.Series): La position actuelle.

        Returns:
            pd.Series: La nouvelle position (poids) pour chaque actif.
        """
        pass

class OptimizationStrategy(Strategy):
    """
    Stratégie d'optimisation de portefeuille basée sur la minimisation
    d'une fonction objectif, avec des contraintes sur les poids des actifs.
    """
    def __init__(self, max_weight: float = 1.0, min_weight: float = 0.0,
                 risk_free_rate: float = 0.02, total_exposure: float = 1.0, 
                 max_turnover: float = None, max_tracking_error: float = None, 
                 lmd_ridge: float = 0.0) -> None:
        """
        Initialise la stratégie d'optimisation avec des paramètres spécifiques.

        Args:
            max_weight (float): Poids maximum par actif.
            min_weight (float): Poids minimum par actif.
            risk_free_rate (float): Taux sans risque utilisé pour le calcul.
            total_exposure (float): Exposition totale du portefeuille (somme des poids).
        """
        super().__init__()
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.risk_free_rate = risk_free_rate
        self.total_exposure = total_exposure
        self.max_turnover = max_turnover
        self.max_tracking_error = max_tracking_error
        self.lmd_ridge = lmd_ridge

    @filter_with_signals
    def get_position(self, historical_data: pd.DataFrame, current_position: pd.Series, benchmark_position: pd.Series = None) -> pd.Series:
        """
        Détermine la nouvelle position (poids) en fonction des données historiques.

        Args:
            historical_data (pd.DataFrame): Les données historiques.
            current_position (pd.Series): La position actuelle.

        Returns:
            pd.Series: La nouvelle position calculée par optimisation.
        """
        # Calcul des rendements quotidiens
        returns = historical_data.pct_change().dropna()

        # Vérifie si suffisamment de données sont disponibles
        if len(returns) < 2:
            return current_position

        # Exclut les colonnes avec des valeurs manquantes
        returns = returns.dropna(axis=1, how='any')

        if returns.empty:
            return current_position

        # Matrice de covariance des rendements
        cov_matrix = returns.cov()

        # Rendements moyens attendus
        expected_returns = returns.mean()

        # Crée les contraintes du portefeuille
        portfolio_constraints = self.create_portfolio_constraints(current_position, cov_matrix, benchmark_position)

        # On introduit les poids du portefeuille dans l'objet pour les utiliser dans la fonction objectif
        self.current_position = current_position

        # Définir les bornes pour les poids (entre 0 et 1 par actif)
        bounds = tuple((0, 1) for _ in range(returns.shape[1]))

        # Poids initiaux égaux pour tous les actifs
        initial_weights = np.array([1 / returns.shape[1]] * returns.shape[1])

        # Résolution de l'optimisation
        result = minimize(
            fun=self.objective_function,
            x0=initial_weights,
            args=(expected_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=portfolio_constraints
        )

        if result.success:
            # Met à jour les poids avec les résultats de l'optimisation
            weights = pd.Series(0.0, index=historical_data.columns)
            weights.update(pd.Series(result.x, index=returns.columns))
            return weights
        else:
            # Avertissement en cas d'échec
            import warnings
            warnings.warn(f"L'optimisation n'a pas réussi : {result.message}. Utilisation des poids précédents.")
            return current_position

    def create_portfolio_constraints(self, current_position, cov_matrix, benchmark_position) -> List[Dict[str, any]]:
        """
        Crée les contraintes pour l'optimisation du portefeuille.

        Returns:
            List[Dict[str, any]]: Liste des contraintes d'optimisation.
        """
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - self.total_exposure},  # Somme des poids = exposition totale
            {'type': 'ineq', 'fun': lambda x: self.max_weight - x},            # Poids <= max_weight
            {'type': 'ineq', 'fun': lambda x: x - self.min_weight}             # Poids >= min_weight
            ]
        
        if self.max_turnover is not None and np.sum(np.abs(current_position)) > 0:
            constraints.append({'type': 'ineq', 'fun': lambda x: self.max_turnover - np.sum(np.abs(x - current_position))})
        
        if benchmark_position is not None and self.max_tracking_error is not None:
            constraints.append({'type': 'ineq', 'fun': lambda x: self.max_tracking_error - np.sqrt((x - benchmark_position).T @ cov_matrix @ (x - benchmark_position))})  # Tracking Error <= max_tracking_error

        return constraints

    @abstractmethod
    def objective_function(self, weights: np.ndarray, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> float:
        """
        Fonction objectif à minimiser (définie dans les sous-classes).

        Args:
            weights (np.ndarray): Poids des actifs.
            expected_returns (pd.Series): Rendements moyens attendus.
            cov_matrix (pd.DataFrame): Matrice de covariance.

        Returns:
            float: Valeur de la fonction objectif.
        """
        pass

class RankedStrategy(Strategy):
    """
    Stratégie basée sur le classement des actifs en fonction de leurs caractéristiques
    ou performances. Les actifs avec les meilleurs rangs reçoivent les poids les plus élevés.
    """
    @filter_with_signals
    def get_position(self, historical_data: pd.DataFrame, current_position: pd.Series, benchmark_position: pd.Series = None) -> pd.Series:
        """
        Calcule la position (poids) des actifs en fonction de leur classement.

        Args:
            historical_data (pd.DataFrame): Les données historiques.
            current_position (pd.Series): La position actuelle.

        Returns:
            pd.Series: Poids normalisés basés sur le classement.
        """
        ranked_assets = self.rank_assets(historical_data)

        num_assets = ranked_assets.count()
        sum_of_ranks = ranked_assets.sum()
        average = sum_of_ranks / num_assets
        weights = (ranked_assets - average)

        total_abs_ranks = sum(abs(weights))
        normalized_weights = weights / total_abs_ranks

        return normalized_weights

    @abstractmethod
    def rank_assets(self, historical_data: pd.DataFrame) -> pd.Series:
        """
        Méthode abstraite pour classer les actifs.

        Args:
            historical_data (pd.DataFrame): Les données historiques.

        Returns:
            pd.Series: Classement des actifs (du meilleur au pire).
        """
        pass