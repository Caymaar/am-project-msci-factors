import numpy as np
import pandas as pd
from backtester272.Strategy import Strategy, RankedStrategy, OptimizationStrategy, filter_with_signals

class EqualWeightStrategy(Strategy):
    """
    Stratégie qui attribue un poids égal à chaque actif.
    """
    @filter_with_signals
    def get_position(self, historical_data: pd.DataFrame, current_position: pd.Series, benchmark_position: pd.Series = None) -> pd.Series:
        """
        Retourne une position avec des poids égaux pour chaque actif.

        Args:
            historical_data (pd.DataFrame): Les données historiques.
            current_position (pd.Series): La position actuelle.

        Returns:
            pd.Series: Nouvelle position avec des poids égaux.
        """
        num_assets = historical_data.shape[1]

        if num_assets == 0:
            return pd.Series()

        weights = pd.Series(1 / num_assets, index=historical_data.columns)
        return weights
    
class RandomStrategy(Strategy):
    """
    Stratégie qui attribue des poids aléatoires normalisés aux actifs.
    """
    @filter_with_signals
    def get_position(self, historical_data: pd.DataFrame, current_position: pd.Series, benchmark_position: pd.Series = None) -> pd.Series:
        """
        Retourne une position avec des poids aléatoires normalisés.

        Args:
            historical_data (pd.DataFrame): Les données historiques.
            current_position (pd.Series): La position actuelle.

        Returns:
            pd.Series: Nouvelle position avec des poids aléatoires.
        """
        weights = np.random.rand(len(historical_data.columns))
        weights /= weights.sum()  # Normaliser les poids pour qu'ils totalisent 1
        return pd.Series(weights, index=historical_data.columns)

class MinVarianceStrategy(OptimizationStrategy):
    """
    Stratégie d'optimisation minimisant la variance du portefeuille.
    """
    def objective_function(self, weights: np.ndarray, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> float:
        """
        Fonction objectif pour minimiser la variance du portefeuille.

        Args:
            weights (np.ndarray): Poids du portefeuille.
            expected_returns (pd.Series): Rendements attendus des actifs.
            cov_matrix (pd.DataFrame): Matrice de covariance.

        Returns:
            float: Variance du portefeuille.
        """
        #portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_variance = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)) * 252)

        # Terme de régularisation Ridge (si max_turnover est défini)
        if self.max_turnover is not None and hasattr(self, "current_position") and np.sum(np.abs(self.current_position)) > 0:
            ridge_penalty = self.lmd_ridge * np.sum((weights - self.current_position) ** 2)  # L2 penalty
        else:
            ridge_penalty = 0

        return portfolio_variance + ridge_penalty
    
class MaxSharpeStrategy(OptimizationStrategy):
    """
    Stratégie d'optimisation maximisant le ratio de Sharpe.
    """
    def objective_function(self, weights: np.ndarray, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> float:
        """
        Fonction objectif pour maximiser le ratio de Sharpe du portefeuille.

        Args:
            weights (np.ndarray): Poids du portefeuille.
            expected_returns (pd.Series): Rendements attendus des actifs.
            cov_matrix (pd.DataFrame): Matrice de covariance.

        Returns:
            float: Négatif du ratio de Sharpe (pour minimisation).
        """
        portfolio_return = np.dot(weights, expected_returns) * 252  # Rendement annualisé
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)) * 252)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility

        # Terme de régularisation Ridge (si max_turnover est défini)
        if self.max_turnover is not None and hasattr(self, "current_position") and np.sum(np.abs(self.current_position)) > 0:
            ridge_penalty = self.lmd_ridge * np.sum((weights - self.current_position) ** 2)  # L2 penalty
        else:
            ridge_penalty = 0

        # Maximiser Sharpe => Minimiser son opposé + pénalité
        return -sharpe_ratio + ridge_penalty

class EqualRiskContributionStrategy(OptimizationStrategy):
    """
    Stratégie Equal Risk Contribution (ERC), où chaque actif contribue également au risque total.
    """
    def __init__(self, lmd_mu: float = 0.0, lmd_var: float = 0.0, **kwargs) -> None:
        """
        Initialise la stratégie ERC avec des paramètres pour pondérer rendement et variance.

        Args:
            lmd_mu (float): Pondération pour maximiser le rendement.
            lmd_var (float): Pondération pour minimiser la variance.
        """
        super().__init__(**kwargs)
        self.lmd_mu = lmd_mu
        self.lmd_var = lmd_var

    def objective_function(self, weights: np.ndarray, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> float:
        """
        Fonction objectif pour équilibrer la contribution au risque.

        Args:
            weights (np.ndarray): Poids du portefeuille.
            expected_returns (pd.Series): Rendements attendus.
            cov_matrix (pd.DataFrame): Matrice de covariance.

        Returns:
            float: Valeur de la fonction objectif ERC.
        """

        cov_matrix = np.array(cov_matrix)

        risk_contributions = ((cov_matrix @ weights) * weights) / np.sqrt((weights.T @ cov_matrix @ weights))
        risk_objective = np.sum((risk_contributions[:, None] - risk_contributions[None, :])**2)

        return_value_objective = -self.lmd_mu * weights.T @ expected_returns
        variance_objective = self.lmd_var * weights.T @ cov_matrix @ weights

        # Terme de régularisation Ridge (si max_turnover est défini)
        if self.max_turnover is not None and hasattr(self, "current_position") and np.sum(np.abs(self.current_position)) > 0:
            ridge_penalty = self.lmd_ridge * np.sum((weights - self.current_position) ** 2)  # L2 penalty
        else:
            ridge_penalty = 0

        return risk_objective + return_value_objective + variance_objective + ridge_penalty
    
class ValueStrategy(RankedStrategy):
    """
    Stratégie basée sur la valeur relative des actifs (ratio prix actuel / prix passé).
    """
    def rank_assets(self, historical_data: pd.DataFrame) -> pd.Series:
        """
        Classe les actifs par leur ratio de valeur relative.

        Args:
            historical_data (pd.DataFrame): Les données historiques.

        Returns:
            pd.Series: Classement des actifs (meilleure valeur = rang élevé).
        """
        last_prices = historical_data.iloc[-1]  # Dernier prix
        prices_one_year_ago = historical_data.iloc[0]  # Prix il y a un an
        coef_asset = last_prices / prices_one_year_ago
        return coef_asset.rank(ascending=False, method='first')

class MomentumStrategy(RankedStrategy):
    """
    Stratégie Momentum basée sur les performances passées des actifs.
    """
    def rank_assets(self, historical_data: pd.DataFrame) -> pd.Series:
        """
        Classe les actifs par leur performance passée.

        Args:
            historical_data (pd.DataFrame): Les données historiques.

        Returns:
            pd.Series: Classement des actifs (meilleures performances = rang élevé).
        """
        returns = historical_data.pct_change().dropna()
        len_window = len(returns)
        delta = int(np.ceil(len_window * (1 / 12)))
        total_returns = returns.rolling(window=len_window - delta).apply(lambda x: (1 + x).prod() - 1)
        latest_returns = total_returns.iloc[-delta]
        latest_returns = latest_returns.dropna()
        return latest_returns.rank(ascending=True, method='first')

class MinVolStrategy(RankedStrategy):
    def rank_assets(self, historical_data: pd.DataFrame) -> pd.Series:
        """
        Classe les actifs en fonction de leur volatilité, où les actifs moins volatils sont favorisés.

        Args:
            historical_data (pd.DataFrame): Les données historiques.

        Returns:
            pd.Series: Classement des actifs en fonction de la volatilité.
        """
        returns = historical_data.pct_change().dropna()
        volatility = returns.std()
        volatility.dropna()
        return volatility.rank(ascending=False, method='first').sort_values()

class CrossingMovingAverage(EqualWeightStrategy):
    """
    Stratégie de croisement de moyennes mobiles basée sur deux périodes.
    """
    def __init__(self, fast_period: int = 30, slow_period: int = 90) -> None:
        """
        Initialise la stratégie avec des périodes de moyennes mobiles.

        Args:
            fast_period (int): Période de la moyenne mobile rapide.
            slow_period (int): Période de la moyenne mobile lente.
        """
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period

    def signals(self, data: pd.DataFrame) -> list:
        """
        Identifie les actifs ayant un croisement de moyennes mobiles.

        Args:
            data (pd.DataFrame): Données historiques.

        Returns:
            list: Liste des actifs avec croisement de moyennes mobiles.
        """
        fast_ma = data.rolling(window=self.fast_period).mean()
        slow_ma = data.rolling(window=self.slow_period).mean()
        crossover = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        last_day_crossover = crossover.iloc[-1]
        return last_day_crossover[last_day_crossover].index.tolist()   
        
    
    