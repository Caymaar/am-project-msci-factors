import numpy as np
import pandas as pd
from backtester272.Strategy import Strategy, RankedStrategy, OptimizationStrategy, filter_with_signals
from tqdm import tqdm

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

class LongOnlyMomentumStrategy(EqualWeightStrategy):
    """
    Stratégie long-only basée sur le momentum à 1 an.
    
    Cette classe calcule le momentum sur une période donnée (par défaut 252 jours) 
    et sélectionne, selon un quantile défini, les actifs ayant le meilleur momentum.
    Les actifs sélectionnés se voient attribuer des poids égaux.
    
    Args:
        quantile (float): Le quantile pour la sélection des actifs.
                        Exemples :
                        - 0.1 pour une construction en décile (top 10%),
                        - 0.2 pour une construction en quintile (top 20%),
                        - 0.25 pour une construction en quartile (top 25%).
    """
    def __init__(self, quantile: float):
        self.quantile = quantile

    def signals(self, historical_data: pd.DataFrame) -> list:
        """
        Calcule la position (poids) en sélectionnant les actifs avec le meilleur momentum 
        selon le quantile défini. Seuls les actifs sélectionnés reçoivent un poids égal.
        
        Args:
            historical_data (pd.DataFrame): Données historiques des prix.

        Returns:
            pd.Series: Positions (poids) normalisés pour le portefeuille long-only.
        """
        momentum = historical_data.pct_change(periods=len(historical_data)-1).iloc[-1]
        # Tri décroissant : les actifs avec le meilleur momentum en tête
        sorted_momentum = momentum.sort_values(ascending=False)
        n_assets = len(sorted_momentum)
        # Sélection des actifs selon le quantile défini
        n_selected = int(np.ceil(n_assets * self.quantile))
        if n_selected == 0:
            n_selected = 1  # au moins un actif
        selected_assets = sorted_momentum.iloc[:n_selected]
        return selected_assets.index.tolist()


class LongOnlyIdiosyncraticMomentumStrategy(EqualWeightStrategy):
    
    def __init__(self, quantile: float, benchmark: pd.Series):
        """
        :param quantile: pour sélectionner un quantile d'actifs (ex. 0.2 pour les 20% les plus attractifs)
        :param benchmark: série de prix du benchmark (indexée par date)
        :param risk_free_rate: taux sans risque (valeur constante ici)
        """
        self.quantile = quantile
        # Conversion du benchmark en rendements journaliers
        self.benchmark_returns = benchmark.pct_change().dropna()

    def signals(self, historical_data: pd.DataFrame) -> list:
        """
        historical_data : DataFrame contenant, pour chaque date (index) et pour chaque actif (colonnes),
        les rendements journaliers. On suppose ici disposer d'au moins 36 mois de données.
        """
        # Filtrer pour conserver les dates communes aux données historiques et au benchmark
        common_dates = historical_data.index.intersection(self.benchmark_returns.index)
        historical_data = historical_data.loc[common_dates]
        benchmark_returns = self.benchmark_returns.loc[common_dates]
        
        # Définir la période de calcul du momentum idiosyncratique : de t-12 à t-2 mois
        try:
            latest_date = historical_data.index[-1]
        except IndexError:
            return []
        window_start = latest_date - pd.Timedelta(days=12*30)
        window_end = latest_date - pd.Timedelta(days=2*30)
        month_ends = pd.date_range(start=window_start, end=window_end, freq='30D')
        
        # Dictionnaire pour stocker le score de momentum idiosyncratique par actif
        momentum_scores = {}
        
        # Pour chaque actif, on va re-estimer mensuellement le modèle sur les 36 mois précédents
        for asset in tqdm(historical_data.columns, desc="Processing assets"):
            monthly_residuals = []
            for m in month_ends:
                # Pour chaque mois m, on choisit le dernier jour disponible <= m dans historical_data
                data_until_m = historical_data.loc[historical_data.index <= m]
                if data_until_m.empty:
                    continue
                # Date cible pour le mois : le dernier jour disponible du mois
                m_target = data_until_m.index[-1]
                
                # Fenêtre de régression : 36 mois précédant m_target (excluant éventuellement les jours trop anciens)
                reg_window_start = m_target - pd.DateOffset(months=36)
                reg_dates = historical_data.loc[(historical_data.index > reg_window_start) & (historical_data.index <= m_target)].index
                # Vérifier que l'on dispose d'un minimum de données pour estimer le modèle
                if len(reg_dates) < 10:
                    continue
                
                # Extraction des rendements pour l'actif et le benchmark sur la fenêtre de régression
                Y = historical_data.loc[reg_dates, asset]
                X = benchmark_returns.loc[reg_dates]

                # Calcul des moyennes
                mean_Y = Y.mean()
                mean_X = X.mean()

                # Calcul de beta : somme((X - mean_X)*(Y - mean_Y)) / somme((X - mean_X)^2)
                beta = ((X - mean_X) * (Y - mean_Y)).sum() / ((X - mean_X) ** 2).sum()

                # Calcul de alpha : moyenne(Y) - beta * moyenne(X)
                alpha = mean_Y - beta * mean_X
                
                # Calcul du résidu pour la date cible m_target
                r_asset = historical_data.loc[m_target, asset]
                r_bench = benchmark_returns.loc[m_target]
                resid = r_asset - (alpha + beta * r_bench)
                monthly_residuals.append(resid)
            
            # Calcul du score de momentum idiosyncratique si des résidus ont été obtenus
            if monthly_residuals:
                mean_resid = np.mean(monthly_residuals)
                std_resid = np.std(monthly_residuals)
                score = mean_resid / std_resid if std_resid != 0 else 0
                momentum_scores[asset] = score
            else:
                momentum_scores[asset] = np.nan
        
        # Tri décroissant des actifs selon leur score de momentum idiosyncratique
        sorted_scores = pd.Series(momentum_scores).sort_values(ascending=False)
        print(sorted_scores)

        # Si sorted_scores est rempli de nan ou de 0
        if sorted_scores.isnull().all() or sorted_scores.eq(0).all():
            return []
        
        # Sélectionner le quantile défini
        n_assets = len(sorted_scores)
        n_selected = int(np.ceil(n_assets * self.quantile))
        if n_selected == 0:
            n_selected = 1  # au moins un actif
        selected_assets = sorted_scores.iloc[:n_selected]
        
        return selected_assets.index.tolist()