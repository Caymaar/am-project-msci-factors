import numpy as np
import pandas as pd
from backtester272.Tactical import AlphaBasedTactical, filter_with_signals
from scipy.stats import zscore

class RankMomentumTactical(AlphaBasedTactical):
    """
    Classe qui implémente la partie tactique Black-Litterman basée sur des signaux momentum.
    
    Les vues sont calculées à partir d'un signal momentum dérivé de l'historique des prix.
    """
    def __init__(self, delta: int = 0, nb_fractile : int =4, **kwargs) -> None:
        """
        Initialise la tactique Momentum avec un paramètre pour définir le delta.

        Args:
            delta (int): Nombre de jours soustrait du jour T. On calcule le rendement entre T - (len(historical_data) - 1) et T - delta.
        """
        super().__init__(**kwargs)
        self.nb_fractile = nb_fractile
        self.delta = delta
        
    @filter_with_signals
    def get_views(self, historical_data: pd.DataFrame, current_position: pd.Series) -> pd.Series:
        """
        Calcule les vues en se basant sur le signal momentum et le fractionne en nb_fractile quantiles.
        On attribue un signal long (1) aux actifs du quantile supérieur et un signal short (-1)
        aux actifs du quantile inférieur.
        """
        # Prix de départ : première observation de la série
        start_prices = historical_data.iloc[0]
        # Prix à T - delta : la (delta+1)-ème dernière observation
        end_prices = historical_data.iloc[-(self.delta + 1)]
        # Calcul du rendement momentum
        momentum = (end_prices / start_prices) - 1

        # Fractionner le signal momentum en nb_fractile quantiles
        # Les labels vont de 0 (plus bas) à nb_fractile-1 (plus haut)
        quantiles = pd.qcut(momentum, q=self.nb_fractile, labels=False, duplicates='drop')

        # Initialiser un vecteur de vues à 0 pour tous les actifs
        views = pd.Series(0, index=momentum.index)
        # Attribuer 1 aux actifs du quantile supérieur (meilleur momentum)
        # views[quantiles == (self.nb_fractile - 1)] = 0
        # Attribuer -1 aux actifs du quantile inférieur (pire momentum)
        views[quantiles == 0] = -1

        
        normalized_views = momentum / momentum.abs().sum()
        #views = np.where(normalized_views > 0, 0, views)
        normalized_views = normalized_views * views

        return normalized_views

    # @filter_with_signals
    # def get_views(self, historical_data: pd.DataFrame, current_position: pd.Series) -> pd.Series:
    #     """
    #     Calcule les vues en se basant sur le momentum des actifs.

    #     Pour chaque actif, le rendement est calculé entre le premier prix de la série
    #     et le prix à T - delta (c'est-à-dire la (delta+1)-ème dernière observation). 
    #     Ce rendement est ensuite normalisé pour obtenir un vecteur de poids.

    #     Args:
    #         historical_data (pd.DataFrame): DataFrame contenant les prix historiques.
    #                                          Les colonnes représentent les actifs.
    #         current_position (pd.Series): Le vecteur de poids actuel (non utilisé ici, mais disponible si besoin).

    #     Returns:
    #         pd.Series: Vecteur de poids basé sur le signal momentum.
    #     """
    #     # Prix de départ : première observation de la série
    #     start_prices = historical_data.iloc[0]
    #     # Prix à T - delta : la (delta+1)-ème dernière observation
    #     end_prices = historical_data.iloc[-(self.delta + 1)]
    #     # Calcul du rendement momentum
    #     momentum = (end_prices / start_prices) - 1
    #     momentum_zscore = zscore(momentum, nan_policy='omit').fillna(0)
    #     # momentum_zscore[f'Fractile_Momentum'] = pd.qcut(momentum_zscore, q=self.nb_fractile, labels=range(1, self.nb_fractile + 1))

    #     quartile_momentum: pd.Series = pd.qcut(momentum_zscore, q=self.nb_fractile, labels=range(1, self.nb_fractile + 1))
    #     quartile_momentum = quartile_momentum.astype(int)
    #     num_long = (quartile_momentum == self.nb_fractile).sum()
    #     num_short = (quartile_momentum == 1).sum()

    #     normalized_views = quartile_momentum.apply(
    #         lambda x: 1 / num_long if x == self.nb_fractile else -1 / num_short if x == 1 else 0
    #     )
    #     print(normalized_views)
    #     return normalized_views

        
class MomentumTactical(AlphaBasedTactical):
    """
    Classe qui implémente la partie tactique Black-Litterman basée sur des signaux momentum.

    Les vues sont calculées à partir d'un signal momentum dérivé de l'historique des prix.
    """
    def __init__(self, delta: int = 0, **kwargs) -> None:
        """
        Initialise la tactique Momentum avec un paramètre pour définir le delta.

        Args:
            delta (int): Nombre de jours soustrait du jour T. On calcule le rendement entre T - (len(historical_data) - 1) et T - delta.
        """
        super().__init__(**kwargs)
        self.delta = delta

    @filter_with_signals
    def get_views(self, historical_data: pd.DataFrame, current_position: pd.Series) -> pd.Series:
        """
        Calcule les vues en se basant sur le momentum des actifs.

        Pour chaque actif, le rendement est calculé entre le premier prix de la série
        et le prix à T - delta (c'est-à-dire la (delta+1)-ème dernière observation). 
        Ce rendement est ensuite normalisé pour obtenir un vecteur de poids.

        Args:
            historical_data (pd.DataFrame): DataFrame contenant les prix historiques.
                                             Les colonnes représentent les actifs.
            current_position (pd.Series): Le vecteur de poids actuel (non utilisé ici, mais disponible si besoin).

        Returns:
            pd.Series: Vecteur de poids basé sur le signal momentum.
        """
        # Prix de départ : première observation de la série
        start_prices = historical_data.iloc[0]
        # Prix à T - delta : la (delta+1)-ème dernière observation
        end_prices = historical_data.iloc[-(self.delta + 1)]
        # Calcul du rendement momentum
        momentum = (end_prices / start_prices) - 1
        # Normalisation pour obtenir un vecteur de poids
        normalized_views = momentum / momentum.abs().sum()
        return normalized_views
    
class ValueTactical(AlphaBasedTactical):
    """
    Classe qui implémente la partie tactique Black-Litterman basée sur un signal de value.
    
    Le signal est calculé comme le rapport de l'ancien prix sur le nouveau prix.
    """

    @filter_with_signals
    def get_views(self, historical_data: pd.DataFrame, current_position: pd.Series) -> pd.Series:
        """
        Calcule les vues en se basant sur un signal value.
        
        Pour chaque actif, le signal est calculé comme le rapport de l'ancien prix (première observation)
        sur le nouveau prix (la (delta+1)-ème dernière observation). 
        Ce signal est ensuite normalisé pour obtenir un vecteur de poids.
        
        Args:
            historical_data (pd.DataFrame): DataFrame contenant les prix historiques.
                                             Les colonnes représentent les actifs.
            current_position (pd.Series): Le vecteur de poids actuel (non utilisé ici, mais disponible si besoin).
            
        Returns:
            pd.Series: Vecteur de poids basé sur le signal value.
        """
        # Prix de départ : première observation de la série
        start_prices = historical_data.iloc[0]
        # Prix à T - delta : la (delta+1)-ème dernière observation
        end_prices = historical_data.iloc[-1]
        # Calcul du signal value : ancien prix / nouveau prix
        value_signal = start_prices / end_prices
        # Normalisation pour obtenir un vecteur de poids
        normalized_views = value_signal / value_signal.abs().sum()
        return normalized_views