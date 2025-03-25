from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import List, Dict
from backtester272.Strategy import filter_with_signals


class Tactical(ABC):
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

class AlphaBasedTactical(Tactical):
    """
    Classe qui implémente une approche Black-Litterman en se basant sur le vecteur
    de poids 'current_position' comme poids d'équilibre, et en y intégrant des vues.
    
    Le paramètre alpha définit l'intensité avec laquelle les vues influencent la position finale.
    """
    def __init__(self, alpha: float = 0.05) -> None:
        """
        Args:
            alpha (float): Poids relatif des vues dans l'ajustement de la position.
                           0 donne la position d'équilibre, 1 la position entièrement basée sur les vues.
        """
        super().__init__()
        self.alpha = alpha
    
    def get_position(self, historical_data: pd.DataFrame, current_position: pd.Series) -> pd.Series:
        """
        Calcule la nouvelle position en combinant la position actuelle et les vues issues de get_views.
        
        Args:
            historical_data (pd.DataFrame): Les données historiques.
            current_position (pd.Series): Le vecteur de poids actuel (position d'équilibre).
            
        Returns:
            pd.Series: La nouvelle position ajustée.
        """

        if current_position.abs().sum() == 0:
            return current_position
        
        # Récupération des vues spécifiques à la stratégie
        view_position = self.get_views(historical_data, current_position)

        # Combinaison linéaire entre la position d'équilibre et la position basée sur les vues
        new_position = (1 - self.alpha) * current_position + self.alpha * view_position
        # Normalisation pour que la somme des poids soit égale à 1
        new_position = new_position / new_position.abs().sum()

        return new_position

    @abstractmethod
    def get_views(self, historical_data: pd.DataFrame, current_position: pd.Series) -> pd.Series:
        """
        Méthode abstraite pour obtenir les vues de la stratégie.
        
        Ces vues doivent être retournées sous forme d'un vecteur de poids normalisé.
        
        Args:
            historical_data (pd.DataFrame): Les données historiques.
            current_position (pd.Series): Le vecteur de poids actuel.
            
        Returns:
            pd.Series: Les vues sous forme de vecteur de poids.
        """
        pass
