# esios_precios_transform.py
from typing import Dict, List, Optional, Union
import pandas as pd
from datetime import datetime
import sys
import os

class ESIOSPreciosTransformer:
    """
    Transformer class for ESIOS price data.
    Handles data cleaning, validation, and transformation operations.
    """

    @staticmethod
    def validate_price_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate price data for common issues.
        
        Args:
            df (pd.DataFrame): Input DataFrame with price data
            
        Returns:
            pd.DataFrame: Validated DataFrame
            
        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        required_cols = ['fecha', 'hora', 'precio', 'id_mercado']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Expected: {required_cols}")
        
        # Ensure fecha is datetime
        df['fecha'] = pd.to_datetime(df['fecha'])
        
        # Remove any rows with null prices
        df = df.dropna(subset=['precio'])
        
        return df

    @staticmethod
    def standardize_prices(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize price values (e.g., convert to same unit, handle outliers).
        
        Args:
            df (pd.DataFrame): Input DataFrame with price data
            
        Returns:
            pd.DataFrame: DataFrame with standardized prices
        """
        # Remove extreme outliers (e.g., prices > 3 std from mean)
        mean_price = df['precio'].mean()
        std_price = df['precio'].std()
        df = df[abs(df['precio'] - mean_price) <= 3 * std_price]
        
        return df

    @staticmethod
    def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add useful time-based features to the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame with price data
            
        Returns:
            pd.DataFrame: DataFrame with additional time features
        """
        # Add day of week
        df['dia_semana'] = df['fecha'].dt.dayofweek
        
        # Add month
        df['mes'] = df['fecha'].dt.month
        
        # Add year
        df['año'] = df['fecha'].dt.year
        
        # Add is_weekend flag
        df['es_finde'] = df['dia_semana'].isin([5, 6]).astype(int)
        
        return df

    @staticmethod
    def aggregate_hourly_prices(df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate 15-minute prices to hourly if needed.
        
        Args:
            df (pd.DataFrame): Input DataFrame with price data
            
        Returns:
            pd.DataFrame: DataFrame with hourly prices
        """
        # Check if data is already hourly
        if not any(':' in str(h) for h in df['hora']):
            return df
            
        # Convert HH:MM format to hour
        df['hour'] = df['hora'].apply(lambda x: int(x.split(':')[0]))
        
        # Aggregate by hour
        agg_df = df.groupby(['fecha', 'hour', 'id_mercado'])['precio'].mean().reset_index()
        
        # Format hour back to string
        agg_df['hora'] = agg_df['hour'].apply(lambda x: f"{x:02d}")
        
        return agg_df.drop('hour', axis=1)

    def transform_market_data(self, 
                            df: pd.DataFrame, 
                            aggregate_to_hourly: bool = False) -> pd.DataFrame:
        """
        Apply all transformation steps to market data.
        
        Args:
            df (pd.DataFrame): Input DataFrame with price data
            aggregate_to_hourly (bool): Whether to aggregate 15-min data to hourly
            
        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        # Validate data
        df = self.validate_price_data(df)
        
        # Standardize prices
        df = self.standardize_prices(df)
        
        # Add time features
        df = self.add_time_features(df)
        
        # Aggregate to hourly if requested
        if aggregate_to_hourly:
            df = self.aggregate_hourly_prices(df)
        
        return df