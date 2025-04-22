import pandas as pd 

class DataValidationUtils:
    """
    Validate data for structural and data types for processed data (volumens and prices) before writing to parquets. 
    """
    def __init__(self):
        self.price_required_cols = ['datetime_utc', 'price', 'id_mercado']
        self.volumenesi90_required_cols = ['datetime_utc', 'volumenes', 'id_mercado', "up_id"]
        self.volumenesi3_required_cols = ['datetime_utc', 'volumenes', 'id_mercado', "tecnologia_id"]

    def validate_data(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Validate price data for structural and data types issues.
        
        Args:
            df (pd.DataFrame): Input DataFrame with price data
        """
        if dataset_type == 'precios':
            df = self._validate_price_dtypes(df)
            df = self._validate_price_cols(df)
        elif dataset_type == 'volumenes_i90':
            df = self._validate_volumenesi90_dtypes(df)
            df = self._validate_volumenesi90_cols(df)
        elif dataset_type == 'volumenes_i3':
            df = self._validate_volumenesi3_dtypes(df)
            df = self._validate_volumenesi3_cols(df)

        return df

    def _validate_price_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate price data for common issues.
        
        Args:
            df (pd.DataFrame): Input DataFrame with price data
            
        """
        
        try:
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)
            df['id_mercado'] = df['id_mercado'].astype('uint8')
            df['price'] = df['price'].astype('float32')

        except Exception as e:
            raise ValueError(f"Error validating price data types: {e}")

        return df

    def _validate_price_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate price data structure to make sure all cols are present and have the correct dtype. 
        
        Args:
            df (pd.DataFrame): Input DataFrame with price data
        """
        
        if not all(col in df.columns for col in self.price_required_cols):
            raise ValueError(f"Missing required columns. Expected: {self.price_required_cols}")
        
        return df
