import pandas as pd 

class DataValidationUtils:
    """
    Validate data for structural and data types for processed data (volumens and prices) before writing to parquets. 
    """
    def __init__(self):

        #processed data structure requirements
        self.processed_price_required_cols = ['datetime_utc', 'precio', 'id_mercado']
        self.processed_volumenesi90_required_cols = ['datetime_utc', 'volumenes', 'id_mercado', "up_id"]
        self.processed_volumenesi3_required_cols = ['datetime_utc', 'volumenes', 'id_mercado', "tecnologia_id"]

        #raw data structure requirements
        self.raw_price_required_cols = ['datetime_utc', 'value', 'indicador_id']
        self.raw_precios_i90_required_cols = []
        self.raw_volumenes_required_cols = []

    def _validate_data_common(self, df: pd.DataFrame, type: str, data: str) -> pd.DataFrame:
        """
        Common validation logic for both raw and processed data.

        Args:
            df (pd.DataFrame): Input DataFrame.
            type (str): Type of data ('raw' or 'processed').
            data (str): Specific dataset type ('precios', 'volumenes_i90', etc.).

        Returns:
            pd.DataFrame: Validated DataFrame.
        """
        df = self._validate_dtypes(df, type, data)
       
        df = self._validate_columns(df, type, data)
        
        return df

    def validate_processed_data(self, df: pd.DataFrame, data: str) -> pd.DataFrame:
        """
        Validate processed data for structural and data types issues.

        Args:
            df (pd.DataFrame): Input DataFrame with processed data.
            data (str): The dataset that will be validated (e.g., 'precios', 'volumenes_i90').

        Returns:
            pd.DataFrame: Validated DataFrame.
        """
        return self._validate_data_common(df, "processed", data)

    def validate_raw_data(self, df: pd.DataFrame, data: str) -> pd.DataFrame:
        """
        Validate raw data for structural and data types issues.

        Args:
            df (pd.DataFrame): Input DataFrame with raw data.
            data (str): The dataset that will be validated (e.g., 'precios').

        Returns:
            pd.DataFrame: Validated DataFrame.
        """
        return self._validate_data_common(df, "raw", data)
               
    def _validate_dtypes(self, df: pd.DataFrame, type: str, data: str) -> pd.DataFrame:
        """
        Validate data types for different datasets.
        
        Args:
            df: Input DataFrame with data
            dataset_type: Either 'raw' or 'processed'
            data_type: Type of data (precio, volumenes_i90, volumenes_i3)
        """
        try:
            
            # Create a copy to avoid modifying the original
            df = df.copy()
            
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)
           
            #for processed data
            if type == "processed":
                #for the different datasets that can be processed
                if data == "precios":
                    df['id_mercado'] = df['id_mercado'].astype('uint8')
                    df['precio'] = df['precio'].astype('float32')
                
                elif data in ["volumenes_i90", "volumenes_i3"]:
                    # Add volumenes validation here
                    pass

                elif data == "precios_i90":
                    pass
                
                print(f"Processed {data} {type} data types validated successfully. {df.dtypes}")
            
            #for raw data
            elif type == "raw":
                #for the different datasets that can be raw
                if data == "precios":
                    df['value'] = df['value'].astype('float32')
                    print(f"Debug - After value conversion: {len(df)} rows")
                    df['indicador_id'] = df['indicador_id'].astype('str')
                    print(f"Debug - After indicador_id conversion: {len(df)} rows")

                elif data == "precios_i90":
                    pass

                elif data in ["volumenes_i90", "volumenes_i3"]:
                    pass
                
                print(f"Raw {data} {type} data types validated successfully. {df.dtypes}")
    
            
        except Exception as e:
            print(f"Debug - DataFrame columns: {df.columns}")
            print(f"Debug - DataFrame dtypes before conversion: {df.dtypes}")
            raise ValueError(f"Error validating {data} {type} data types: {e}")
            
        return df

    def _validate_columns(self, df: pd.DataFrame, type: str, data: str) -> pd.DataFrame:
        """
        Validate data structure to make sure all columns are present.
        
        Args:
            df: Input DataFrame with data
            dataset_type: Either 'raw' or 'processed'
            data_type: Type of data (precio, volumenes_i90, volumenes_i3)
        """
        required_cols = None
        
        if type == "processed":
            if data == "precios":
                required_cols = self.processed_price_required_cols
            elif data == "volumenes_i90":
                required_cols = self.processed_volumenesi90_required_cols
            elif data == "volumenes_i3":
                required_cols = self.processed_volumenesi3_required_cols

        elif type == "raw":
            if data == "precios":
                required_cols = self.raw_price_required_cols
            elif data == "precios_i90":
                required_cols = self.raw_precios_i90_required_cols
            elif data in ["volumenes_i90", "volumenes_i3"]:
                required_cols = self.raw_volumenes_required_cols
            
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Expected: {required_cols}")
        
        print(f"{type.upper()} {data.upper()} data structure validated successfully.")
        
        return df
    
    
    
    