import pandas as pd 

class DataValidationUtils:
    """
    Validate data for structural and data types for processed data (volumens and prices) before writing to parquets. 
    """
    def __init__(self):

        #processed data structure requirements
        self.processed_price_required_cols = ['datetime_utc', 'price', 'id_mercado']
        self.processed_volumenesi90_required_cols = ['datetime_utc', 'volumenes', 'id_mercado', "up_id"]
        self.processed_volumenesi3_required_cols = ['datetime_utc', 'volumenes', 'id_mercado', "tecnologia_id"]

        #raw data structure requirements
        self.raw_price_required_cols = ['datetime_utc', 'value', 'indicador_id', 'id_mercado']
        self.raw_precios_i90_required_cols = []
        self.raw_volumenes_required_cols = []

    def validate_processed_data(self, df: pd.DataFrame, data: str) -> pd.DataFrame:
        """
        Validate price data for structural and data types issues.
        
        Args:
            df (pd.DataFrame): Input DataFrame with price data
            type (str): Type of data to validate (raw or processed)
            data (str): the datat that will be validated(price, volumenes_i90, volumenes_i3, precios_i90)
        """
        type = "processed"

        if data == 'precios':
            df = self._validate_dtypes(df, type, data)
            df = self._validate_columns(df, type, data)
        elif data == 'volumenes_i90':
            df = self._validate_dtypes(df, type, data)
            df = self._validate_columns(df, type, data)
        elif data == 'volumenes_i3':
            df = self._validate_dtypes(df, type, data)
            df = self._validate_columns(df, type, data)
        elif data == 'precios_i90':
            df = self._validate_dtypes(df, type, "precios_i90")
            df = self._validate_columns(df, type, "precios_i90")

        return df

    def validate_raw_data(self, df, data):
        """
        Validate raw data for structural and data types issues.
        
        Args:
            df: Input DataFrame with raw data
        """
        type = "raw"

        if data == 'precios':
            df = self._validate_dtypes(df, type, data)
            df = self._validate_columns(df, type, data)
        elif data == 'volumenes_i90':
            df = self._validate_dtypes(df, type, data)
            df = self._validate_columns(df, type, data)
        elif data == 'volumenes_i3':
            df = self._validate_dtypes(df, type, data)
            df = self._validate_columns(df, type, data)
        elif data == 'precios_i90':
            df = self._validate_dtypes(df, type, "precios_i90")
            df = self._validate_columns(df, type, "precios_i90")

        return df
               
    def _validate_dtypes(self, df: pd.DataFrame, type: str, data: str) -> pd.DataFrame:
        """
        Validate data types for different datasets.
        
        Args:
            df: Input DataFrame with data
            dataset_type: Either 'raw' or 'processed'
            data_type: Type of data (price, volumenes_i90, volumenes_i3)
        """
        try:
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)
            
            #for processed data
            if type == "processed":
                #for the different datasets that can be processed
                if data == "precios":
                    df['id_mercado'] = df['id_mercado'].astype('uint8')
                    df['price'] = df['price'].astype('float32')
                
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
                    df['id_mercado'] = df['id_mercado'].astype('str')
                    df['precio'] = df['precio'].astype('float32')
                    df['indicador_id'] = df['indicador_id'].astype('str')

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
            data_type: Type of data (price, volumenes_i90, volumenes_i3)
        """
        required_cols = None
        
        if type == "processed":
            if data == "price":
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
        
        print(f"{data.capitalize()} {type} data structure validated successfully. {df.columns}")
        
        return df
    
    
    
    