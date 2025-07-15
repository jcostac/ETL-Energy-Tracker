import pandas as pd 

class DataValidationUtils:
    """
    Validate data for structural and data types for processed data (volumens and prices) before writing to parquets. 
    """
    def __init__(self):

        #processed data structure requirements
        """
        Initialize the DataValidationUtils instance with required column definitions for various energy market dataset schemas.
        
        Defines the expected columns for both processed and raw datasets, supporting multiple formats and sources for prices and volumes. These definitions are used for structural validation of DataFrames prior to further processing or saving.
        """
        self.processed_price_required_cols = ['datetime_utc', "id_mercado", "precio"]
        self.processed_volumenes_i90_required_cols = ['datetime_utc', "up", 'volumenes', 'id_mercado']
        self.processed_volumenes_omie_required_cols = ['datetime_utc', "uof", 'volumenes', 'id_mercado']
        self.processed_volumenes_mic_required_cols = ['datetime_utc', "uof", 'volumenes', "precio", 'id_mercado', "fecha_fichero"]
        self.processed_volumenes_i3_required_cols = ['datetime_utc', "tecnologia", 'volumenes','id_mercado']

        #raw data structure requirements
        self.raw_precios_esios_required_cols = ['datetime_utc', 'value', 'indicador_id']
        self.raw_precios_i90_required_cols = ["fecha", "precios", "Redespacho", "Sentido", "Unidad de Programación", "hora", "granularity"]
        self.raw_volumenes_i90_required_cols = ["Unidad de Programación", "fecha", "volumenes", "hora", "granularity"]
        self.raw_volumenes_i3_required_cols = ["Concepto", "fecha", "volumenes", "hora", "granularity"]
        self.raw_volumenes_omie_required_cols = ["Fecha", "Unidad", "id_mercado", "Energía Compra/Venta", "Ofertada (O)/Casada (C)"]
        self.raw_volumenes_mic_required_cols = ["Fecha", "Contrato", "Precio", "Cantidad", "id_mercado"]

    def _validate_data_common(self, df: pd.DataFrame, type: str, validation_schema_type: str) -> pd.DataFrame:
        """
        Performs structural and data type validation on a DataFrame for a specified dataset schema.
        
        Validates that the DataFrame has the correct columns and data types according to the given data type ('raw' or 'processed') and schema (e.g., "precios_esios", "volumenes_i90", etc.). Returns the validated DataFrame.
         
        Returns:
            pd.DataFrame: The validated DataFrame with enforced schema and data types.
        """
        df = self._validate_dtypes(df, type, validation_schema_type)
       
        df = self._validate_columns(df, type, validation_schema_type)
        
        return df

    def validate_processed_data(self, df: pd.DataFrame, validation_schema_type: str) -> pd.DataFrame:
        """
        Validate the structure and data types of a processed energy market dataset according to the specified schema.
        
        Parameters:
            df (pd.DataFrame): The processed dataset to validate.
            validation_schema_type (str): The schema type to validate against (e.g., "precios_esios", "volumenes_i90").
        
        Returns:
            pd.DataFrame: The validated DataFrame with enforced column presence and data types.
        """
        return self._validate_data_common(df, "processed", validation_schema_type)

    def validate_raw_data(self, df: pd.DataFrame, validation_schema_type: str) -> pd.DataFrame:
        """
        Validate the structure and data types of a raw energy market dataset according to the specified schema.
        
        Parameters:
        	df (pd.DataFrame): The raw dataset to validate.
        	validation_schema_type (str): The schema type defining required columns and data types (e.g., "precios_esios", "volumenes_i90", "volumenes_i3", "volumenes_omie", "volumenes_mic").
        
        Returns:
        	pd.DataFrame: The validated DataFrame with enforced schema and data types.
        """
        return self._validate_data_common(df, "raw", validation_schema_type)
               
    def _validate_dtypes(self, df: pd.DataFrame, type: str, validation_schema_type: str) -> pd.DataFrame:
        """
        Validate and enforce correct data types for columns in a DataFrame according to the specified dataset type and schema.
        
        Depending on the schema, converts date/time columns to pandas datetime (with UTC where applicable), and enforces appropriate numeric and string types for key columns such as prices, volumes, and categorical fields. Raises a ValueError if type conversion fails.
        
        Parameters:
            df (pd.DataFrame): The DataFrame to validate and convert.
            type (str): Indicates whether the data is 'raw' or 'processed'.
            validation_schema_type (str): The schema type (e.g., 'precios_esios', 'precios_i90', 'volumenes_i90', 'volumenes_omie').
        
        Returns:
            pd.DataFrame: The DataFrame with validated and converted data types.
        """
        try:
            
            # Create a copy to avoid modifying the original
            df = df.copy()
            
            if 'datetime_utc' in df.columns:
                df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)
            elif 'fecha' in df.columns:
                df['fecha'] = pd.to_datetime(df['fecha'])
            elif 'delivery_period_utc' in df.columns: #for mic market
                df['delivery_period_utc'] = pd.to_datetime(df['delivery_period_utc'], utc=True)
           
            #for processed data
            if type == "processed":
                #for precios related datasets
                if validation_schema_type in ["precios_esios", "precios_i90"]:
                    df['id_mercado'] = df['id_mercado'].astype('uint8')
                    df['precio'] = df['precio'].astype('float32')
                
                #for volumenes related datasets
                elif validation_schema_type in ["volumenes_i90", "volumenes_i3", "volumenes_omie"]:
                    df['id_mercado'] = df['id_mercado'].astype('uint8')
                    df['volumenes'] = df['volumenes'].astype('float32')

                    if 'up' in df.columns:
                        df['up'] = df['up'].astype('str')
                    if 'tecnologia' in df.columns:
                        df['tecnologia'] = df['tecnologia'].astype('str')
                    if 'uof' in df.columns:
                        df['uof'] = df['uof'].astype('str')
                    if 'precio' in df.columns:
                        df['precio'] = df['precio'].astype('float32')
                    if 'tipo_transaccion' in df.columns:
                        df['tipo_transaccion'] = df['tipo_transaccion'].astype('str')
                
                print(f"{type.capitalize()} {validation_schema_type} data types validated successfully.")
            
            #for raw data
            elif type == "raw":
                #for precios datasets coming from ESIOS
                if validation_schema_type == "precios_esios":
                    df['value'] = df['value'].astype('float32')
                    df['indicador_id'] = df['indicador_id'].astype('str')
                    
                #for precios datasets coming from I90
                elif validation_schema_type == "precios_i90":
                    df['precios'] = df['precios'].astype('float32')
                    df['Unidad de Programación'] = df['Unidad de Programación'].astype('str')
                    df['hora'] = df['hora'].astype('str')
                
                #for volumenes datasets coming from I90 or I3
                elif validation_schema_type in ["volumenes_i90", "volumenes_i3"]:
                    df['volumenes'] = df['volumenes'].astype('float32')
                    df['hora'] = df['hora'].astype('str')

                elif validation_schema_type == "volumenes_omie":
                    df['Unidad'] = df['Unidad'].astype('str')
                    df["Energía Compra/Venta"] = df["Energía Compra/Venta"].astype('float32')

                elif validation_schema_type == "volumenes_mic":
                    df['Cantidad'] = df['Cantidad'].astype('float32')
                    df['Precio'] = df['Precio'].astype('float32')

                
                print(f"{type.capitalize()} {validation_schema_type} data types validated successfully.")
    
            
        except Exception as e:
            print(f"Debug - DataFrame columns: {df.columns}")
            print(f"Debug - DataFrame dtypes before conversion: {df.dtypes}")
            raise ValueError(f"Error validating {validation_schema_type} {type} data types: {e}")
            
        return df

    def _validate_columns(self, df: pd.DataFrame, type: str, validation_schema_type: str) -> pd.DataFrame:
        """
        Verify that the DataFrame contains all required columns for the specified data type and schema.
        
        Raises:
            ValueError: If any required column is missing from the DataFrame.
        
        Returns:
            The input DataFrame if all required columns are present.
        """
        required_cols = None
        
        if type == "processed":
            if validation_schema_type == "precios_esios" or validation_schema_type == "precios_i90":
                required_cols = self.processed_price_required_cols
            elif validation_schema_type == "volumenes_i90":
                required_cols = self.processed_volumenes_i90_required_cols
            elif validation_schema_type == "volumenes_i3":
                required_cols = self.processed_volumenes_i3_required_cols
            elif validation_schema_type == "volumenes_omie":
                required_cols = self.processed_volumenes_omie_required_cols
            elif validation_schema_type == "volumenes_mic":
                required_cols = self.processed_volumenes_mic_required_cols

        elif type == "raw":
            if validation_schema_type == "precios_esios":
                required_cols = self.raw_precios_esios_required_cols
            elif validation_schema_type == "precios_i90":
                required_cols = self.raw_precios_i90_required_cols
            elif validation_schema_type == "volumenes_i90":
                required_cols = self.raw_volumenes_i90_required_cols
            elif validation_schema_type == "volumenes_i3":
                required_cols = self.raw_volumenes_i3_required_cols
            elif validation_schema_type == "volumenes_omie":
                required_cols = self.raw_volumenes_omie_required_cols
            elif validation_schema_type == "volumenes_mic":
                required_cols = self.raw_volumenes_mic_required_cols

        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Expected: {required_cols}")
        
        print(f"{type.upper()} {validation_schema_type.upper()} data structure validated successfully.")
        
        return df
    
    
    
    