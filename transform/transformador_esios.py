import pandas as pd
from datetime import datetime
from utilidades.etl_date_utils import DateUtilsETL
from utilidades.db_utils import DatabaseUtils
import pytz
import pretty_errors

class TransformadorESIOS:
    def __init__(self):
        self.indicators_to_filter_by_country = [600,612,613,614,615,616,617,618,1782]
        
    def transform_price_data(self, data: dict) -> pd.DataFrame:
        
        #extract granularity from json data -> can be "Quince minutos" or "Hora"
        granularity = data["indicator"]["tiempo"][0]["name"]

        #Procesamos los datos de interés en un dataframe que se ecnuentran en el key values
        df_data = pd.DataFrame(data['indicator']['values'])

        if len(df_data)>0: #si no tenemos un df vacío

            if int(self.indicator_id) in self.indicators_to_filter_by_country:
                #filter by country if applicable
                df_data = df_data[df_data['geo_id'] == 3].reset_index() #geo id 3 is Spain
            
            # FECHA COLUMN PROCESSING
            # Extremos la fecha de la columna datetime ie. from 2025-03-25T21:00:00Z to 2025-03-25
            df_data['fecha'] = df_data.apply(lambda row: datetime.strptime(row['datetime'][:10],'%Y-%m-%d').date(), axis=1)


            # HORA COLUMN PROCESSING
            #extraemos la zona horaria de la columna datetime ie. from "datetime": "2025-03-24T06:00:00.000+01:00" to 1
            df_data['zona_horaria'] = df_data.apply(lambda row: int(row['datetime'][-4]), axis=1)

            #Procesamos la hora para dejarla en el formato de la BBDD y aplicamos los cambios necesarios para los dias de 23 y 25 horas
            if granularity == "Quince minutos":
                # Extract hour and minute from datetime, keeping the HH:MM format
                df_data['hora_real'] = df_data.apply(lambda row: row['datetime'][11:16], axis=1)
                # creamos la columna hora con el formato de la BBDD y aplicamos los cambios necesarios para los dias de 23 y 25 horas
                df_data["hora"] = df_data.apply(TimeUtils.ajuste_quinceminutal_ESIOS, axis=1, special_dates=filtered_transition_dates)
                
            elif granularity == "Hora":
                # Extraemos la hora de la columna datetime ie. from "datetime": "2025-03-24T06:00:00.000+01:00" to 6
                df_data['hora_real'] = df_data.apply(lambda row: int(row['datetime'][11:13])+1, axis=1)
                # creamos la columna hora con el formato de la BBDD y aplicamos los cambios necesarios para los dias de 23 y 25 horas
                df_data["hora"] = df_data.apply(TimeUtils.ajuste_horario_ESIOS, axis=1, special_dates=filtered_transition_dates)
            
            else:
                raise ValueError(f"Granularity not supported. Current granularity: {granularity}")
        

            # MISCELLANEOUS COLUMN PROCESSING
            #rename value to precio and select columns of interest
            df_data.rename(columns={'value': 'precio'}, inplace=True)
            # Select and reorder columns
            df_data = df_data[['fecha', 'hora', 'precio', 'geo_id', 'zona_horaria']]
            #add id_mercado to dataframe 
            df_data['id_mercado'] = self.market_id_map[indicator_id] #ie. if diario -> indicador esiso is 600 -> then id_mercado is 1
          
            
            #FINAL PROCESSING
            df_data['fecha'] = pd.to_datetime(df_data['fecha'])
            df_data = df_data[['fecha', 'hora', 'precio', 'id_mercado']]
            df_data.sort_values(by=['fecha', 'hora'], inplace=True)
            df_data = df_data[df_data['fecha'] > pd.to_datetime(fecha_inicio_carga)]
            df_data = df_data[df_data['fecha'] <= pd.to_datetime(fecha_fin_carga)]

        return df_data
        pass
    
    def convert_granularity(self, data: dict) -> pd.DataFrame:
        """
        Converts the granularity of the data to the desired granularity. Resample from hourly to 15-min data
        for data before the granulairty change date for a market. 
        """
        
        # Extract data on change date and modify hour format to 15-min format
        change_date_data = data[data['datetime_utc'].dt.date <= change_date.date()].copy()

        # Map each hour to its corresponding 15-min format
        minute_map = {0: '00', 1: '15', 2: '30', 3: '45'}
        change_date_data['hora'] = change_date_data.apply(
            lambda row: f"{row['datetime'].hour:02d}:{minute_map[row.name % 4]}:00", 
            axis=1  # Change data to format 00:00:00, 00:15:00, 00:30:00, 00:45:00
        )
                    
        pass
  
class Diario(TransformadorESIOS):
    def __init__(self):
        super().__init__()
        self.indicator_id = self.get_indicator_id("Diario") # ie. 600

    def transform_price_data(self, data: dict) -> pd.DataFrame:
        df_data = super().transform_price_data(data)
        return df_data

class Intra(TransformadorESIOS):
    def __init__(self):
        super().__init__()
        self.intra_name_map = {
            1: "Intra 1",
            2: "Intra 2",
            3: "Intra 3",
            4: "Intra 4",
            5: "Intra 5",
            6: "Intra 6",
            7: "Intra 7"
        }
        self.intra_reduccion_fecha = datetime.strptime('2024-06-13', '%Y-%m-%d')
        self.cambio_granularidad_fecha = datetime.strptime('2025-03-19', '%Y-%m-%d')

    def transform_price_data(self, data: dict, intra_num: int) -> pd.DataFrame:
        """
        Transforms the price data for a specific intraday market.
        
        Args:
            data (dict): Dictionary containing the API response data
            intra_num (int): Intraday market number (1-7)
            
        Returns:
            pd.DataFrame: DataFrame with transformed intraday price data
        """
        self.indicator_id = self.get_indicator_id(self.intra_name_map[intra_num])
        df_data = super().transform_price_data(data)
        return df_data

class Secundaria(TransformadorESIOS):
    def __init__(self):
        super().__init__()
        self.secundaria_name_map = {
            1: "Secundaria a subir",  # indicator 2130
            2: "Secundaria a bajar",  # indicator 634
        }
        self.precio_dual_fecha = datetime.strptime('2024-11-20', '%Y-%m-%d')
        self.cambio_granularidad_fecha = datetime.strptime('2022-05-24', '%Y-%m-%d')
    
    def transform_price_data(self, data: dict, secundaria_id: int) -> pd.DataFrame:
        """
        Transforms the price data for secondary regulation.
        
        Args:
            data (dict): Dictionary containing the API response data
            secundaria_id (int): Secondary regulation type (1: subir, 2: bajar)
            
        Returns:
            pd.DataFrame: DataFrame with transformed secondary regulation price data
        """
        self.indicator_id = self.get_indicator_id(self.secundaria_name_map[secundaria_id])
        df_data = super().transform_price_data(data)
        return df_data

class Terciaria(TransformadorESIOS):
    def __init__(self):
        super().__init__()
        self.terciaria_name_map = {
            1: "Terciaria a subir",          # 677 -> changes to 2197
            2: "Terciaria a bajar",          # 676 -> changes to 2197
            3: "Terciaria directa a subir",  # 10400 (not affected)
            4: "Terciaria directa a bajar",  # 10401 (not affected)
            5: "Terciaria programada unico", # 2197 (new indicator after change)
        }
        self.precio_unico_fecha = datetime.strptime('2024-12-10', '%Y-%m-%d')
        self.cambio_granularidad_fecha = datetime.strptime('2022-05-24', '%Y-%m-%d')
    
    def transform_price_data(self, data: dict, terciaria_id: int) -> pd.DataFrame:
        """
        Transforms the price data for tertiary regulation.
        
        Args:
            data (dict): Dictionary containing the API response data
            terciaria_id (int): Tertiary regulation type (1-5)
            
        Returns:
            pd.DataFrame: DataFrame with transformed tertiary regulation price data
        """
        self.indicator_id = self.get_indicator_id(self.terciaria_name_map[terciaria_id])
        df_data = super().transform_price_data(data)
        return df_data

class RR(TransformadorESIOS):
    def __init__(self):
        super().__init__()
        self.indicator_id = self.get_indicator_id("RR a subir")
        self.cambio_granularidad_fecha = datetime.strptime('2022-05-24', '%Y-%m-%d')
    
    def transform_price_data(self, data: dict) -> pd.DataFrame:
        """
        Transforms the price data for RR (Replacement Reserve).
        
        Args:
            data (dict): Dictionary containing the API response data
            
        Returns:
            pd.DataFrame: DataFrame with transformed RR price data
        """
        df_data = super().transform_price_data(data)
        return df_data
