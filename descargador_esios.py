import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import sys
import os
import pretty_errors
from dotenv import load_dotenv
load_dotenv()

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Use absolute imports
from utilidades.db_utils import DatabaseUtils
from utilidades.etl_date_utils import OldTimeUtils, TimeUtils

class DescargadorESIOS:

    def __init__(self):
        self.esios_token = os.getenv('ESIOS_TOKEN')
        self.download_window = 93
        self.bbdd_engine = DatabaseUtils.create_engine('pruebas_BT')
        self.indicator_id_map, self.market_id_map = self.get_market_id_mapping()
        self.indicators_to_filter_by_country = [600,612,613,614,615,616,617,618,1782]

    def get_market_id_mapping(self) -> dict[str, str]:
        """
        Obtiene el mapping de los IDs de los mercados de ESIOS.
        Returns:
            dict: 1. indicator_id_map: Un diccionario con los nombres de los mercados y sus respectivos IDs de ESIOS.
                        i.e {'Intra 4': 600, 'Intra 5': 612, 'Intra 6': 613, 'Intra 7': 614}
                  2. market_id_map: Un diccionario con los IDs de los mercados de ESIOS y sus IDs de mercado en la BBDD.
                        i.e {600: 1, 612: 2, 613: 3, 614: 4}
        """
        #get all market ids with indicator_esios_precios != 0
        df_mercados = DatabaseUtils.read_table(self.bbdd_engine, 'Mercados', columns=['id', 'mercado', 'indicador_esios_precios as indicador', 'is_quinceminutal'], 
                                               where_clause='indicador_esios_precios != 0')
        
        #get idnicator map with mercado as key and indicator as value i.e {'Intra 4': 600, 'Intra 5': 612, 'Intra 6': 613, 'Intra 7': 614}
        indicator_map = dict(zip(df_mercados['mercado'], df_mercados['indicador']))
        market_id_map = dict(zip(df_mercados['indicador'], df_mercados['id']))

        
        #convert indicator value to str to avoid type errors
        indicator_id_map = {str(key): str(value) for key, value in indicator_map.items()}
        market_id_map = {str(key): str(value) for key, value in market_id_map.items()}

        return indicator_id_map, market_id_map
    
    def get_indicator_id(self, mercado: str):
        """
        Obtiene el ID del indicador de ESIOS para un mercado específico.

        Args:
            indicator_id (int): El ID del indicador de ESIOS."""
        try: 
            return self.indicator_id_map[mercado]
        except KeyError:
            raise ValueError(f"Mercado no encontrado en el mapping: {mercado}")

    def get_esios_data(self, indicator_id: str, fecha_inicio_carga: str = None, fecha_fin_carga: str = None):
        """
        Descarga los datos de ESIOS para un indicador específico en un rango de fechas.
        
        Args:
            indicator_id (str): El ID del indicador para el cual se están solicitando los datos.
            fecha_inicio_carga (str): La fecha de inicio para la solicitud de datos en formato 'YYYY-MM-DD'.
            fecha_fin_carga (str): La fecha de fin para la solicitud de datos en formato 'YYYY-MM-DD'.
        
        Returns:
            pd.DataFrame: DataFrame con los datos solicitados.
        """
        if hasattr(self, 'cambio_granularidad_fecha'):
            change_date = self.cambio_granularidad_fecha
            fecha_inicio_dt = datetime.strptime(fecha_inicio_carga, '%Y-%m-%d')
            fecha_fin_dt = datetime.strptime(fecha_fin_carga, '%Y-%m-%d')
            
            dfs = []
            
            # If date range spans the granularity change
            if fecha_inicio_dt < change_date and fecha_fin_dt > change_date:
                # Get hourly data up to change date (inclusive)
                df_before = self._make_esios_request(
                    indicator_id,
                    fecha_inicio_carga,
                    change_date.strftime('%Y-%m-%d')
                )
                
                if not df_before.empty:
                    # Extract change date data and modify hour format to 15-min format
                    change_date_data = df_before[df_before['fecha'] == pd.Timestamp(change_date)].copy()
                    # Map each hour to its corresponding 15-min format
                    minute_map = {0: '00', 1: '15', 2: '30', 3: '45'}
                    change_date_data['hora'] = change_date_data.apply(
                        lambda row: f"{int(row['hora']):02d}:{minute_map[row.name % 4]}", 
                        axis=1
                    )
                    
                    # Remove change date from before DataFrame and append modified change date data
                    df_before = df_before[df_before['fecha'] < pd.Timestamp(change_date)]
                    dfs.append(df_before)
                    dfs.append(change_date_data)
                
                # Get 15-min data after change date
                df_after = self._make_esios_request(
                    indicator_id,
                    (change_date).strftime('%Y-%m-%d'),
                    fecha_fin_carga
                )
                if not df_after.empty:
                    dfs.append(df_after)
                
                if dfs:
                    final_df = pd.concat(dfs, ignore_index=True)
                    final_df.sort_values(['fecha', 'hora'], inplace=True)
                    return final_df
                
                return pd.DataFrame()
            
            # If entirely before or after change date, use original logic
            return self._make_esios_request(indicator_id, fecha_inicio_carga, fecha_fin_carga)
        
        # For markets without granularity changes, use original logic
        return self._make_esios_request(indicator_id, fecha_inicio_carga, fecha_fin_carga)

    def _make_esios_request(self, indicator_id: str, fecha_inicio_carga: str, fecha_fin_carga: str) -> pd.DataFrame:
        """
        Internal method that handles the actual API call and data processing.

        Args:
            indicator_id (str): The ID of the indicator for which data is being requested.
            fecha_inicio_carga (str): The start date for the data request in 'YYYY-MM-DD' format.
            fecha_fin_carga (str): The end date for the data request in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: A DataFrame containing the requested data from the ESIOS API.

        Raises:
            ValueError: If the start date is greater than the end date.
            Exception: If there is an error during the API call or data processing.
        """

        #check if fecha inicio < fecha fin 
        if fecha_inicio_carga and fecha_fin_carga:
            if fecha_inicio_carga > fecha_fin_carga:
                raise ValueError("La fecha de inicio de carga no puede ser mayor que la fecha de fin de carga")


        if fecha_inicio_carga is None and fecha_fin_carga is None:
            fecha_inicio_carga_dt = datetime.now() - timedelta(days=self.download_window) # 93 days ago
            fecha_inicio_carga = fecha_inicio_carga_dt.strftime('%Y-%m-%d') #string format
            fecha_fin_carga_dt = datetime.now() - timedelta(days=self.download_window) + timedelta(days=1) # 92 days from now
            fecha_fin_carga = fecha_fin_carga_dt.strftime('%Y-%m-%d') #string format
            print(f"No se han proporcionado fechas de carga, se descargarán datos entre {fecha_inicio_carga} y {fecha_fin_carga}")

        else:
            #convert sttring to datetime
            fecha_inicio_carga_dt = datetime.strptime(fecha_inicio_carga, '%Y-%m-%d')
            fecha_fin_carga_dt = datetime.strptime(fecha_fin_carga, '%Y-%m-%d')
        

        filtered_transition_dates = TimeUtils.get_transition_dates(fecha_inicio_carga_dt, fecha_fin_carga_dt)

        try:
            # Download ESIOS data for each market of interest
            url = f"http://api.esios.ree.es/indicators/{indicator_id}"
            params = {
                'start_date': f"{fecha_inicio_carga}T21:00:00Z",
                'end_date': f"{fecha_fin_carga}T23:00:00Z"
            }
            headers = {
                'x-api-key': self.esios_token,
                'Authorization': f'Token token={self.esios_token}'
            }
            response = requests.get(url, headers=headers, params=params)
            data = response.json()



        except Exception as e:
            raise Exception(f"Error al descargar los datos de ESIOS para el indicador {indicator_id}: {e}")


        #extract granularity from json data -> can be "Quince minutos" or "Hora"
        granularity = data["indicator"]["tiempo"][0]["name"]

        #Procesamos los datos de interés en un dataframe que se ecnuentran en el key values
        df_data = pd.DataFrame(data['indicator']['values'])

        if len(df_data)>0: #si no tenemos un df vacío

            if int(indicator_id) in self.indicators_to_filter_by_country:
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

    def save_data(self, df_data: pd.DataFrame, dev: bool, table_name: str = None):
        """
        Saves data to the database, handling granularity changes if applicable.
        For classes with granularity changes (Intra, Secundaria, Terciaria, RR), data is saved to:
        - Precios_horarios before the change date
        - Precios_quinceminutales after the change date
        
        Args:
            df_data (pd.DataFrame): DataFrame containing the data to save
            dev (bool): If True, save to the development database
            table_name (str, optional): Override the default table name determination
        
        Note:
            Child classes with granularity changes must define cambio_granularidad_fecha as an attribute
        """
        if df_data.empty:
            print("No hay datos para guardar")
            return
        
        # Define the tables to save to in case dev is True or False (development or production)
        prod_tables = ['Precios_horarios', 'Precios_quinceminutales']
        dev_tables = ['Precios_horarios_dev', 'Precios_quinceminutales_dev']

        if dev: #if dev is True, save to development tables
            tabla_horaria = dev_tables[0]
            tabla_quinceminutal = dev_tables[1]
        else: #if dev is False, save to production tables
            tabla_horaria = prod_tables[0]
            tabla_quinceminutal = prod_tables[1]

        # Print start of save process for the specific child class
        print(f"Inicio del proceso de guardado para {self.__class__.__name__}")

        # Get the first and last date in the dataframe and conveer to right format
        fecha_inicio = df_data['fecha'].min().strftime('%Y-%m-%d')
        fecha_fin = df_data['fecha'].max().strftime('%Y-%m-%d')
        fecha_inicio_dt = datetime.strptime(fecha_inicio, '%Y-%m-%d')
        fecha_fin_dt = datetime.strptime(fecha_fin, '%Y-%m-%d')
        
        # If table_name is provided, use it directly
        if table_name:
            print(f"Guardando datos en {table_name}")
            DatabaseUtils.write_table(self.bbdd_engine, df_data, table_name)
            return
        
        # Handle classes with granularity changes
        if hasattr(self, 'cambio_granularidad_fecha'):
            change_date = self.cambio_granularidad_fecha
            
            # Handle data that spans the granularity change date
            if fecha_inicio_dt < change_date and fecha_fin_dt >= change_date:
                # Split data into before and after change date
                before_change = df_data[df_data['fecha'] < pd.Timestamp(change_date)]
                after_change = df_data[df_data['fecha'] >= pd.Timestamp(change_date)]
                
                # Drop duplicates before saving to avoid integrity errors
                if not before_change.empty:
                    print(f"Guardando datos anteriores al {change_date.strftime('%Y-%m-%d')} en {tabla_horaria}")
                    DatabaseUtils.write_table(self.bbdd_engine, before_change, tabla_horaria)
                
                # Save data after granularity change date to 15-min prices table if exists
                if not after_change.empty:
                    print(f"Guardando datos posteriores al {change_date.strftime('%Y-%m-%d')} en {tabla_quinceminutal}")
                    DatabaseUtils.write_table(self.bbdd_engine, after_change, tabla_quinceminutal)
                
            # Handle data that is entirely after the change date -> save to 15-min table
            elif fecha_inicio_dt >= change_date:
                print(f"Guardando datos en {tabla_quinceminutal} (todos los datos ocurren después del cambio de granularidad el  {change_date.strftime('%Y-%m-%d')})")
                DatabaseUtils.write_table(self.bbdd_engine, df_data, tabla_quinceminutal)
                
            # Handle data that is entirely before the change date -> save to hourly table
            else:
                print(f"Guardando datos en {tabla_horaria} (todos los datos ocurren antes del cambio de granularidad el {change_date.strftime('%Y-%m-%d')})")
                DatabaseUtils.write_table(self.bbdd_engine, df_data, tabla_horaria)
        
        # Handle classes without granularity changes
        else:
            has_minutes = any(':' in str(h) for h in df_data['hora'] if h is not None)
            table_name = tabla_quinceminutal if has_minutes else tabla_horaria
            print(f"Guardando datos en {table_name}")
            DatabaseUtils.write_table(self.bbdd_engine, df_data, table_name)

    def get_db_data(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None, 
                    indicator_ids: Optional[list[int]] = None, is_quinceminutal: bool = False):
        """
        Obtiene los datos de la base de datos.
        
        Args:
            fecha_inicio_carga: Fecha inicial en formato YYYY-MM-DD
            fecha_fin_carga: Fecha final en formato YYYY-MM-DD
            indicator_ids: Lista de IDs de indicadores ESIOS para filtrar
            is_quinceminutal: Si True, obtiene datos de Precios_quinceminutales, si False, de Precios_horarios
            
        Returns:
            pd.DataFrame: DataFrame con los datos de la base de datos
        """
        #cehck if fecha inicio carga is older than fecha fin carga
        if fecha_inicio_carga and fecha_fin_carga:
            if fecha_inicio_carga > fecha_fin_carga:
                raise ValueError("La fecha de inicio de carga no puede ser mayor que la fecha de fin de carga")
            else:
                # Construct where clause for dates
                where_clause = ""
                where_clause = f'fecha between "{fecha_inicio_carga}" and "{fecha_fin_carga}"'
                
        # Add market filter if indicator_ids provided
        if indicator_ids:
            mercados_list = ", ".join([str(item) for item in indicator_ids])
            market_filter = f'id_mercado in ({mercados_list})'
            where_clause = f'{where_clause} and {market_filter}' if where_clause else market_filter

        # Select appropriate table based on is_quinceminutal parameter
        table_name = 'Precios_quinceminutales' if is_quinceminutal else 'Precios_horarios'

        # Get data from database
        df = DatabaseUtils.read_table(
            self.bbdd_engine, 
            table_name,
            where_clause=where_clause
        )

        return df

class Diario(DescargadorESIOS):

    def __init__(self):
        super().__init__() 
        self.indicator_id = self.get_indicator_id("Diario") # ie. 600

    def get_prices(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None):
        return self.get_esios_data(self.indicator_id, fecha_inicio_carga, fecha_fin_carga)
    
    def save_data(self, df_data: pd.DataFrame, dev: bool):
        """
        Saves daily market data to the database. Always saves to Precios_horarios
        regardless of date.
        
        Args:
            df_data (pd.DataFrame): DataFrame containing the daily market data
        """
        if dev:
            table_name = 'Precios_horarios_dev'
        else:
            table_name = 'Precios_horarios'

        super().save_data(df_data, dev, table_name)

    def get_db_prices(self, fecha_inicio: str, fecha_fin: str):
        return super().get_db_data(fecha_inicio, fecha_fin, indicator_ids= [self.indicator_id], is_quinceminutal=False)

class Intra(DescargadorESIOS):

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
        
    def get_prices(self, fecha_inicio_carga: str, fecha_fin_carga: str, intra_lst: list[int]) -> pd.DataFrame:
        """
        Descarga los datos de ESIOS para los mercados intradiarios especificados.
        Automatically handles regulatory changes by downloading only available intras after cutoff dates.

        Args:
            fecha_inicio_carga (str): La fecha de inicio de la carga en formato YYYY-MM-DD
            fecha_fin_carga (str): La fecha de fin de la carga en formato YYYY-MM-DD
            intra_ids (list[int]): Lista de IDs de mercados intradiarios (1-7)

        Returns:
            pd.DataFrame: DataFrame con los precios de los mercados intradiarios solicitados
        """
    
        # Validate function input for intra ids
        invalid_intra_nums = [intra_num for intra_num in intra_lst if intra_num not in self.intra_name_map]
        if invalid_intra_nums:
            raise ValueError(f"Invalid intra markets: {invalid_intra_nums}. Valid intra markets are 1-7")

        # Convert dates to datetime for comparison
        fecha_inicio_dt = datetime.strptime(fecha_inicio_carga, '%Y-%m-%d')
        fecha_fin_dt = datetime.strptime(fecha_fin_carga, '%Y-%m-%d')
        
        dfs = [] #list of dataframes to concatenate

        # If date range spans the regulatory change
        if fecha_inicio_dt < self.intra_reduccion_fecha and fecha_fin_dt > self.intra_reduccion_fecha:
            # Split into two periods: before and after regulatory change
            
            # Period 1: Before regulatory change - download all requested intras up to the intra reduction date
            for intra_num in intra_lst:
                intra_indicator_id = self.get_indicator_id(self.intra_name_map[intra_num])
                df = self.get_esios_data(
                    intra_indicator_id,
                    fecha_inicio_carga,
                    (self.intra_reduccion_fecha.strftime('%Y-%m-%d')) #en fecha reduccion intra aun no aplica el cambio
                )
                if not df.empty:
                    dfs.append(df)

            # Period 2: After regulatory change - download only Intras 1-3 after the intra reduction date
            intras_after_change = [intra_num for intra_num in intra_lst if intra_num <= 3]
            #download all intras after regulatory change
            for intra_num in intras_after_change:
                intra_indicator_id = self.get_indicator_id(self.intra_name_map[intra_num])
                df = self.get_esios_data(
                    intra_indicator_id,
                    (self.intra_reduccion_fecha).strftime('%Y-%m-%d'),
                    fecha_fin_carga
                )
                if not df.empty:
                    dfs.append(df)

        # If date range entirely after regulatory change
        elif fecha_inicio_dt >= self.intra_reduccion_fecha:
            intras_after_change = [intra_num for intra_num in intra_lst if intra_num <= 3]

            #download all intras after regulatory change
            for intra_num in intras_after_change:
                intra_indicator_id = self.get_indicator_id(self.intra_name_map[intra_num])
                df = self.get_esios_data(
                    intra_indicator_id,
                    fecha_inicio_carga,
                    fecha_fin_carga
                )
                if not df.empty:
                    dfs.append(df)

        # If date range entirely before regulatory change
        else:
            #download all intras before regulatory change
            for intra_num in intra_lst:
                #get indicador id for each intra
                intra_indicator_id = self.get_indicator_id(self.intra_name_map[intra_num])
                #download data
                df = self.get_esios_data(
                    intra_indicator_id,
                    fecha_inicio_carga,
                    fecha_fin_carga
                )
                if not df.empty:
                    dfs.append(df)

        # Combine all DataFrames
        if dfs:
            final_df = pd.concat(dfs, ignore_index=True)
            final_df.sort_values(['fecha', 'hora', 'id_mercado'], inplace=True)
            return final_df
        else:
            return pd.DataFrame()
    
    def save_data(self, df_data: pd.DataFrame, dev: bool):
        """
        Saves intraday market data to the database. Handles granularity change on 2025-03-19:
        - Before 2025-03-19: Saves to Precios_horarios
        - After 2025-03-19: Saves to Precios_quinceminutales
        
        Args:
            df_data (pd.DataFrame): DataFrame containing the intraday market data
            dev (bool): If True, save to the development database
        """
        super().save_data(df_data, dev)

    def get_db_data(self, fecha_inicio_carga: str, fecha_fin_carga: str, intra_ids: list[int]):
        # Example with multiple indicators for Intra markets
        indicator_ids = []
        for intra_id in intra_ids:
            indicator_id = self.get_indicator_id(self.intra_name_map[intra_id])
            indicator_ids.append(indicator_id)

        return super().get_db_data(fecha_inicio_carga, fecha_fin_carga, indicator_ids=indicator_ids)

class Secundaria(DescargadorESIOS):

    def __init__(self):
        super().__init__()
        self.secundaria_name_map = {
            1: "Secundaria a subir",  # indicator 2130
            2: "Secundaria a bajar",  # indicator 634
        }
        self.precio_dual_fecha = datetime.strptime('2024-11-20', '%Y-%m-%d')
        self.cambio_granularidad_fecha = datetime.strptime('2022-05-24', '%Y-%m-%d')
        
    def get_prices(self, fecha_inicio_carga: str, fecha_fin_carga: str, secundaria_lst: list[int]) -> pd.DataFrame:
        """
        Descarga los datos de ESIOS para secundaria, manejando el cambio regulatorio del 20/11/2024
        donde se cambia de precio único (634) a precio dual (634 para bajar, 2130 para subir).

        Args:
            fecha_inicio_carga (str): La fecha de inicio de la carga en formato YYYY-MM-DD
            fecha_fin_carga (str): La fecha de fin de la carga en formato YYYY-MM-DD
            secundaria_lst (list[int]): Lista de tipos de secundaria [1: subir, 2: bajar]

        Returns:
            pd.DataFrame: DataFrame con los precios de secundaria
        """
        # Validate input
        invalid_ids = [id for id in secundaria_lst if id not in self.secundaria_name_map]
        if invalid_ids:
            raise ValueError(f"Invalid secundaria types: {invalid_ids}. Valid types are 1 (subir) and 2 (bajar)")

        # Convert dates to datetime for comparison
        fecha_inicio_dt = datetime.strptime(fecha_inicio_carga, '%Y-%m-%d')
        fecha_fin_dt = datetime.strptime(fecha_fin_carga, '%Y-%m-%d')
        
        dfs = []

        # If date range spans the regulatory change
        if fecha_inicio_dt < self.precio_dual_fecha and fecha_fin_dt > self.precio_dual_fecha:
            # Before change: only use bajar indicator (634) for all data
            df = self.get_esios_data(
                self.get_indicator_id("Secundaria a bajar"),
                fecha_inicio_carga,
                (self.precio_dual_fecha - timedelta(days=1)).strftime('%Y-%m-%d')  # End on 19th
            )
            if not df.empty:
                dfs.append(df)

            # After change: use both indicators if requested
            for sec_id in secundaria_lst:
                df = self.get_esios_data(
                    self.get_indicator_id(self.secundaria_name_map[sec_id]),
                    self.precio_dual_fecha.strftime('%Y-%m-%d'),  # Start from 20th
                    fecha_fin_carga 
                )
                if not df.empty:
                    dfs.append(df)

        # If entirely after regulatory change
        elif fecha_inicio_dt >= self.precio_dual_fecha:
            for sec_id in secundaria_lst:
                df = self.get_esios_data(
                    self.get_indicator_id(self.secundaria_name_map[sec_id]),
                    fecha_inicio_carga,
                    fecha_fin_carga
                )
                if not df.empty:
                    dfs.append(df)

        # If entirely before regulatory change
        else:
            # Only use bajar indicator (634) regardless of what was requested
            df = self.get_esios_data(
                self.get_indicator_id("Secundaria a bajar"),
                fecha_inicio_carga,
                fecha_fin_carga
            )
            if not df.empty:
                dfs.append(df)

        # Combine all DataFrames
        if dfs:
            final_df = pd.concat(dfs, ignore_index=True)
            final_df.sort_values(['fecha', 'hora', 'id_mercado'], inplace=True)
            return final_df
        else:
            return pd.DataFrame()

    def save_data(self, df_data: pd.DataFrame, dev: bool):
        """
        Saves secondary regulation data to the database. Handles granularity change on 2022-05-24:
        - Before 2022-05-24: Saves to Precios_horarios
        - After 2022-05-24: Saves to Precios_quinceminutales
        
        Args:
            df_data (pd.DataFrame): DataFrame containing the secondary regulation data
            dev (bool): If True, save to the development database
        """
        super().save_data(df_data, dev)

    def get_db_prices(self, fecha_inicio: str, fecha_fin: str, secundaria_lst: list[int]):
        """
        Obtiene los datos de la base de datos, teniendo en cuenta el cambio regulatorio.
        """
        fecha_inicio_dt = datetime.strptime(fecha_inicio, '%Y-%m-%d')
        
        indicator_ids = []
        if fecha_inicio_dt < self.precio_dual_fecha:
            # Before change: only use bajar indicator
            indicator_ids.append(self.get_indicator_id("Secundaria a bajar"))
        else:
            # After change: use requested indicators
            for sec_id in secundaria_lst:
                indicator_ids.append(self.get_indicator_id(self.secundaria_name_map[sec_id]))

        return super().get_db_data(fecha_inicio, fecha_fin, indicator_ids=indicator_ids, is_quinceminutal=True)

class Terciaria(DescargadorESIOS):

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
        
    def get_prices(self, fecha_inicio_carga: str, fecha_fin_carga: str, terciaria_lst: list[int]) -> pd.DataFrame:
        """
        Descarga los datos de ESIOS para terciaria, manejando el cambio regulatorio del 10/12/2024
        donde se cambia de precio dual (676 bajar, 677 subir) a precio único (2197).

        Args:
            fecha_inicio_carga (str): La fecha de inicio de la carga en formato YYYY-MM-DD
            fecha_fin_carga (str): La fecha de fin de la carga en formato YYYY-MM-DD
            terciaria_lst (list[int]): Lista de tipos de terciaria [1: subir, 2: bajar, 3: directa subir, 
                                                                   4: directa bajar, 5: programada único]

        Returns:
            pd.DataFrame: DataFrame con los precios de terciaria
        """
        # Validate input
        invalid_ter_nums = [ter_num for ter_num in terciaria_lst if ter_num not in self.terciaria_name_map]
        if invalid_ter_nums:
            raise ValueError(f"Invalid terciaria types: {invalid_ter_nums}. Valid types are 1-5")

        # Convert dates to datetime for comparison
        fecha_inicio_dt = datetime.strptime(fecha_inicio_carga, '%Y-%m-%d')
        fecha_fin_dt = datetime.strptime(fecha_fin_carga, '%Y-%m-%d')
        
        dfs = []
        
        # Separate affected and unaffected markets
        affected_markets = [id for id in terciaria_lst if id in [1, 2, 5]]
        unaffected_markets = [id for id in terciaria_lst if id in [3, 4]]

        #TERCIARIA DIRECTA
        # Handle unaffected markets by regulatory change (terciaria directa) -> download normally
        for ter_num in unaffected_markets:
            #get indicator id
            indicator_id = self.get_indicator_id(self.terciaria_name_map[ter_num])
            #download data
            df = self.get_esios_data(
                indicator_id,
                fecha_inicio_carga,
                fecha_fin_carga
            )
            if not df.empty:
                dfs.append(df)

        #TERCIARIA PROGRAMADA
        # Handle affected markets based on date ranges (terciaria programada)
        if fecha_inicio_dt < self.precio_unico_fecha and fecha_fin_dt >= self.precio_unico_fecha:
            # Period 1: Before change - use dual prices (676, 677)
            for ter_num in [ter_num for ter_num in affected_markets if ter_num in [1, 2]]:
                #get indicator id
                indicator_id = self.get_indicator_id(self.terciaria_name_map[ter_num])
                #download data
                df = self.get_esios_data(
                    indicator_id,
                    fecha_inicio_carga,
                    (self.precio_unico_fecha - timedelta(days=1)).strftime('%Y-%m-%d')
                )
                if not df.empty:
                    dfs.append(df)

            # Period 2: After change - use single price (2197)
            if any(ter_num in affected_markets for ter_num in [1, 2, 5]):
                indicator_id = self.get_indicator_id("Terciaria programada unico")
                df = self.get_esios_data(
                    indicator_id,
                    (self.precio_unico_fecha).strftime('%Y-%m-%d'),
                    fecha_fin_carga
                )
                if not df.empty:
                    dfs.append(df)

        # If entirely after regulatory change
        elif fecha_inicio_dt >= self.precio_unico_fecha:
            if any(ter_num in affected_markets for ter_num in [1, 2, 5]):
                indicator_id = self.get_indicator_id("Terciaria programada unico")
                df = self.get_esios_data(
                    indicator_id,
                    fecha_inicio_carga,
                    fecha_fin_carga
                )
                if not df.empty:
                    dfs.append(df)

        # If entirely before regulatory change
        else:
            for ter_num in [ter_num for ter_num in affected_markets if ter_num in [1, 2]]:
                indicator_id = self.get_indicator_id(self.terciaria_name_map[ter_num])
                df = self.get_esios_data(
                    indicator_id,
                    fecha_inicio_carga,
                    fecha_fin_carga
                )
                if not df.empty:
                    dfs.append(df)

        # Combine all DataFrames
        if dfs:
            final_df = pd.concat(dfs, ignore_index=True)
            final_df.sort_values(['fecha', 'hora', 'id_mercado'], inplace=True)
            return final_df
        else:
            return pd.DataFrame()

    def save_data(self, df_data: pd.DataFrame, dev: bool):
        """
        Saves terciaria data to the database. Handles granularity change on 2022-05-24:
        - Before 2022-05-24: Saves to Precios_horarios
        - After 2022-05-24: Saves to Precios_quinceminutales
        
        Args:
            df_data (pd.DataFrame): DataFrame containing the terciaria data
            dev (bool): If True, save to the development database
        """
        super().save_data(df_data, dev)

    def get_db_prices(self, fecha_inicio: str, fecha_fin: str, terciaria_lst: list[int]):
        """
        Obtiene los datos de la base de datos, teniendo en cuenta el cambio regulatorio.
        """
        fecha_inicio_dt = datetime.strptime(fecha_inicio, '%Y-%m-%d')
        
        indicator_ids = []
        
        # Always include unaffected markets
        for ter_num in [ter_num for ter_num in terciaria_lst if ter_num in [3, 4]]:
            indicator_ids.append(self.get_indicator_id(self.terciaria_name_map[ter_num]))

        # Handle affected markets based on date
        if fecha_inicio_dt < self.precio_unico_fecha:
            # Before change: use dual prices
            for ter_num in [ter_num for ter_num in terciaria_lst if ter_num in [1, 2]]:
                indicator_ids.append(self.get_indicator_id(self.terciaria_name_map[ter_num]))
        else:
            # After change: use single price
            if any(ter_num in terciaria_lst for ter_num in [1, 2, 5]):
                indicator_ids.append(self.get_indicator_id("Terciaria programada unico"))

        return super().get_db_data(fecha_inicio, fecha_fin, indicator_ids=indicator_ids, is_quinceminutal=True)

class RR(DescargadorESIOS):

    def __init__(self):
        super().__init__()
        self.indicator_id = self.get_indicator_id("RR a subir") #precio unico que se guarda en el indicador de RR a subir
        self.cambio_granularidad_fecha = datetime.strptime('2022-05-24', '%Y-%m-%d')

    def get_rr_data(self, fecha_inicio_carga, fecha_fin_carga):
        """
        Descarga los datos de ESIOS para un indicador específico en un rango de fechas. Para RR (precio único) y se guarda en el indicador de RR a subir.
        """
        return self.get_esios_data(self.indicator_id, fecha_inicio_carga, fecha_fin_carga)
        
    def save_data(self, df_data: pd.DataFrame, dev: bool):
        """
        Saves RR (Replacement Reserve) data to the database. Handles granularity change on 2022-05-24:
        - Before 2022-05-24: Saves to Precios_horarios
        - After 2022-05-24: Saves to Precios_quinceminutales
        
        The table selection can be overridden by providing a table_name parameter.

        Args:
            df_data (pd.DataFrame): DataFrame containing the RR market data. 
            table_name (str, optional): Override the default table selection logic and save to this table instead.
                                      Must be either 'Precios_horarios' or 'Precios_quinceminutales'.

        Note:
            The granularity change date (2022-05-24) is defined in self.cambio_granularidad_fecha
        """
        super().save_data(df_data, dev)

    def get_db_data(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None):
        return super().get_db_data(fecha_inicio_carga, fecha_fin_carga, indicator_ids=[self.indicator_id])

def test_usage_download():
    diario = Diario()
    diario_data = diario.get_prices(fecha_inicio_carga='2024-06-10', fecha_fin_carga='2024-06-16')
    print(diario_data)
    breakpoint()

    intra = Intra() 
    #downloading data for all intras between regulatory change (intra reduction) 
    intra_data = intra.get_prices(fecha_inicio_carga='2024-06-10', fecha_fin_carga='2024-06-16', intra_lst=[1,2,3,4,5,6,7])
    print(intra_data)
    breakpoint()
    sec = Secundaria()
    #downloading data for both secondary markets between regulatory change
    sec_data = sec.get_prices(fecha_inicio_carga='2024-11-18', fecha_fin_carga='2024-11-22', secundaria_lst=[1,2])
    print(sec_data)
    breakpoint()

    ter = Terciaria()
    #downloading data for all terciarias between regulatory change (precio unificado terciaria)
    ter_data = ter.get_prices(fecha_inicio_carga='2024-12-08', fecha_fin_carga='2024-12-22', terciaria_lst=[1,2,3,4,5])
    print(ter_data)
    breakpoint()

    rr = RR()
    #downloading data for all rr (precio unico rr)
    rr_data = rr.get_rr_data(fecha_inicio_carga='2024-12-08', fecha_fin_carga='2024-12-22')
    print(rr_data)
    breakpoint()

def test_usage_save():
    diario = Diario()
    diario_data = diario.get_prices(fecha_inicio_carga='2024-07-10', fecha_fin_carga='2024-07-11')
    diario.save_data(diario_data, dev=True)
    breakpoint()

    diario_data2 = diario.get_prices(fecha_inicio_carga='2024-07-13', fecha_fin_carga='2024-07-16')
    diario.save_data(diario_data2, dev=True)
    breakpoint()

    intra = Intra()
    intra_data = intra.get_prices(fecha_inicio_carga='2024-07-10', fecha_fin_carga='2024-07-16', intra_lst=[1,2,3,4,5,6,7])
    intra.save_data(intra_data, dev=True)
    breakpoint()

    sec = Secundaria()
    sec_data1= sec.get_prices(fecha_inicio_carga='2024-12-18', fecha_fin_carga='2024-12-22', secundaria_lst=[1,2])
    sec.save_data(sec_data1, dev=True)
    breakpoint()

    sec_data2= sec.get_prices(fecha_inicio_carga='2022-05-18', fecha_fin_carga='2022-05-28', secundaria_lst=[1,2])
    sec.save_data(sec_data2, dev=True)
    breakpoint()

    ter = Terciaria()
    ter_data = ter.get_prices(fecha_inicio_carga='2024-12-08', fecha_fin_carga='2024-12-22', terciaria_lst=[1,2,3,4,5])
    ter.save_data(ter_data, dev=True)
    breakpoint()

    rr = RR()
    rr_data = rr.get_rr_data(fecha_inicio_carga='2022-05-22', fecha_fin_carga='2022-05-28')
    rr.save_data(rr_data, dev=True)
    breakpoint()
    
def main():
    # diario = Diario()
    # diario_data = diario.get_prices(fecha_inicio_carga='2024-06-10', fecha_fin_carga='2024-06-16')
    # print(diario_data)
    # expected_rows = 24 * (datetime.strptime('2024-06-16', '%Y-%m-%d') - datetime.strptime('2024-06-10', '%Y-%m-%d')).days
    # assert len(diario_data) == expected_rows, f"Diario data should have {expected_rows} rows"


    # intra = Intra()
    # # Before regulatory change - will use all intras 1-7
    # df1 = intra.get_prices('2024-05-01', '2024-06-01', [1,2,3,4,5,6,7])
    # # After regulatory change - will only use intras 1-3
    # df2 = intra.get_prices('2024-07-01', '2024-07-31', [1,2,3,4,5,6,7])
    # #Spanning regulatory change - will use all before change and only 1-3 after
    # df3 = intra.get_prices('2024-06-10', '2024-06-16', [1,2,3,4,5,6,7])
    # #Print unique id_mercado before and after regulatory change
    # print(df1['id_mercado'].unique())
    # print(df2['id_mercado'].unique()) 
    # print(f"after", df3[df3['fecha'] > intra.intra_reduction_date]['id_mercado'].unique())
    # print(f"before", df3[df3['fecha'] <= intra.intra_reduction_date]['id_mercado'].unique())

    # # sec = Secundaria()
    # # Before regulatory change - will only use indicator 634
    # df1 = sec.get_prices('2024-10-01', '2024-11-01', [1, 2])
    # # After regulatory change - will use both indicators
    # df2 = sec.get_prices('2024-12-01', '2024-12-31', [1, 2])
    # # Spanning regulatory change - will use 634 before change and both after
    # df3 = sec.get_prices('2024-11-01', '2024-12-01', [1, 2])
    # #print unique id_mercado before and after regulatory change
    # print(df1['id_mercado'].unique(), df1.head())
    # print(df2['id_mercado'].unique(), df2.head())
    # print(df3['id_mercado'].unique(), df3.head()) 

    # # Test Terciaria functionality
    # ter = Terciaria()
    # # Test 1: Basic functionality - check if data is returned and has expected columns
    # ter_data = ter.get_prices(fecha_inicio_carga='2024-12-08', fecha_fin_carga='2024-12-20', terciaria_lst=[1,2,3,4,5])
    # assert isinstance(ter_data, pd.DataFrame), "get_prices should return a DataFrame"
    # assert all(col in ter_data.columns for col in ['fecha', 'hora', 'precio', 'id_mercado']), "DataFrame missing expected columns"
    
    # # Test 2: Date range validation
    # try:
    #     ter_data = ter.get_prices(fecha_inicio_carga='2024-12-20', fecha_fin_carga='2024-12-08', terciaria_lst=[1])
    #     assert False, "Should raise error when start date is after end date"
    # except ValueError:
    #     print("✓ Date validation working correctly")
        
    # # Test 3: Check terciaria_lst filtering for range spanning regulatory change
    # test_lst = [1,2]
    # ter_data = ter.get_prices(fecha_inicio_carga='2024-12-08', fecha_fin_carga='2024-12-20', terciaria_lst=test_lst)
    # print(ter_data.head(68))
    # assert 5 not in ter_data['id_mercado'].unique(), "Download logic is not working for range spanning regulatory change"
    # print("All Terciaria tests passed successfully!")

    # rr = RR()
    # rr_data = rr.get_rr_data(fecha_inicio_carga='2024-12-08', fecha_fin_carga='2024-12-20')
    # print(rr_data)

    ter = Terciaria()
    ter_data = ter.get_prices(fecha_inicio_carga='2024-05-20', fecha_fin_carga='2024-05-30', terciaria_lst=[1])
    ter_data.sort_values(by=['fecha', 'hora', 'id_mercado'], inplace=True)
    print(ter_data.head(100))
    
if __name__ == "__main__":
    #main()
    #test_usage_download()
    test_usage_save()

