import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional
import sys
import os
import pretty_errors
from dotenv import load_dotenv
load_dotenv()
import pytz
from pathlib import Path
from deprecated import deprecated
# Get the absolute path to the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
# Use absolute imports
from utilidades.db_utils import DatabaseUtils
from configs.esios_config import DiarioConfig, IntraConfig, SecundariaConfig, TerciariaConfig, RRConfig

class DescargadorESIOS:
    """
    Clase base para descargar datos de ESIOS.
    """
    def __init__(self):
        self.esios_token = os.getenv('ESIOS_TOKEN')
        self.download_window = 93
        self.madrid_tz = pytz.timezone('Europe/Madrid')
        
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
        if hasattr(self, 'cambio_granularidad_fecha'): #if the class has a cambio de granularidad, we need to handle it
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
                    dfs.append(df_before)
                
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
            pd.DataFrame: A DataFrame containing the requested data from the ESIOS API with an extra column for "granularidad".
                     Returns an empty DataFrame if no data is found for the period.

        Raises:
            ValueError: If the start date is greater than the end date.
            Exception: If there is an error during the API call or data validation process. 
        """
        # Convert string dates to Madrid local datetime (tz aware) start of day 00:00:00
        start_local = self.madrid_tz.localize(datetime.strptime(fecha_inicio_carga, '%Y-%m-%d'))

        # For end date, we want the end of the day (23:55:00 -> max esios time in a day 23:55:00)
        end_local = self.madrid_tz.localize(datetime.strptime(fecha_fin_carga, '%Y-%m-%d').replace(hour=23, minute=55, second=0))

        # Convert to UTC for API request
        start_utc = start_local.astimezone(pytz.UTC)
        end_utc = end_local.astimezone(pytz.UTC)

        try:
            # Download ESIOS data for each market of interest
            url = f"http://api.esios.ree.es/indicators/{indicator_id}"
            params = {
                'start_date': start_utc.strftime('%Y-%m-%dT%H:%M:%SZ'), #convert to string format in UTC
                'end_date': end_utc.strftime('%Y-%m-%dT%H:%M:%SZ') #convert to string format in UTC
            }
            headers = {
                'x-api-key': self.esios_token,
                'Authorization': f'Token token={self.esios_token}'
            }
            
            # Log request information
            print(f"Realizando solicitud API para indicador {indicator_id} desde {fecha_inicio_carga} hasta {fecha_fin_carga}")
            
            # Request timeout handling (10 seconds connection, 30 seconds read)
            response = requests.get(url, headers=headers, params=params, timeout=(20, 40))
            
            # Check HTTP status code
            if response.status_code != 200:
                if response.status_code == 401:
                    raise ValueError(f"Error de autenticación (401): Token ESIOS no válido o expirado")
                elif response.status_code == 403:
                    raise ValueError(f"Error de permisos (403): No tiene acceso al indicador {indicator_id}")
                elif response.status_code == 404:
                    raise ValueError(f"Indicador no encontrado (404): El indicador {indicator_id} no existe en ESIOS")
                elif response.status_code == 429:
                    raise ValueError(f"Demasiadas solicitudes (429): Se ha excedido el límite de solicitudes a la API de ESIOS")
                elif 500 <= response.status_code < 600:
                    raise ConnectionError(f"Error del servidor ESIOS ({response.status_code}): Intente nuevamente más tarde")
                else:
                    raise ValueError(f"Error HTTP {response.status_code}: {response.text}")
            
            # Try to parse JSON response
            try:
                data = response.json()
            except ValueError as e:
                raise ValueError(f"Error al parsear la respuesta JSON: {e}. Contenido: {response.text[:200]}...")
            
            #extract granularity from json data -> can be "Quince minutos" or "Hora"
            granularidad = data["indicator"]["tiempo"][0]["name"].strip()

            #validate data structure
            if not self.validate_data_structure(data, granularidad):
                return pd.DataFrame() #return empty dataframe if no data found
            
            #if no errors, procesamos los datos de interés en un dataframe que se ecnuentran en el key values
            df_data = pd.DataFrame(data['indicator']['values'])
            df_data['granularidad'] = granularidad

            return df_data

        except requests.exceptions.ConnectTimeout:
            raise ConnectionError(f"Tiempo de conexión agotado al intentar conectar con la API de ESIOS. Verifique su conexión a Internet.")
        except requests.exceptions.ReadTimeout:
            raise ConnectionError(f"Tiempo de espera agotado al leer datos de la API de ESIOS. El servidor puede estar sobrecargado.")
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Error de conexión con la API de ESIOS: {e}. Verifique su conexión a Internet.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error en la solicitud HTTP a la API de ESIOS: {e}")
        except Exception as e:
            raise Exception(f"Error inesperado al descargar los datos de ESIOS para el indicador {indicator_id}: {e}")

    def validate_data_structure(self, data: dict, granularidad: str) -> bool:
        """
        Validate the structure of the data returned by the ESIOS API.
        
        Args:
            data (dict): The data returned from the ESIOS API.

        Returns:
            bool: True if the data structure is valid, False if no data found but structure is valid,
                  raises ValueError if structure is invalid
        """
        try:
            # Check if data has the expected structure
            if 'indicator' not in data or 'values' not in data['indicator']:
                raise ValueError(f"Formato de respuesta inesperado: {data.keys()}")

            else:
                # Extract indicator name and ID for logging
                indicator_name = data['indicator'].get('name', self.__class__.__name__)

            # If values list is empty, return False instead of raising an error
            if not data['indicator']['values']:
                print(f"No se encontraron datos para el indicador: {indicator_name},  en el período solicitado")
                return False #handled as an empty dataframe return in wrapper function get precios
            
            #validate granularidad
            if granularidad != "Quince minutos" and granularidad != "Hora":
                raise ValueError(f"Granularidad inesperada: {granularidad}")
            
            print(f"Validación de datos exitosa para el indicador: {indicator_name}")
            
        except Exception as e:
            raise ValueError(f"Error al validar la estructura de los datos: {e}")
        
        # Return True if all checks pass and data exists
        return True

    @deprecated(action="error", reason="Method used in old ETL pipeline, now deprecated")
    def save_data_to_db(self, df_data: pd.DataFrame, dev: bool, table_name: str = None):
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

class DiarioPreciosDL(DescargadorESIOS):

    def __init__(self):
        super().__init__() 

        #get config from esios config file 
        self.config = DiarioConfig()
        self.indicator_id = self.config.indicator_id

    def get_prices(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None):
        return self.get_esios_data(self.indicator_id, fecha_inicio_carga, fecha_fin_carga)

class IntraPreciosDL(DescargadorESIOS):

    def __init__(self):
        super().__init__()

        #get config from esios config file 
        self.config = IntraConfig()
        self.intra_name_map = self.config.intra_name_map #map of intra numbers to names
        self.intra_reduccion_fecha = self.config.intra_reduccion_fecha #date of regulatory change for intras
        self.cambio_granularidad_fecha = self.config.cambio_granularidad_fecha #date of granularidad change for intras
        
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

        # CASE 1: If date range spans the regulatory change
        if fecha_inicio_dt < self.intra_reduccion_fecha and fecha_fin_dt > self.intra_reduccion_fecha:
            # Split into two periods: before and after regulatory change
            
            # Period 1: Before regulatory change - download all requested intras up to the intra reduction date
            for intra_num in intra_lst:
                intra_indicator_id = self.config.get_indicator_id(self.intra_name_map[intra_num])
                df = self.get_esios_data(
                    intra_indicator_id,
                    fecha_inicio_carga,
                    (self.intra_reduccion_fecha.strftime('%Y-%m-%d')) #en fecha reduccion intra aun no aplica el cambio
                )
                if not df.empty:
                    dfs.append(df)

            # Period 2: After regulatory change - download only Intras 1-3 after the intra reduction date
            intras_after_change = [intra_num for intra_num in intra_lst if intra_num <= 3]

            #download all valid intras after regulatory change
            for intra_num in intras_after_change: #only 1-3 are valid after regulatory change
                intra_indicator_id = self.config.get_indicator_id(self.intra_name_map[intra_num])
                df = self.get_esios_data(
                    intra_indicator_id,
                    (self.intra_reduccion_fecha).strftime('%Y-%m-%d'),
                    fecha_fin_carga
                )
                if not df.empty:
                    dfs.append(df)

        # CASE 2: If date range entirely after regulatory change
        elif fecha_inicio_dt >= self.intra_reduccion_fecha:
            intras_after_change = [intra_num for intra_num in intra_lst if intra_num <= 3]

            #download all intras after regulatory change
            for intra_num in intras_after_change:
                intra_indicator_id = self.config.get_indicator_id(self.intra_name_map[intra_num])
                df = self.get_esios_data(
                    intra_indicator_id,
                    fecha_inicio_carga,
                    fecha_fin_carga
                )
                if not df.empty:
                    dfs.append(df)

        # CASE 3: If date range entirely before regulatory change
        else:
            #download all intras before regulatory change
            for intra_num in intra_lst:
                #get indicador id for each intra
                intra_indicator_id = self.config.get_indicator_id(self.intra_name_map[intra_num])
                #download data
                df = self.get_esios_data(
                    intra_indicator_id,
                    fecha_inicio_carga,
                    fecha_fin_carga
                )
                if not df.empty:
                    dfs.append(df)

        # Combine all DataFrames if necessary
        if dfs:
            final_df = pd.concat(dfs, ignore_index=True)
            return final_df
        else:
            print(f"No data found for intras {intra_lst} in the date range {fecha_inicio_carga} to {fecha_fin_carga}")
            return pd.DataFrame()
    
class SecundariaPreciosDL(DescargadorESIOS):

    def __init__(self):
        super().__init__()
        self.config = SecundariaConfig()
        self.secundaria_name_map = self.config.secundaria_name_map
        self.precio_dual_fecha = self.config.precio_dual_fecha
        self.cambio_granularidad_fecha = self.config.cambio_granularidad_fecha
        
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
                self.config.get_indicator_id("Secundaria a bajar"),
                fecha_inicio_carga,
                (self.precio_dual_fecha - timedelta(days=1)).strftime('%Y-%m-%d')  # End on 19th
            )
            if not df.empty:
                dfs.append(df)

            # After change: use both indicators if requested
            for sec_id in secundaria_lst:
                df = self.get_esios_data(
                    self.config.get_indicator_id(self.secundaria_name_map[sec_id]),
                    self.precio_dual_fecha.strftime('%Y-%m-%d'),  # Start from 20th
                    fecha_fin_carga 
                )
                if not df.empty:
                    dfs.append(df)

        # If entirely after regulatory change
        elif fecha_inicio_dt >= self.precio_dual_fecha:
            for sec_id in secundaria_lst:
                df = self.get_esios_data(
                    self.config.get_indicator_id(self.secundaria_name_map[sec_id]),
                    fecha_inicio_carga,
                    fecha_fin_carga
                )
                if not df.empty:
                    dfs.append(df)

        # If entirely before regulatory change
        else:
            # Only use bajar indicator (634) regardless of what was requested
            df = self.get_esios_data(
                self.config.get_indicator_id("Secundaria a bajar"),
                fecha_inicio_carga,
                fecha_fin_carga
            )
            if not df.empty:
                dfs.append(df)

        # Combine all DataFrames
        if dfs:
            final_df = pd.concat(dfs, ignore_index=True)
            return final_df
        else:
            return pd.DataFrame()

class TerciariaPreciosDL(DescargadorESIOS):

    def __init__(self):
        super().__init__()

        #get config from esios config file 
        self.config = TerciariaConfig()
        self.terciaria_name_map = self.config.terciaria_name_map
        self.precio_unico_fecha = self.config.precio_unico_fecha
        self.cambio_granularidad_fecha = self.config.cambio_granularidad_fecha
        
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
            indicator_id = self.config.get_indicator_id(self.terciaria_name_map[ter_num])
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
                indicator_id = self.config.get_indicator_id(self.terciaria_name_map[ter_num])
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
                indicator_id = self.config.get_indicator_id("Terciaria programada unico")
                #download data
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
                indicator_id = self.config.get_indicator_id("Terciaria programada unico")
                
                #download data
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
                indicator_id = self.config.get_indicator_id(self.terciaria_name_map[ter_num])
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
            return final_df
        else:
            return pd.DataFrame()

class RRPreciosDL(DescargadorESIOS):
    """
    Clase para descargar datos de ESIOS para RR (precio único).
    """
    def __init__(self):
        super().__init__()

        #get config from esios config file 
        self.config = RRConfig()
        self.indicator_id = self.config.indicator_id
        self.cambio_granularidad_fecha = self.config.cambio_granularidad_fecha

    def get_prices(self, fecha_inicio_carga, fecha_fin_carga):
        """
        Descarga los datos de ESIOS para un indicador específico en un rango de fechas. Para RR (precio único) y se guarda en el indicador de RR a subir.
        """
        return self.get_esios_data(self.indicator_id, fecha_inicio_carga, fecha_fin_carga)


if __name__ == "__main__":
    #main()
    descargadores = [DiarioPreciosDL(), IntraPreciosDL(), SecundariaPreciosDL(), 
                     TerciariaPreciosDL(), RRPreciosDL()]
    #test_usage_save()
