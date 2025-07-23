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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utilidades.db_utils import DatabaseUtils
from configs.esios_config import DiarioConfig, IntraConfig, SecundariaConfig, TerciariaConfig, RRConfig
from utilidades.proxy_utils import ProxyManager
import requests.exceptions

class DescargadorESIOS:
    """
    Clase base para descargar datos de ESIOS.
    """
    def __init__(self):

        #get esios token from environment variable
        self.esios_token = os.getenv('ESIOS_TOKEN')
        if not self.esios_token:
            raise ValueError("ESIOS_TOKEN environment variable not set.")
        
        #download window in days
        self.download_window = 93

        #madrid timezone
        self.madrid_tz = pytz.timezone('Europe/Madrid')

        # Set proxy usage directly to False
        self.use_proxies = False

        # Conditionally initialize proxy manager (used in make esios request)
        if self.use_proxies:
            self.proxy_manager = ProxyManager()
        else:
            self.proxy_manager = None

    def get_esios_data(self, indicator_id: str, fecha_inicio: str = None, fecha_fin: str = None) -> pd.DataFrame:
        """
        Downloads data from ESIOS for a specific indicator within a date range.
        
        Args:
            indicator_id (str): The indicator ID for which data is being requested
            fecha_inicio (str): Start date for data request in 'YYYY-MM-DD' format
            fecha_fin (str): End date for data request in 'YYYY-MM-DD' format
        
        Returns:
            pd.DataFrame: DataFrame containing the requested data with granularity information
        """
        return self._make_esios_request(indicator_id, fecha_inicio, fecha_fin)

    def _make_esios_request(self, indicator_id: str, fecha_inicio: str, fecha_fin: str) -> pd.DataFrame:
        """
        Internal method that handles the actual API call and data processing.
        Can now run with or without proxies based on self.use_proxies.

        Args:
            indicator_id (str): The ID of the indicator for which data is being requested.
            fecha_inicio (str): The start date for the data request in 'YYYY-MM-DD' format.
            fecha_fin (str): The end date for the data request in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: A DataFrame containing the requested data from the ESIOS API with an extra column for "granularidad".
                     Returns an empty DataFrame if no data is found for the period.

        Raises:
            ValueError: If the start date is greater than the end date.
            Exception: If there is an error during the API call or data validation process. 
        """
        # Convert string dates to Madrid local datetime (tz aware) start of day 00:00:00
        start_local = self.madrid_tz.localize(datetime.strptime(fecha_inicio, '%Y-%m-%d'))

        # For end date, we want the end of the day (23:55:00 -> max esios time in a day 23:55:00)
        end_local = self.madrid_tz.localize(datetime.strptime(fecha_fin, '%Y-%m-%d').replace(hour=23, minute=55, second=0))

        # Convert to end time and start time to UTC for API request
        start_utc = start_local.astimezone(pytz.UTC)
        end_utc = end_local.astimezone(pytz.UTC)

        url = f"https://api.esios.ree.es/indicators/{indicator_id}"
        params = {
            'start_date': start_utc.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'end_date': end_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
        }
        headers = {
            'Accept': 'application/json; application/vnd.esios-api-v2+json',
            'Content-Type': 'application/json',
            'Host': 'api.esios.ree.es',
            'x-api-key': self.esios_token,
        }

        if self.use_proxies:
            # --- Proxy Logic ---
            # Maximum number of proxy attempts before giving up
            max_attempts = 10 
            attempt = 0
            last_exception = None

            while attempt < max_attempts:
                # Ensure proxy manager is initialized
                if not self.proxy_manager:
                    raise Exception("Proxy Manager not initialized, but use_proxies is True.")

                # Get the next available proxy from the rotation
                proxy = self.proxy_manager.get_next_proxy()
                if not proxy:
                    print("No available proxies left to try.")
                    break
                
                # Format the proxy for requests library
                proxies = self.proxy_manager.format_proxy(proxy)
                
                try:
                    # Log the current attempt and proxy being used
                    print(f"Attempt {attempt + 1}/{max_attempts}: API request for indicator {indicator_id} via proxy {proxies.get('http') or proxies.get('https')}")
                    
                    # Make the actual API request with the current proxy
                    # Timeout: (connect timeout, read timeout) in seconds
                    response = requests.get(url, headers=headers, params=params, timeout=(20, 40), proxies=proxies)

                    # Check HTTP status code for proxy-specific issues first
                    if response.status_code in [403, 407]:
                        # 403: Forbidden, 407: Proxy Authentication Required
                        print(f"Proxy {proxies.get('http') or proxies.get('https') } failed with status {response.status_code}, blacklisting and retrying...")
                        self.proxy_manager.mark_bad_proxy(proxy)
                        attempt += 1
                        last_exception = requests.exceptions.RequestException(f"Proxy error: {response.status_code}")
                        continue
                    # Check for other errors after proxy check
                    elif response.status_code != 200:
                        self._handle_response_error(response, indicator_id)

                    # Success path (process data)
                    return self._process_response(response, indicator_id)

                except (requests.exceptions.ProxyError, requests.exceptions.ConnectTimeout, 
                        requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                    # Handle proxy-specific connection errors
                    print(f"Proxy {proxies.get('http') or proxies.get('https') } connection failed: {e}, blacklisting and retrying...")
                    self.proxy_manager.mark_bad_proxy(proxy)
                    last_exception = e
                    attempt += 1
                    continue
                except Exception as e:
                    # For other exceptions, raise immediately
                    raise Exception(f"Unexpected error during proxied ESIOS request for indicator {indicator_id}: {e}")

            # If loop finishes without success (max attempts reached)
            raise Exception(f"All proxy attempts failed for indicator {indicator_id}. Last error: {last_exception}")

        else:
            # --- Direct Logic (No Proxies) ---
            try:
                print(f"Direct API request for indicator {indicator_id} from {fecha_inicio} to {fecha_fin}")
                response = requests.get(url, headers=headers, params=params, timeout=(20, 40))

                if response.status_code != 200:
                    self._handle_response_error(response, indicator_id)

                # Success path (process data)
                return self._process_response(response, indicator_id)

            except requests.exceptions.RequestException as e:
                raise Exception(f"Direct ESIOS request failed for indicator {indicator_id}: {e}")
            except Exception as e:
                raise Exception(f"Unexpected error during direct ESIOS request for indicator {indicator_id}: {e}")

    def _handle_response_error(self, response: requests.Response, indicator_id: str):
        """
        Raises specific exceptions based on the HTTP status code of an ESIOS API response.
        
        Parameters:
            response (requests.Response): The HTTP response object from the ESIOS API.
            indicator_id (str): The ESIOS indicator ID relevant to the request.
        
        Raises:
            ValueError: For authentication errors, forbidden access, missing indicators, or other HTTP errors.
            ConnectionError: For rate limiting or server-side errors.
        """
        if response.status_code == 401:
            raise ValueError(f"Authentication error (401): Invalid or expired ESIOS_TOKEN.")
        elif response.status_code == 403:
            raise ValueError(f"Forbidden (403): Check API token permissions or IP restrictions. Response: {response.text[:200]}")
        elif response.status_code == 404:
            raise ValueError(f"Indicator not found (404): Indicator {indicator_id} does not exist.")
        elif response.status_code == 429:
            raise ConnectionError(f"Too Many Requests (429): Rate limit exceeded.")
        elif 500 <= response.status_code < 600:
            raise ConnectionError(f"ESIOS Server Error ({response.status_code}): Try again later. Response: {response.text[:200]}")
        else:
            raise ValueError(f"HTTP Error {response.status_code}: {response.text[:200]}")

    def _process_response(self, response: requests.Response, indicator_id: str) -> pd.DataFrame:
        """Helper method to parse JSON and process the data. Extracts granularity and validates data structure.
        Args:
            response (requests.Response): The response from the ESIOS API.
            indicator_id (str): The ID of the indicator for which data is being requested.

        Returns:
            pd.DataFrame: A DataFrame containing the requested data from the ESIOS API with an extra column for "granularidad".
        """
        try:
            data = response.json()
        except ValueError as e:
            raise ValueError(f"Failed to parse JSON response: {e}. Content: {response.text[:200]}...")

        # Extract granularity
        indicator_data = data.get('indicator', {})
        time_info = indicator_data.get('tiempo', [])
        granularidad = "Unknown" # Default
        if time_info and isinstance(time_info, list) and len(time_info) > 0 and 'name' in time_info[0]: #defensive checks
            granularidad = time_info[0]['name'].strip()
        else:
            print(f"Warning: Could not determine granularity for indicator {indicator_id}. Response structure: {data.keys()}")

        if not self.validate_data_structure(data, granularidad):
            # validate_data_structure prints message if no data found
            return pd.DataFrame()

        df_data = pd.DataFrame(indicator_data.get('values', []))
        if df_data.empty:
            print(f"No values found for indicator {indicator_id} in the response.")
            return pd.DataFrame()

        df_data['granularidad'] = granularidad
        df_data["indicador_id"] = indicator_id

        return df_data

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
            if 'indicator' not in data or 'values' not in data.get('indicator', {}):
                # Allow empty 'values' list, but 'indicator' key must exist
                indicator_present = 'indicator' in data
                values_present = isinstance(data.get('indicator'), dict) and 'values' in data['indicator']
                if not indicator_present or not values_present :
                    raise ValueError(f"Unexpected response format. Keys: {data.keys()}. Indicator content: {data.get('indicator', 'Not Found')}")

            indicator_name = data.get('indicator', {}).get('name', self.__class__.__name__)
            values = data.get('indicator', {}).get('values', [])

            # If values list is None or empty, return False (no data)
            if values is None or len(values) == 0 :
                print(f"No data values found for indicator: {indicator_name} in the requested period.")
                return False # Correctly indicates no data, not invalid structure

            # Validate granularity if known
            if granularidad not in ["Unknown", "Quince minutos", "Hora"]:
                print(f"Warning: Unexpected granularity '{granularidad}' for indicator {indicator_name}")

        except Exception as e:
            # Raise only if the structure is truly invalid, not just missing values
            raise ValueError(f"Error validating data structure: {e}. Data snippet: {str(data)[:200]}")

        # Return True if structure seems ok and data values exist
        return True

class DiarioPreciosDL(DescargadorESIOS):

    def __init__(self):
        super().__init__() 
        #get config from esios config file 
        self.config = DiarioConfig()
        self.indicator_id = self.config.indicator_id

    def get_prices(self, fecha_inicio: Optional[str] = None, fecha_fin: Optional[str] = None):
        return self.get_esios_data(self.indicator_id, fecha_inicio, fecha_fin)

class IntraPreciosDL(DescargadorESIOS):

    def __init__(self):
        super().__init__()

        #get config from esios config file 
        self.config = IntraConfig()
        self.intra_name_map = self.config.intra_name_map #map of intra numbers to names
        self.intra_reduccion_fecha = self.config.intra_reduccion_fecha #date of regulatory change for intras
        self.cambio_granularidad_fecha = self.config.cambio_granularidad_fecha #date of granularidad change for intras
        
    def get_prices(self, fecha_inicio: str, fecha_fin: str, intra_lst: list[int]) -> pd.DataFrame:
        """
        Descarga los datos de ESIOS para los mercados intradiarios especificados.
        Automatically handles regulatory changes by downloading only available intras after cutoff dates.

        Args:
            fecha_inicio (str): La fecha de inicio de la carga en formato YYYY-MM-DD
            fecha_fin (str): La fecha de fin de la carga en formato YYYY-MM-DD
            intra_ids (list[int]): Lista de IDs de mercados intradiarios (1-7)

        Returns:
            pd.DataFrame: DataFrame con los precios de los mercados intradiarios solicitados
        """
    
        # Validate function input for intra ids
        invalid_intra_nums = [intra_num for intra_num in intra_lst if intra_num not in self.intra_name_map]
        if invalid_intra_nums:
            raise ValueError(f"Invalid intra markets: {invalid_intra_nums}. Valid intra markets are 1-7")

        # Convert dates to datetime for comparison
        fecha_inicio_dt = datetime.strptime(fecha_inicio, '%Y-%m-%d')
        fecha_fin_dt = datetime.strptime(fecha_fin, '%Y-%m-%d')
        
        dfs = [] #list of dataframes to concatenate

        # CASE 1: If date range spans the regulatory change
        if fecha_inicio_dt < self.intra_reduccion_fecha and fecha_fin_dt > self.intra_reduccion_fecha:
            # Split into two periods: before and after regulatory change
            
            # Period 1: Before regulatory change - download all requested intras up to the intra reduction date
            for intra_num in intra_lst:
                intra_indicator_id = self.config.get_indicator_id(self.intra_name_map[intra_num])
                df = self.get_esios_data(
                    intra_indicator_id,
                    fecha_inicio,
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
                    fecha_fin
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
                    fecha_inicio,
                    fecha_fin
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
                    fecha_inicio,
                    fecha_fin
                )
                if not df.empty:
                    dfs.append(df)

        # Combine all DataFrames if necessary
        if dfs:
            final_df = pd.concat(dfs, ignore_index=True)
            return final_df
        else:
            print(f"No data found for intras {intra_lst} in the date range {fecha_inicio} to {fecha_fin}")
            return pd.DataFrame()
    
class SecundariaPreciosDL(DescargadorESIOS):

    def __init__(self):
        super().__init__()
        self.config = SecundariaConfig()
        self.secundaria_name_map = self.config.secundaria_name_map
        self.precio_dual_fecha = self.config.precio_dual_fecha
        self.cambio_granularidad_fecha = self.config.cambio_granularidad_fecha
        
    def get_prices(self, fecha_inicio: str, fecha_fin: str, secundaria_lst: list[int]) -> pd.DataFrame:
        """
        Downloads ESIOS data for secundaria for a specific day.
        Handles the regulatory change on 20/11/2024 where it changes from single price (634) 
        to dual price (634 for down, 2130 for up).
        
        Args:
            fecha_inicio (str): Load date in YYYY-MM-DD format (assumed same as fecha_fin)
            fecha_fin (str): Load date in YYYY-MM-DD format (assumed same as fecha_inicio)
            secundaria_lst (list[int]): List of secundaria types [1: up, 2: down]
        
        Returns:
            pd.DataFrame: DataFrame with secundaria prices for the requested day
        """
        # Validate input types
        invalid_ids = [id for id in secundaria_lst if id not in self.secundaria_name_map]
        if invalid_ids:
            raise ValueError(f"Invalid secundaria types: {invalid_ids}. Valid types are 1 (subir) and 2 (bajar)")
        
         # Ensure we're downloading data for a single day
        if fecha_inicio != fecha_fin:
            print(f"Warning: fecha_inicio ({fecha_inicio}) and fecha_fin ({fecha_fin}) differ. " 
                  f"Using fecha_inicio as the target date, only data range of one day will be downloaded.")
        
        # Use only the start date for simplicity
        target_date = fecha_inicio  # Only hours will vary between dates
        target_date_dt = datetime.strptime(target_date, '%Y-%m-%d')
        
        dfs = []
        
        # If date is on or after the regulatory change - use dual pricing
        if target_date_dt >= self.precio_dual_fecha:
            for sec_id in secundaria_lst:
                df = self.get_esios_data(
                    self.config.get_indicator_id(self.secundaria_name_map[sec_id]),
                    target_date,
                    target_date
                )
                if not df.empty:
                    dfs.append(df)
        
        # If date is before the regulatory change - use single price (bajar indicator)
        else:
            df = self.get_esios_data(
                self.config.get_indicator_id("Secundaria a bajar"),
                target_date,
                target_date
            )
            if not df.empty:
                dfs.append(df)
        
        # Combine all DataFrames if any exist
        if dfs:
            final_df = pd.concat(dfs, ignore_index=True)
            return final_df
        else:
            print(f"No data found for secundaria types {secundaria_lst} on date {target_date}")
            return pd.DataFrame()

class TerciariaPreciosDL(DescargadorESIOS):

    def __init__(self):
        super().__init__()

        #get config from esios config file 
        self.config = TerciariaConfig()
        self.terciaria_name_map = self.config.terciaria_name_map
        self.precio_unico_fecha = self.config.precio_unico_fecha
        self.cambio_granularidad_fecha = self.config.cambio_granularidad_fecha
        
    def get_prices(self, fecha_inicio: str, fecha_fin: str, terciaria_lst: list[int]) -> pd.DataFrame:
        """
        Descarga los datos de ESIOS para terciaria para un día específico.
        Maneja el cambio regulatorio del 10/12/2024 donde se cambia de precio dual (676 bajar, 677 subir) a precio único (2197).
        
        Args:
            fecha_inicio (str): Fecha de carga en formato YYYY-MM-DD (se asume que es la misma que fecha_fin)
            fecha_fin (str): Fecha de carga en formato YYYY-MM-DD (se asume que es la misma que fecha_inicio)
            terciaria_lst (list[int]): Lista de tipos de terciaria [1: subir, 2: bajar, 3: directa subir, 
                                                                  4: directa bajar, 5: programada único]
        
        Returns:
            pd.DataFrame: DataFrame con los precios de terciaria para el día solicitado
        """
        # Validate input
        invalid_ter_nums = [ter_num for ter_num in terciaria_lst if ter_num not in self.terciaria_name_map]
        if invalid_ter_nums:
            raise ValueError(f"Invalid terciaria types: {invalid_ter_nums}. Valid types are 1-5")
        
        # Ensure we're downloading data for a single day
        if fecha_inicio != fecha_fin:
            print(f"Warning: fecha_inicio ({fecha_inicio}) and fecha_fin ({fecha_fin}) differ. " 
                  f"Using fecha_inicio as the target date, only data for one day will be downloaded.")
        
        # Use only the start date for simplicity
        target_date = fecha_inicio #fecha carga == fecha fin carga only hours will vary between both dates
        target_date_dt = datetime.strptime(target_date, '%Y-%m-%d')
        
        dfs = []
        
        # Separate affected and unaffected markets
        affected_markets = [id for id in terciaria_lst if id in [1, 2, 5]]
        unaffected_markets = [id for id in terciaria_lst if id in [3, 4]]
        
        # CASE 1: Handle unaffected markets (terciaria directa) - always download normally
        for ter_num in unaffected_markets:
            indicator_id = self.config.get_indicator_id(self.terciaria_name_map[ter_num])
            df = self.get_esios_data(indicator_id, target_date, target_date)
            if not df.empty:
                dfs.append(df)
        
        # CASE 2: Handle affected markets (terciaria programada) based on date
        # If date is on or after the regulatory change
        if target_date_dt >= self.precio_unico_fecha:

            # Use the new single price indicator, iterate once if any of the affected markets are requested
            if any(ter_num in affected_markets for ter_num in [1, 2, 5]):
                indicator_id = self.config.get_indicator_id("Terciaria programada unico")
                df = self.get_esios_data(indicator_id, target_date, target_date)
                if not df.empty:
                    dfs.append(df)
        
        # CASE 3: If date is before the regulatory change
        else:
            # Use the old dual price indicators, iterate over both terciaria subir and terciaria bajar 
            for ter_num in [ter_num for ter_num in affected_markets if ter_num in [1, 2]]:
                indicator_id = self.config.get_indicator_id(self.terciaria_name_map[ter_num])
                df = self.get_esios_data(indicator_id, target_date, target_date)
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

    def get_prices(self, fecha_inicio, fecha_fin):
        """
        Descarga los datos de ESIOS para un indicador específico en un rango de fechas. Para RR (precio único) y se guarda en el indicador de RR a subir.
        """
        return self.get_esios_data(self.indicator_id, fecha_inicio, fecha_fin)


if __name__ == "__main__":
    #main()
    descargadores = [DiarioPreciosDL(), IntraPreciosDL(), SecundariaPreciosDL(), 
                     TerciariaPreciosDL(), RRPreciosDL()]
    #test_usage_save()
