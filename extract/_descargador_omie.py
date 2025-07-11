import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import zipfile
import os
import pretty_errors
from pathlib import Path
import sys
from collections import defaultdict

# Get the absolute path to the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
# Use absolute imports
from configs.omie_config import DiarioConfig, IntraConfig, IntraContinuoConfig
 
class OMIEDownloader:
    """
    Base class for downloading OMIE program data files for intradiario and diario.
    """
 
    def __init__(self):
        """
        Initializes the OMIEDownloader base class with a default temporary tracking folder and a placeholder for configuration.
        """
        self.tracking_folder = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data_lake", "temporary")
        )
        self.config = None
           
    def descarga_datos_omie_mensuales(self, month: int = None, year: int = None, months_ago: int = 4) -> pd.DataFrame:
        """
        Download and process the aggregated supply and demand curve data for a specified or default month from OMIE.
        
        Downloads the monthly ZIP file containing aggregated curve data for the given market type, extracts and processes all contained CSV files, and returns the combined data as a pandas DataFrame. The target month defaults to four months prior to the current date if not specified. Raises an exception if the download fails or file processing encounters an error.
        
        Returns:
            pd.DataFrame: Combined and processed data for the specified month.
        """
 
        #get the date for which we have a complete month of data (check if it is at least todays month -4)
        target_date = self._date_validation(month, year, months_ago)
 
        # Format the year and month as a string with leading zeros if necessary ex: 202504
        year_month = f"{target_date.year}{str(target_date.month).zfill(2)}"
 
        # Format the filename using the year_month string ex: curva_pbc_uof_202504.zip
        filename = self.filename_pattern.format(year_month=year_month)
 
        # Construct the full URL for the file ex: https://www.omie.es/es/file-download?parents=curva_pbc_uof&filename=curva_pbc_uof_202504.zip
        url = f"{self.base_url}{filename}"
 
 
        response = requests.get(url)
 
        if response.status_code == 200:
            # Save the zip file to tracking folder
            temp_zip_filename = f"{self.mercado}_{filename}"
            temp_zip_path = self._save_file(temp_zip_filename, response.content, self.tracking_folder)
 
            # Process the zip file
            try:
                with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
 
 
                    # Initialize an empty list to store data from each curva agregada
                    all_dfs = []
 
                    # Process each file in the zip
                    for file in file_list:
                        with zip_ref.open(file) as f:
 
                            # Read the CSV data
                            df = pd.read_csv(f, sep=";", skiprows=2, encoding='latin-1', skip_blank_lines=True)
 
                            # Process the dataframe
                            df = self._process_df(df, file)
 
                            # Add the dataframe to our list
                            all_dfs.append(df)
 
                    # Concatenate all dataframes into a single monthly dataframe
                    if all_dfs:
                        monthly_df = pd.concat(all_dfs, ignore_index=True)
                        return monthly_df
                    else:
                        print(f"No data files found in {filename}")
 
            except Exception as e:
                print(f"Error processing {file}: {e}")
                raise e
 
            finally:
                # Clean up the temporary zip file
                os.remove(temp_zip_path)
 
        else:
            raise Exception(f"Failed to download {filename}. Status code: {response.status_code}")
  
    def descarga_datos_omie_latest_day(self, month: int = None, year: int = None, months_ago: int = 3) -> pd.DataFrame:
        """
        Download and return OMIE aggregated supply and demand curve data for the latest available day of a specified or recent month.
        
        This method retrieves the ZIP archive for the given month (default: three months ago), identifies the most recent day's data within the archive, and processes the relevant CSV files. For the intraday market, it collects all session files for the latest date; for other markets, only the latest file is processed. Files corresponding to known error dates are skipped.
        
        Parameters:
        	month (int, optional): Target month (1-12). If not provided, determined by `months_ago`.
        	year (int, optional): Target year. If not provided, determined by `months_ago`.
        	months_ago (int, optional): Number of months before the current month to use if `month` and `year` are not specified. Default is 3.
        
        Returns:
        	pd.DataFrame: DataFrame containing the processed data for the latest available day, or an empty DataFrame if no valid data is found.
        """
 
        target_date = self._date_validation(month, year, months_ago)
 
        # Format the year and month as a string with leading zeros if necessary ex: 202504
        year_month = f"{target_date.year}{str(target_date.month).zfill(2)}"
 
        # Format the filename using the year_month string ex: curva_pbc_uof_202504.zip
        filename = self.filename_pattern.format(year_month=year_month)
 
        # Construct the full URL for the file ex: https://www.omie.es/es/file-download?parents=curva_pbc_uof&filename=curva_pbc_uof_202504.zip
        url = f"{self.base_url}{filename}"
 
 
        response = requests.get(url)
 
        if response.status_code == 200:
            # Save the zip file to tracking folder
            temp_zip_filename = f"{self.mercado}_{filename}"
            temp_zip_path = self._save_file(temp_zip_filename, response.content, self.tracking_folder)
 
            # Process the zip file
            try:
                with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
 
 
                    # Initialize an empty list to store data from each curva agregada
                    all_dfs = []
 
                    # get the latest file by date created in the zip
                    latest_file = max(file_list, key=lambda x: zip_ref.getinfo(x).date_time)


                    #get latest file date
                    latest_file_date = self._extract_date_from_filename(latest_file)
 
                    #get error dates
                    error_dates = self.config.get_error_data()['fecha']
 
                    #do not process files that have an error
                     # Get all files for the latest date
                    if self.mercado == "intra": #get all latest files for all intras
                        latest_date_files = []
                        for file in file_list:
                            try:
                                file_date = self._extract_date_from_filename(file)
                                if file_date == latest_file_date and file_date not in error_dates:
                                    latest_date_files.append(file)
                            except ValueError:
                                print(f"An error ocurred processing latest file: {file}")
                                continue
                    else:
                        if latest_file_date not in error_dates:
                            latest_date_files = [latest_file]
                        else:
                            print(f"Latest file on {latest_file_date} has an error. Skipping...")
                            return pd.DataFrame()
 
                    # Process each file for the latest date
                    for file in latest_date_files:
                        with zip_ref.open(file) as f:
 
                            # Read the CSV data
                            df = pd.read_csv(f, sep=";", skiprows=2, encoding='latin-1')
 
                            # Process the dataframe
                            df = self._process_df(df, file)

                            all_dfs.append(df)
 
 
                    # Concatenate all dataframes into a single monthly dataframe
                    if all_dfs:
                        latest_day_df = pd.concat(all_dfs, ignore_index=True)
                        return latest_day_df
                    else:
                        print(f"No data files found in {filename}")
 
            except Exception as e:
                print(f"Error processing {latest_file}: {e}")
                raise e
 
            finally:
                # Clean up the temporary zip file
                os.remove(temp_zip_path)
 
        else:
            raise Exception(f"Failed to download {filename}. Status code: {response.status_code}")
   
    ####// MAIN DOWNLOADER METHOD //####
    def descarga_omie_datos(self, fecha_inicio_carga: str, fecha_fin_carga: str, intras: list = None) -> dict:
        """
        Download and process daily OMIE market data files for a specified date range.
        
        Downloads monthly ZIP archives containing daily CSV files from OMIE, filters and processes files within the specified date range, and optionally restricts to specific intraday sessions. Excludes files with known error dates. Returns a dictionary mapping each year-month to a list of processed dataframes for the matching days.
        
        Parameters:
            fecha_inicio_carga (str): Start date in 'YYYY-MM-DD' format.
            fecha_fin_carga (str): End date in 'YYYY-MM-DD' format.
            intras (list, optional): List of intraday session numbers to include; if None, includes all available sessions.
        
        Returns:
            dict: Dictionary with keys as 'YYYYMM' strings and values as lists of processed pandas DataFrames for each matching file.
        """
        # Validate dates
        start_date, end_date = self._date_validation(fecha_inicio=fecha_inicio_carga, fecha_fin=fecha_fin_carga)
       
        # Get all months in range
        current_date = start_date.replace(day=1)
        end_month = end_date.replace(day=1)
       
        # Dictionary to store data per month
        monthly_data_dct = {}
 
        # Get error data
       
        # Download data for each month in the range
        while current_date <= end_month:
            year = current_date.year
            month = current_date.month
           
            # Format the year and month for filename
            year_month = f"{year}{str(month).zfill(2)}"
           
            # Format the filename using the year_month string
            filename = self.filename_pattern.format(year_month=year_month)
           
            # Construct the full URL for the file
            url = f"{self.base_url}{filename}"
           
            print(f"\nDownloading {filename} for {year_month}...")
            response = requests.get(url)
           
            if response.status_code == 200:
                # Save the zip file to tracking folder
                temp_zip_filename = f"{self.mercado}_{filename}"
                temp_zip_path = self._save_file(temp_zip_filename, response.content, self.tracking_folder)
               
                try:
                    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                        file_list = zip_ref.namelist()
                       
                        # Filter files for the requested days ie store files that need to be read from the zip
                        filtered_files = []
                        for file in file_list:

                            #filter by intras we want to donwload if needed
                            if intras is not None:
                                #check intras are valid
                                parsed_intras = self._parse_intra_list(intras)

                                #if intras is not None, filter files to process by intras
                                if file.split('.')[0][-1:] not in parsed_intras:
                                    print(f"Skipping {file} because it is not the intras list provided for download")
                                    continue
                            try:
                                file_date = self._extract_date_from_filename(file)
                               
                                #do not process files that have an error
                                error_dates = self.config.get_error_data()['fecha']
 
                                # Check if the file date is within the range of dates
                                if (start_date <= file_date <= end_date) and (file_date not in error_dates):
                                    filtered_files.append(file)
 
                            except ValueError:
                                # If date parsing fails, skip this file
                                print(f"Error parsing date from {file}")
                                raise e
                       
                        # If there are files to be read from the zip, process them
                        if filtered_files:
                            filtered_files = self._check_latest_file(filtered_files, zip_ref)
                            #create a list at the year_month key to store the processed dataframes
                            monthly_data_dct[year_month] = []
                           
                            # Process the filtered files
                            for file in filtered_files:
                                print(f"Processing {file}")
                                with zip_ref.open(file) as f:
                                    # Read the CSV data
                                    df = pd.read_csv(f, sep=";", skiprows=2, encoding='latin-1', skip_blank_lines=True)


                                    # Process the dataframe
                                    processed_df = self._process_df(df, file)
                                   
                                    # Store the processed dataframe in the year_month key
                                    monthly_data_dct[year_month].append(processed_df)
                           
                            print(f"Processed {len(filtered_files)} files for {year_month}")
                            return monthly_data_dct
 
                        else:
                            print(f"No matching files found in {filename} for the date range")

                        
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    raise e
               
                finally:
                    # Clean up the temporary zip file
                    os.remove(temp_zip_path)
           
            else:
                print(f"Failed to download {filename}. Status code: {response.status_code}")
           
            # Move to next month
            current_date = current_date + relativedelta(months=1)
       
        return monthly_data_dct
   
    def _parse_intra_list(self, intras: list) -> list:
        """
        Validate and convert a list of intraday session numbers to strings.
        
        Parameters:
            intras (list): List of session numbers to validate and convert.
        
        Returns:
            list: List of session numbers as strings.
        
        Raises:
            ValueError: If any session number is not in the range 1 to 7.
        """
        if intras is None:
            return None
        else:
            if intras not in [1, 2, 3, 4, 5, 6, 7]:
                raise ValueError(f"Invalid intras list: {intras}. Intras must be 1, 2, 3, 4, 5, 6 or 7")
            
        intras = [str(intra) for intra in intras]

        return intras
    
    def _process_df(self, df: pd.DataFrame, file_name: str = None) -> pd.DataFrame:
        """
        Processes and standardizes an OMIE data DataFrame, converting columns to appropriate types and handling intraday session and duplicates.
        
        For intraday market data, extracts the session number from the filename, adds it as a column, and aggregates duplicate rows by summing energy and averaging price values.
        
        Parameters:
            df (pd.DataFrame): Raw DataFrame from an OMIE CSV file.
            file_name (str, optional): Filename used to extract session information for intraday data.
        
        Returns:
            pd.DataFrame: The processed DataFrame with standardized columns and, for intraday data, session and aggregated duplicates.
        """
        # Drop any completely empty rows and columns
        df = df.dropna(axis=0, how='all')
        df = df.dropna(axis=1, how='all')
        
        # Process energy column
        if 'Energía Compra/Venta' in df.columns:
            # Replace commas with periods for decimal conversion
            df['Energía Compra/Venta'] = df['Energía Compra/Venta'].str.replace(',', '.', regex=False)
            
            # Remove periods that are used as thousands separators
            df['Energía Compra/Venta'] = df['Energía Compra/Venta'].str.replace(r'(?<=\d)\.(?=\d{3})', '', regex=True)
            
            # Convert to float, handling any remaining non-numeric values
            df['Energía Compra/Venta'] = pd.to_numeric(df['Energía Compra/Venta'])

        # Process price column with same conversion logic as energy
        if 'Precio Compra/Venta' in df.columns:
            # Replace commas with periods for decimal conversion
            df['Precio Compra/Venta'] = df['Precio Compra/Venta'].str.replace(',', '.', regex=False)
            
            # Remove periods that are used as thousands separators
            df['Precio Compra/Venta'] = df['Precio Compra/Venta'].str.replace(r'(?<=\d)\.(?=\d{3})', '', regex=True)
            
            # Convert to float, handling any remaining non-numeric values
            df['Precio Compra/Venta'] = pd.to_numeric(df['Precio Compra/Venta'])
 
        # Process date column
        if 'Fecha' in df.columns:
            df['Fecha'] = pd.to_datetime(df['Fecha'], format="%d/%m/%Y")
            # Extract the date part directly
            df['Fecha'] = df['Fecha'].dt.date
 
        # Add session column and grouping dups for intra dataset
        if self.mercado == "intra":
            # Extract session from filename - handle .1 extension properly
            if file_name.endswith('.1'):
                # For files ending with .1: curva_pibc_uof_2025010102.1
                # Session is the last digit before .1
                #sometimes the OMIE retards upload files with a "1" instead of a "01" hence its safe to take only the digit before the "."
                session_str = file_name.split('.')[0][-1:]
            else:
                raise ValueError(f"Invalid filename format: {file_name}")
            
            if session_str in ['1', '2', '3', '4', '5', '6', '7']:
                df["sesion"] = int(session_str)
                df["sesion"] = df["sesion"].astype("Int64")
                print(f"Added sesion column with value: {session_str}")
            else:
                print(f"Session '{session_str}' not in valid session list")
                raise ValueError(f"Session '{session_str}' not in valid session list (fichero mal nominado)")

            # Check if there are duplicates for intra market
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                print(f"Found {duplicate_count} duplicate rows. Grouping and aggregating...")
                
                # Get all columns except energy and price for grouping
                grouping_columns = [col for col in df.columns 
                                  if col not in ['Energía Compra/Venta', 'Precio Compra/Venta']]
                
                # Create aggregation dictionary
                agg_dict = {}
                
                # Sum energy values
                if 'Energía Compra/Venta' in df.columns:
                    agg_dict['Energía Compra/Venta'] = 'sum'
                
                # Average price values
                if 'Precio Compra/Venta' in df.columns:
                    agg_dict['Precio Compra/Venta'] = 'mean'
                
                # Group by all columns except energy and price, then aggregate
                if agg_dict and grouping_columns:
                    df = df.groupby(grouping_columns, as_index=False).agg(agg_dict)
                    print(f"After grouping: {len(df)} rows remaining")
            else:
                print("No duplicates found in the dataframe")
 
        if self.mercado == "continuo":
            pass
 
        return df
   
    def _date_validation(self, month: int = None, year: int = None, months_ago: int = 4,
                         fecha_inicio: str = None, fecha_fin: str = None) -> datetime:
        """
                         Validates and returns appropriate date(s) based on user input for OMIE data downloads.
                         
                         Depending on the provided arguments, this method:
                         - Validates a start and end date range, ensuring the end date is not before the start date and is at least 93 days before today.
                         - Validates a specific month and year, ensuring the month is at least 93 days before today.
                         - If neither is provided, returns the date corresponding to a specified number of months ago from today.
                         
                         Parameters:
                             month (int, optional): Target month (1-12).
                             year (int, optional): Target year (e.g., 2025).
                             months_ago (int, optional): Number of months to subtract from today if no explicit date is given.
                             fecha_inicio (str, optional): Start date in 'YYYY-MM-DD' format.
                             fecha_fin (str, optional): End date in 'YYYY-MM-DD' format.
                         
                         Returns:
                             datetime or tuple: A single datetime object for month/year or months_ago cases, or a tuple (start_date, end_date) if a date range is provided.
                         
                         Raises:
                             ValueError: If the provided dates are invalid or do not meet the minimum allowed date criteria.
                         """
        # Set minimum allowed date (93 days before current date)
        min_allowed_date = datetime.today() - relativedelta(days=93)
       
        # Case 1: Validating fecha_inicio and fecha_fin
        if fecha_inicio and fecha_fin:
            start_date = datetime.strptime(fecha_inicio, "%Y-%m-%d")
            end_date = datetime.strptime(fecha_fin, "%Y-%m-%d")
           
            # Validate dates
            if end_date < start_date:
                raise ValueError("End date must be after start date")
           
            if end_date > min_allowed_date:
                raise ValueError(f"End date must be at least 93 days before current date (before {min_allowed_date.strftime('%Y-%m-%d')})")
           
            return start_date, end_date
       
        # Case 2: Validating month and year
        elif month and year:
            target_date = datetime.strptime(f"{year}{str(month).zfill(2)}", "%Y%m")
            if datetime(target_date.year, target_date.month, 1) > min_allowed_date.replace(day=1):
                raise ValueError(f"The month and year provided are not valid. The month must be at least 93 days before current date.")
            return target_date
       
        # Case 3: Default to months_ago from current date
        else:
            target_date = datetime.today()
            target_date = target_date - relativedelta(months=months_ago)
            return target_date
 
    def _save_file(self, filename, content, directory):
        """
        Save content to a file in the specified directory, creating the directory if it does not exist.
        
        Supports saving bytes as binary files, pandas DataFrames as CSV files, and strings as text files. Returns the full path to the saved file.
        """
 
        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
 
        try:
            # Save the file
            if isinstance(content, bytes):
                with open(f"{directory}/{filename}", "wb") as f:
                    f.write(content)
 
            # Save the dataframe as a CSV
            elif isinstance(content, pd.DataFrame):
                content.to_csv(f"{directory}/{filename}", index=False, encoding='utf-8')
 
            # Save the string as a text file
            else:
                with open(f"{directory}/{filename}", "w", encoding='utf-8') as f:
                    f.write(content)
 
            print(f"File saved: {directory}/{filename}")
            return f"{directory}/{filename}"
 
        except Exception as e:
            print(f"Error saving file: {e}")
            raise e
 
    def _extract_date_from_filename(self, filename: str) -> datetime:
        """
        Extracts the date from an OMIE data filename according to the market type.
        
        Parameters:
            filename (str): OMIE data filename containing the date segment.
        
        Returns:
            datetime: The date extracted from the filename.
        
        Raises:
            ValueError: If the filename does not contain a valid date segment.
        """
        # Get the last part after underscore which contains the date
        date_str = filename.split('_')[-1]
       
        try:
            if self.mercado == "intra":
                # For intra market, handle both formats: YYYYMMDDXX and YYYYMMDDXX.1
                if '.' in date_str:
                    date_str = date_str.split('.')[0]  # Remove the .1 part first
                
                if len(date_str) >= 8:  # Handles YYYYMMDD, YYYYMMDDX, and YYYYMMDDXX formats
                    date_str = date_str[:8]  # Take only YYYYMMDD
                else:
                    raise ValueError(f"Error extracting date from filename: {filename}")
            else:
                # For other markets, handle format with .1
                if '.' in date_str:
                    date_str = date_str.split('.')[0]  # Remove the .1 part
                elif len(date_str) != 8:  # Should be YYYYMMDD
                    raise ValueError(f"Invalid filename format: {filename}")
           
            return datetime.strptime(date_str, "%Y%m%d")
           
        except ValueError as e:
            raise e

    def _check_latest_file(self, files: list, zip_ref: zipfile.ZipFile = None) -> list:
        """
        Return only the latest version of each session file for the intraday market, based on modification time in the ZIP archive.
        
        For the intraday market, groups files by session and selects the most recently modified file per session. For other markets or if no ZIP reference is provided, returns the input file list unchanged.
        
        Parameters:
        	files (list): List of file names to filter.
        	zip_ref (zipfile.ZipFile, optional): Reference to the ZIP archive containing the files.
        
        Returns:
        	list: List of file names representing the latest version per session (intraday) or the original list (other markets).
        """
        if self.mercado != "intra" or zip_ref is None:
            return files

        # Group files by a sesion key 
        # curva_pibc_uof_2025010102.1 -> key would be 2
        grouped_files = defaultdict(list)
        for f in files:
            # The part after the last '_' is like 'YYYYMMDDSS.version' or 'YYYYMMDDSS'
            key = f.split('_')[-1].split('.')[0][-1]
            grouped_files[key].append(f)

        latest_files_list = []
        for key, file_group in grouped_files.items():
            if len(file_group) > 1:
                # Find the latest file in the group based on modification time in the zip
                latest_file = max(file_group, key=lambda x: zip_ref.getinfo(x).date_time)
                latest_files_list.append(latest_file)
            else:
                # If only one file, it's the latest by default
                latest_files_list.append(file_group[0])

        if len(latest_files_list) < len(files):
            print(f"Filtered out {len(files) - len(latest_files_list)} older version file(s) for intraday market.")

        return latest_files_list
        
class DiarioOMIEDownloader(OMIEDownloader):
    """
    Downloader for OMIE Diario data curvas agregadas de oferta y demanda mensuales.
    """
 
    def __init__(self):
        """
        Initializes the DiarioOMIEDownloader with configuration settings for the daily OMIE market.
        
        Sets up the downloader with the appropriate base URL, market identifier, and filename pattern for downloading daily aggregated supply and demand curve data.
        """
        super().__init__()
        self.config = DiarioConfig()
        self.base_url = self.config.base_url
        self.mercado = self.config.mercado
        self.filename_pattern = self.config.filename_pattern
 
class IntraOMIEDownloader(OMIEDownloader):
    """
    Downloader for OMIE Intradiario data curvas agregadas de oferta y demanda mensuales.
    """
 
    def __init__(self):
        """
        Initializes the intraday OMIE downloader with configuration settings for the intradiario market.
        """
        super().__init__()
        self.config = IntraConfig()
        self.base_url = self.config.base_url
        self.mercado = self.config.mercado
        self.filename_pattern = self.config.filename_pattern
 
class ContinuoOMIEDownloader(OMIEDownloader):
 
    "Downloader for OMIE Continuo data curvas agregadas de oferta y demanda mensuales."
 
    def __init__(self):
        """
        Initializes the downloader for the continuous OMIE market using the IntraContinuoConfig settings.
        """
        super().__init__()
        self.config = IntraContinuoConfig()
        self.base_url = self.config.base_url
        self.mercado = self.config.mercado
        self.filename_pattern = self.config.filename_pattern
 
 
if __name__ == "__main__":
    #diario = DiarioOMIEDownloader()
    #diario.descarga_omie_datos(fecha_inicio="2025-01-01", fecha_fin="2025-01-01") 

    intradiario = IntraOMIEDownloader()
    intradiario.descarga_omie_datos(fecha_inicio_carga="2025-03-03", fecha_fin_carga="2025-03-03")

    continuo = ContinuoOMIEDownloader()
    continuo.descarga_omie_datos(fecha_inicio_carga="2025-03-03", fecha_fin_carga="2025-03-03")
    #intradiario.descarga_datos_omie_latest_day()
    #intradiario.descarga_datos_omie_mensuales()