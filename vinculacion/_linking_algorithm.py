import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path
import pretty_errors
import pytz
import hashlib
import asyncio
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utilidades.db_utils import DatabaseUtils
from utilidades.storage_file_utils import RawFileUtils
from vinculacion.configs.vinculacion_config import VinculacionConfig
from vinculacion._extract_transform_linking_data import VinculacionDataExtractor
from utilidades.progress_utils import with_progress  

class UOFUPLinkingAlgorithm:
    """Core algorithm for linking UOFs to UPs based on volume matching"""
    
    def __init__(self, database_name: str = None):
        """
        Initialize the UOFUPLinkingAlgorithm with configuration, data extraction, database utilities, and raw file utilities.
        
        Parameters:
            database_name (str, optional): Name of the database to use. If not provided, defaults to the configured database name.
        """
        self.config = VinculacionConfig()
        self.data_extractor = VinculacionDataExtractor()
        self.db_utils = DatabaseUtils()
        self.database_name = self.config.DATABASE_NAME if not database_name else database_name #energy_tracker == default bbdd name
        self.raw_file_utils = RawFileUtils()
    
    def _get_engine(self):
        """
        Create and return a database engine for the configured database.
        
        Raises:
            ValueError: If the engine creation fails.
        """
        try:
            return self.db_utils.create_engine(self.database_name)
        except Exception as e:
            raise ValueError(f"❌ Error getting engine: {e}")
        
    def _get_active_ups(self) -> pd.DataFrame:
        """
        Retrieve all active (non-obsolete) UPs from the database.
        
        Returns:
            pd.DataFrame: DataFrame containing unique active UPs from the `up_listado` table.
        """
        try:
            engine = self._get_engine()
                
            # Use DatabaseUtils.read_table with where clause
            where_clause = "obsoleta != 1 OR obsoleta IS NULL"
            ups_df = self.db_utils.read_table(
                engine=engine,
                table_name=self.config.UP_LISTADO_TABLE,
                columns=['up'],
                where_clause=where_clause
            )
            
            # Remove duplicates if any
            ups_df = ups_df.drop_duplicates(subset=['up'])
                
            print(f"📋 Found {len(ups_df)} active UPs in database")
            return ups_df
            
        except Exception as e:
            print(f"❌ Error fetching active UPs: {e}")
            raise e
        
        finally:
            engine.dispose()
            
    def _prepare_volume_data(self, omie_data: pd.DataFrame, 
                          i90_data: pd.DataFrame, target_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
                          Prepare OMIE and I90 volume data for a specific date by filtering, standardizing, and aggregating records for comparison.
                          
                          The function filters both OMIE (UOF) and I90 (UP) datasets to the target date (using Madrid local time), removes zero or missing volume entries, standardizes data types, and aggregates volumes by entity, hour, and market. It raises an exception if required columns are missing in either dataset.
                          
                          Parameters:
                              omie_data (pd.DataFrame): OMIE dataset containing UOF volume records.
                              i90_data (pd.DataFrame): I90 dataset containing UP volume records.
                              target_date (str): Date to filter records by, in 'YYYY-MM-DD' format.
                          
                          Returns:
                              Tuple[pd.DataFrame, pd.DataFrame]: Prepared OMIE and I90 DataFrames, filtered and aggregated for the target date.
                          """
        print("\n🔧 PREPARING VOLUME DATA FOR COMPARISON")
        print("-"*45)

        try:
            target_dt = datetime.strptime(target_date, '%Y-%m-%d').date()
        
            # Prepare OMIE data (UOFs)
            omie_prepared = omie_data.copy()
            if 'uof' in omie_prepared.columns and 'volumenes' in omie_prepared.columns:
                omie_prepared = omie_prepared[['datetime_utc', 'uof', 'volumenes', 'id_mercado']].copy()
                omie_prepared['hour'] = omie_prepared['datetime_utc'].dt.hour
                
                # Convert UTC datetime to local time, then extract date for filtering
                madrid_tz = pytz.timezone('Europe/Madrid')
                
                # Filter for target date using local time conversion
                omie_local_dates = omie_prepared['datetime_utc'].dt.tz_convert(madrid_tz).dt.date
                omie_prepared = omie_prepared[omie_local_dates == target_dt].copy()

                # Drop rows where volumenes is 0 or NA
                omie_prepared = omie_prepared[
                    (omie_prepared['volumenes'] != 0) & 
                    (omie_prepared['volumenes'].notna())
                ]

                #assure id_mercado column is integer
                omie_prepared['id_mercado'] = omie_prepared['id_mercado'].astype(int)

                # Standardize volumenes dtype to float64
                omie_prepared['volumenes'] = omie_prepared['volumenes'].astype(np.float64).round(2)
                
                #group by uof and hour, then sum volumenes
                omie_prepared = omie_prepared.groupby(['uof', 'hour', 'id_mercado'])['volumenes'].sum().reset_index()
                
                print(f"✅ OMIE data prepared: {len(omie_prepared)} records, {omie_prepared['uof'].nunique()} unique UOFs")
            else:
                print("❌ OMIE data missing required columns")
                raise Exception("OMIE data missing required columns")
                
            # Prepare I90 data (UPs)
            i90_prepared = i90_data.copy()
            if 'up' in i90_prepared.columns and 'volumenes' in i90_prepared.columns:

                #if tipo_transaccion is in the columns, filter by "Mercado"
                if 'tipo_transaccion' in i90_prepared.columns:
                    i90_prepared = i90_prepared[i90_prepared['tipo_transaccion'] == 'Mercado']

                i90_prepared = i90_prepared[['datetime_utc', 'up', 'volumenes', 'id_mercado']].copy()
                i90_prepared['hour'] = i90_prepared['datetime_utc'].dt.hour
                
                # Filter for target date using local time conversion
                i90_local_dates = i90_prepared['datetime_utc'].dt.tz_convert(madrid_tz).dt.date
                i90_prepared = i90_prepared[i90_local_dates == target_dt].copy()

                # Drop rows where volumenes is 0 or NA
                i90_prepared = i90_prepared[
                    (i90_prepared['volumenes'] != 0) & 
                    (i90_prepared['volumenes'].notna())
                ]

                # Standardize volumenes dtype to float64
                i90_prepared['volumenes'] = i90_prepared['volumenes'].astype(np.float64).round(2)

                #group by up and hour, then sum volumenes
                i90_prepared = i90_prepared.groupby(['up', 'hour', 'id_mercado'])['volumenes'].sum().reset_index()

                print(f"✅ I90 data prepared: {len(i90_prepared)} records, {i90_prepared['up'].nunique()} unique UPs")
            else:
                print("❌ I90 data missing required columns")
                raise Exception("I90 data missing required columns")
        
        except Exception as e:
            print(f"❌ Error preparing volume data: {e}")
            raise e
        
            
        return omie_prepared, i90_prepared
    
    def _compute_hourly_hash(self, volume_list: List[float]) -> str:
        """
        Compute an MD5 hash representing a 24-hour volume profile.
        
        Parameters:
        	volume_list (List[float]): List of 24 hourly volume values.
        
        Returns:
        	str: Hexadecimal MD5 hash string of the concatenated volume profile.
        """
        try:
            volume_str = ','.join(map(str, volume_list))
            return hashlib.md5(volume_str.encode()).hexdigest()
        except Exception as e:
            print(f"❌ Error computing hourly hash: {e}")
            raise e

    @with_progress(message="Creating combined volume profiles...", interval=2)
    async def _create_combined_volume_profiles(self, df: pd.DataFrame, up_or_uof: str) -> Tuple[Dict[str, List[float]], Dict[str, str]]:
        """
        Asynchronously generates combined volume profiles and their hashes for each entity across all markets.
        
        For each unique entity (UOF or UP), concatenates hourly volumes from markets 1 to 4 into a single profile list and computes its hash. Returns dictionaries mapping each entity to its combined profile and corresponding hash value.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing volume data with 'id_mercado' and 'hour' columns.
            up_or_uof (str): Column name specifying the entity type ('uof' or 'up').
        
        Returns:
            Tuple[Dict[str, List[float]], Dict[str, str]]: 
                - Dictionary mapping entity names to their combined volume profiles.
                - Dictionary mapping entity names to their profile hash values.
        """
        profiles = {}
        hashes = {}
        
        try:
            unique_entities = df[up_or_uof].unique()
            
            # Process entities concurrently
            tasks = [
                self._process_single_entity(df, up_or_uof, entity_name) 
                for entity_name in unique_entities
            ]
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks)
            
            # Combine results
            for entity_name, profile, hash_value in results:
                profiles[entity_name] = profile
                hashes[entity_name] = hash_value
                
                if entity_name == "ZABU" or entity_name == "TERE":
                    print(f"{up_or_uof}: {entity_name}")
                    print(f"volume_profile: {profile}")
                    print(f"hash: {hash_value}")
                    print("-"*50)
            
            return profiles, hashes
        
        except Exception as e:
            print(f"❌ Error creating combined volume profiles: {e}")
            raise e

    async def _process_single_entity(self, df: pd.DataFrame, up_or_uof: str, entity_name: str) -> Tuple[str, List[float], str]:
        """
        Asynchronously generates the combined volume profile and hash for a single entity across all markets.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing volume data.
            up_or_uof (str): Column name specifying the entity type ('uof' or 'up').
            entity_name (str): Name of the entity to process.
        
        Returns:
            tuple: (entity_name, combined_volume_profile, hash_value), where combined_volume_profile is a list of nonzero hourly volumes across markets 1 to 4, and hash_value is the MD5 hash of this profile.
        """
        # Filter data for current entity
        entity_data = df[df[up_or_uof] == entity_name]
        
        combined_volume_profile = []
        
        # Process each market (1: diario, 2-4: intra sessions) in order
        for id_mercado in [1, 2, 3, 4]:
            # Filter for the current market
            market_data = entity_data[entity_data['id_mercado'] == id_mercado]
            market_data = market_data.sort_values('hour')
            
            market_profile = []  # This will hold the volume profile for the current market

            for hour in range(24):  # Iterate over all hours in the day
                hour_data = market_data[market_data['hour'] == hour]  # Filter for the current hour

                if not hour_data.empty:  # Check if there is data for the current hour
                    hour_volume = hour_data['volumenes'].iloc[0]
                    if hour_volume != 0:  # If the volume is not 0, append the volume to the profile
                        market_profile.append(hour_volume)
                    else:  # If the volume is 0, do not append anything to the profile
                        pass

            combined_volume_profile.extend(market_profile)  # Append the market profile to the combined profile
        
        # Compute hash (potentially CPU-bound, so we might want to run it in a thread pool)
        hash_value = await asyncio.to_thread(self._compute_hourly_hash, combined_volume_profile)
        
        return entity_name, combined_volume_profile, hash_value

    @with_progress(message="Creating combined volume profiles...", interval=2)
    async def _run_matching_round(self, omie_df: pd.DataFrame, i90_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Performs a single asynchronous matching round between UOFs and UPs using their prepared volume profiles.
        
        Given OMIE (UOF) and I90 (UP) DataFrames, this method generates combined volume profiles and hashes for each entity, then compares these hashes to identify exact and ambiguous matches. Returns two DataFrames: one with unique exact matches and another with ambiguous matches requiring further resolution.
        
        Parameters:
            omie_df (pd.DataFrame): Prepared OMIE data for UOFs.
            i90_df (pd.DataFrame): Prepared I90 data for UPs.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: DataFrames of exact matches and ambiguous matches, each with columns ['up', 'uof', 'match_type'].
        """
        if omie_df.empty or i90_df.empty:
            print("⚠️  No data available for this matching round.")
            return pd.DataFrame(columns=['up', 'uof', 'match_type']), pd.DataFrame(columns=['up', 'uof', 'match_type'])

        # 1. Create volume profiles for UOFs and UPs concurrently
        print("\n🔧 Creating volume profiles for UOFs and UPs...")    
        
        uof_task = self._create_combined_volume_profiles(omie_df, 'uof')
        up_task = self._create_combined_volume_profiles(i90_df, 'up')
        
        # Execute both profile creation tasks concurrently
        (uof_profiles, uof_hashes), (up_profiles, up_hashes) = await asyncio.gather(uof_task, up_task)
        
        print(f"✅ Created {len(uof_hashes)} UOF profiles and {len(up_hashes)} UP profiles")
        
        # 3. Create reverse mapping for lookup # ie: hash: list of uofs
        hash_to_uofs = {}
        for uof, hash_val in uof_hashes.items():
            if hash_val not in hash_to_uofs:
                hash_to_uofs[hash_val] = []
            hash_to_uofs[hash_val].append(uof)
            
        # 4. Find matches based on up hashes
        print("\n🔍 Finding hash-based matches...")
        exact_matches, ambiguous_matches = self._find_hash_matches(up_hashes, hash_to_uofs)

        exact_matches_df = pd.DataFrame(exact_matches)
        ambiguous_matches_df = pd.DataFrame(ambiguous_matches)
        
        print(f"\n📊 MATCHING ROUND RESULTS:")
        print(f"   ✅ Unique exact matches: {len(exact_matches_df)}")
        print(f"   ⚠️  Ambiguous matches: {len(ambiguous_matches_df)}")
        
        return exact_matches_df, ambiguous_matches_df
    
    @with_progress(message="Finding hash matches...", interval=2)
    def _find_hash_matches(self, up_hashes: Dict[str, str], hash_to_uofs: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """
        Finds exact and ambiguous matches between UPs and UOFs based on identical volume profile hashes.
        
        Compares UP hashes to UOF hashes to identify unique (1:1) matches and ambiguous cases where multiple UPs or UOFs share the same hash.
        
        Parameters:
            up_hashes (Dict[str, str]): Mapping of UP names to their hash values.
            hash_to_uofs (Dict[str, List[str]]): Mapping of hash values to lists of UOF names.
        
        Returns:
            Tuple[List[Dict[str, str]], List[Dict[str, str]]]: Two lists of match dictionaries: exact unique matches and ambiguous matches.
        """
        exact_matches = list()
        ambiguous_matches = list()
        
        # Create reverse mapping to count UPs per hash
        hash_to_ups = {}
        for up, up_hash in up_hashes.items():
            if up_hash not in hash_to_ups:
                hash_to_ups[up_hash] = []   
            hash_to_ups[up_hash].append(up)
        
        for up, up_hash in up_hashes.items():
            # Check if this hash has a match in UOFs
            if up_hash in hash_to_uofs: 
                matching_uofs = hash_to_uofs[up_hash]
                matching_ups = hash_to_ups[up_hash]
                
                # True 1:1 match requires BOTH conditions:
                if len(matching_uofs) == 1 and len(matching_ups) == 1:
                    # Perfect 1:1 match
                    exact_matches.append({
                        'up': up,
                        'uof': matching_uofs[0],
                        "match_type": "exact_unique"
                    })
                else:
                    # Ambiguous - multiple UOFs OR multiple UPs with same hash
                    for uof in matching_uofs:
                        ambiguous_matches.append({
                            'up': up,
                            'uof': uof,
                            'match_type': 'ambiguous'
                        })

        return exact_matches, ambiguous_matches
        
    def _resolve_name_matches(self, ambiguous_matches_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Resolve ambiguous UP-UOF matches by selecting pairs where the UP and UOF names are identical.
        
        Parameters:
            ambiguous_matches_df (pd.DataFrame): DataFrame containing ambiguous UP-UOF matches.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing a DataFrame of matches resolved by name and a DataFrame of remaining ambiguous matches.
        """
        if ambiguous_matches_df.empty:
            return pd.DataFrame(), ambiguous_matches_df
            
        print(f"\n🎯 RESOLVING AMBIGUOUS MATCHES BY NAME MATCHING")
        print("-"*50)
        
        name_resolved_matches = []
        remaining_ambiguous = []
        
        # Group by UP to handle each UP's ambiguous matches
        for up in ambiguous_matches_df['up'].unique():
            up_matches = ambiguous_matches_df[ambiguous_matches_df['up'] == up]
            matching_uofs = up_matches['uof'].tolist()
            
            # Check if UP name matches any of the UOF names
            if up in matching_uofs:
                print(f"🎯 Found name match: UP '{up}' matches UOF '{up}' (resolving ambiguity)")
                # Add the name match as resolved
                name_resolved_matches.append({
                    'up': up,
                    'uof': up,
                    'match_type': 'name_resolved'
                })
            else:
                # Keep all matches for this UP as still ambiguous
                for _, match in up_matches.iterrows():
                    remaining_ambiguous.append(match.to_dict())
        
        name_resolved_df = pd.DataFrame(name_resolved_matches)
        remaining_ambiguous_df = pd.DataFrame(remaining_ambiguous)
        
        print(f"✅ Resolved {len(name_resolved_df)} matches by name matching")
        print(f"⚠️  Remaining ambiguous matches: {len(remaining_ambiguous_df)}")
        
        return name_resolved_df, remaining_ambiguous_df
    
    def _resolve_uof_conflicts(self, matches_df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        """
        Remove matches where a UOF is linked to multiple UPs, returning only conflict-free matches.
        
        Parameters:
        	matches_df (pd.DataFrame): DataFrame containing UOF-UP match pairs.
        
        Returns:
        	Tuple[pd.DataFrame, list]: A tuple with the cleaned matches DataFrame (excluding conflicted UOFs) and a list of UPs involved in conflicts.
        """
        if matches_df.empty:
            return matches_df, []
        
        print(f"\n🔍 CHECKING FOR UOF CONFLICTS")
        print("-"*40)
        
        # Find UOFs that are matched to multiple UPs (have a value count > 1)
        uof_counts = matches_df['uof'].value_counts()
        conflicted_uofs = uof_counts[uof_counts > 1].index.tolist()
        
        if not conflicted_uofs:
            print("✅ No UOF conflicts detected")
            return matches_df, []
        
        print(f"⚠️  Found {len(conflicted_uofs)} UOFs with conflicts:")
        for uof in conflicted_uofs:
            ups = matches_df[matches_df['uof'] == uof]['up'].tolist()
            print(f"   UOF {uof} matched to UPs: {', '.join(ups)}")
        
        # Remove all conflicted matches for now (conservative approach)
        # Alternative: implement priority-based resolution
        clean_matches_df = matches_df[~matches_df['uof'].isin(conflicted_uofs)].copy()
        
        print(f"🔧 Removed {len(matches_df) - len(clean_matches_df)} conflicted matches")
        print(f"📊 Remaining matches: {len(clean_matches_df)}")

        #get ups from conflicted uofs
        conflicted_ups = matches_df[matches_df['uof'].isin(conflicted_uofs)]['up'].unique()
        print(f"🔧 Conflicted UPs: {', '.join(conflicted_ups)}")
        
        return clean_matches_df, conflicted_ups

    def _create_final_matches_df(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a DataFrame of final UP-UOF matches with only the essential columns and a date stamp.
        
        Parameters:
            matches_df (pd.DataFrame): DataFrame containing matched UP-UOF pairs and metadata.
        
        Returns:
            pd.DataFrame: DataFrame with columns ['up', 'uof', 'date_updated'] representing the final matches.
        """
        if matches_df.empty:
            return pd.DataFrame(columns=['up', 'uof'])
        
        # Keep only the essential columns and remove duplicates
        final_df = matches_df[['up', 'uof']]
        final_df['date_updated'] =  datetime.now().date().strftime('%Y-%m-%d')

        print(f"📊 Final matches summary:")
        print(f"   - Total unique UP-UOF pairs: {len(final_df)}")
        print(f"   - Unique UPs matched: {final_df['up'].nunique()}")
        print(f"   - Unique UOFs matched: {final_df['uof'].nunique()}")
        
        return final_df

    def _check_raw_data_exists_for_date(self, target_date: str, market: str, dataset_type: str, source: str) -> bool:
        """
        Check if a raw CSV data file exists for a given date, market, dataset type, and source.
        
        Returns:
            bool: True if the file exists and contains data for the specified date; otherwise, False.
        """
        try:
            
            dt = datetime.strptime(target_date, '%Y-%m-%d')
            year, month = dt.year, dt.month

            if source == 'omie':
                file_path = self.raw_file_utils.raw_path / market / str(year) / f"{month:02d}" / f"{dataset_type}.csv"
            elif source == 'i90':
                file_path = self.raw_file_utils.raw_path / market / str(year) / f"{month:02d}" / f"{dataset_type}.csv"
            else:

                return False

            if not file_path.exists():
                print(f"INFO: Raw file not found: {file_path}")
                return False

            if source == 'omie':
                df = pd.read_csv(file_path)
                if 'Fecha' not in df.columns:
                    return False
                dates_in_file = pd.to_datetime(df['Fecha']).dt.strftime('%Y-%m-%d')
                if target_date in dates_in_file.values:
                    return True
                else:
                    return False
            
            elif source == 'i90':
                df = pd.read_csv(file_path)
                if 'fecha' not in df.columns:
                    return False
                dates_in_file = pd.to_datetime(df['fecha']).dt.strftime('%Y-%m-%d')
                if target_date in dates_in_file.values:
                    return True
                else:
                    
                    return False

        except Exception as e:
            print(f"ERROR: Could not check for existing raw data for {source.upper()} {market.upper()} on {target_date}: {e}")
            return False

    ### MAIN METHOD TO LINK UOFs TO UPs FOR A GIVEN DATE ###
    async def link_uofs_to_ups(self, target_date: str, ups_to_link: List[str] = None, save_to_db: bool = False) -> Dict:
        """
        Links UOFs to UPs for a specified date using a two-round volume profile matching process with ambiguity and conflict resolution.
        
        This asynchronous method orchestrates the full pipeline for linking UOFs (units of operation) to UPs (units of production) by:
        - Preparing and validating required raw data for the target date (and previous day if needed).
        - Extracting and transforming OMIE and I90 datasets.
        - Matching UOFs to UPs based on combined hourly volume profiles across multiple markets.
        - Resolving ambiguous matches by name and, if necessary, using historical data from the previous day.
        - Handling conflicts where a UOF is matched to multiple UPs.
        - Optionally saving the resulting links to the database.
        
        Parameters:
            target_date (str): The date for which to perform the linking (format: YYYY-MM-DD).
            ups_to_link (List[str], optional): List of UPs to restrict the linking process to. If not provided, all active UPs are considered.
            save_to_db (bool, optional): If True, saves the resulting links to the database.
        
        Returns:
            Dict: A dictionary containing:
                - 'target_date': The date for which linking was performed.
                - 'links_df': DataFrame of final UOF-UP links.
                - 'success': Boolean indicating if the process completed successfully.
                - 'message': Status or error message.
        """
        print(f"\n🚀 STARTING UOF-UP LINKING PROCESS")
        print(f"Target Date: {target_date}")
        print("="*60)
        
        results = {
            'target_date': target_date,
            'links_df': pd.DataFrame(columns=['up', 'uof']),
            'success': False,
            'message': ''
        }
        
        try:
            # Step 1: Get all active UPs if ups_to_link is not provided
            active_ups = self._get_active_ups()

            #if ups list we wanto to link  is provided, filter the active ups to only the ones in the list
            if ups_to_link:
                active_ups = active_ups[active_ups['up'].isin(ups_to_link)]
                print(f"🔍 Filtered active UPs to only the ones in the list: {ups_to_link}")
                print(f"🔍 Number of UPs we are filtering for: {len(active_ups)}")

                if active_ups.empty:
                    print("❌ UPs to link not found in the active UPs list")
                    raise ValueError("UPs to link not found in the active UPs list")
                
            # Step 2: Check for data and extract if needed
            omie_exists = self._check_raw_data_exists_for_date(target_date, 'diario', 'volumenes_omie', 'omie') and \
                          self._check_raw_data_exists_for_date(target_date, 'intra', 'volumenes_omie', 'omie')
            i90_exists = self._check_raw_data_exists_for_date(target_date, 'diario', 'volumenes_i90', 'i90') and \
                         self._check_raw_data_exists_for_date(target_date, 'intra', 'volumenes_i90', 'i90')

            if omie_exists and i90_exists:
                print(f"✅ Raw data for {target_date} already exists. Skipping extraction.")
            else:
                
                print(f"ℹ️ Raw data for {target_date} not fully available. Starting extraction...")
                self.data_extractor.extract_data_for_matching(target_date)

            all_data = self.data_extractor.transform_and_combine_data_for_linking(target_date)


            
            omie_combined = all_data.get('omie_combined')
            i90_combined = all_data.get('i90_combined')

            if omie_combined is None or omie_combined.empty or i90_combined is None or i90_combined.empty:
                raise Exception("❌ Required data not available for matching on target date.")

            # Step 3: Prepare volume data (timezone conversion, date filtering, grouping, rounding)
            omie_prepared, i90_prepared = self._prepare_volume_data(omie_combined, i90_combined, target_date)


            # Filter for active UPs only
            i90_prepared_active = i90_prepared[i90_prepared['up'].isin(active_ups['up'])]
            print(f"🔍 UPs after filtering: {len(i90_prepared['up'].unique())} -> {len(i90_prepared_active['up'].unique())}")
            
            # --- ROUND 1: Matching on target_date ---
            print("\n\n--- ROUND 1: MATCHING ON TARGET DATE ---")
            exact_matches_r1, ambiguous_matches_r1 = await self._run_matching_round(omie_prepared, i90_prepared_active)
            
            # --- AMBIGUITY RESOLUTION ---
            print("\n\n--- RESOLVING AMBIGUITIES ---")
            
            # Step 1: Name Matching
            name_resolved_df, remaining_ambiguous_df = self._resolve_name_matches(ambiguous_matches_r1)
            
            # Collect all confirmed matches so far
            confirmed_matches = pd.concat([exact_matches_r1, name_resolved_df], ignore_index=True)
            
            # Get already matched UPs and UOFs to exclude from further matching
            matched_ups = set(confirmed_matches['up'].tolist()) if not confirmed_matches.empty else set()
            matched_uofs = set(confirmed_matches['uof'].tolist()) if not confirmed_matches.empty else set()
            
            print(f"🔒 Excluding {len(matched_ups)} already matched UPs and {len(matched_uofs)} already matched UOFs from further matching")
            
            all_matches_list = [confirmed_matches]
            
            # Step 2: Historical Matching - only for remaining unmatched entities
            if not remaining_ambiguous_df.empty:
                print(f"\n📅 Attempting to resolve {len(remaining_ambiguous_df)} matches with historical data...")
                
                historical_date = (pd.to_datetime(target_date) - timedelta(days=1)).strftime('%Y-%m-%d')
                
                # Check for historical data and extract if needed
                omie_hist_exists = self._check_raw_data_exists_for_date(historical_date, 'diario', 'volumenes_omie', 'omie') and \
                                   self._check_raw_data_exists_for_date(historical_date, 'intra', 'volumenes_omie', 'omie')
                i90_hist_exists = self._check_raw_data_exists_for_date(historical_date, 'diario', 'volumenes_i90', 'i90') and \
                                  self._check_raw_data_exists_for_date(historical_date, 'intra', 'volumenes_i90', 'i90')

                if omie_hist_exists and i90_hist_exists:
                    print(f"✅ Raw historical data for {historical_date} already exists. Skipping extraction.")
                else:
                    print(f"ℹ️ Raw historical data for {historical_date} not fully available. Starting extraction...")
                    self.data_extractor.extract_data_for_matching(historical_date)
                
                historic_data = self.data_extractor.transform_and_combine_data_for_linking(historical_date)

                omie_hist = historic_data.get('omie_combined')
                i90_hist = historic_data.get('i90_combined')

                if omie_hist is not None and not omie_hist.empty and i90_hist is not None and not i90_hist.empty:
                    
                    # FILTER RAW HISTORICAL DATA TO EXCLUDE ALREADY MATCHED ENTITIES and only active ups for i90
                    # Remove UOFs and UPs that were already matched in Round 1
                    omie_hist_filtered_raw = omie_hist[~omie_hist['uof'].isin(matched_uofs)]
                    i90_hist_filtered_raw = i90_hist[~i90_hist['up'].isin(matched_ups)]

                    #get only active ups in i90_hist_filtered_raw
                    i90_hist_filtered_raw = i90_hist_filtered_raw[i90_hist_filtered_raw['up'].isin(active_ups['up'])]
                    
                    print(f"🔍 Historical matching scope (after excluding already matched):")
                    print(f"   - Total UOFs in historical data: {len(omie_hist['uof'].unique())}")
                    print(f"   - Available UOFs (excluding matched): {len(omie_hist_filtered_raw['uof'].unique())}")
                    print(f"   - Total UPs in historical data: {len(i90_hist['up'].unique())}")
                    print(f"   - Available UPs (excluding matched): {len(i90_hist_filtered_raw['up'].unique())}")
                    
                    # Now prepare the filtered historical data
                    omie_hist_prepared, i90_hist_prepared = self._prepare_volume_data(
                        omie_hist_filtered_raw, i90_hist_filtered_raw, historical_date
                    )

                    # --- ROUND 2: Matching on historical_date ---
                    print("\n\n--- ROUND 2: MATCHING ON HISTORICAL DATE (UNMATCHED ENTITIES ONLY) ---")
                    exact_matches_r2, ambiguous_matches_r2 = await self._run_matching_round(omie_hist_prepared, i90_hist_prepared)
                    
                    if not exact_matches_r2.empty:
                        exact_matches_r2['match_type'] = 'historical_resolved'
                        all_matches_list.append(exact_matches_r2)
                    
                    print(f"✅ Resolved {len(exact_matches_r2)} from historical data.")
                    print(f"⚠️ Unresolved after historical check: {len(ambiguous_matches_r2)}")
                else:
                    print("✅ All ambiguous matches were already resolved - no historical matching needed")
            
            # --- FINALIZATION ---
            # Consolidate all resolved matches
            final_resolved_matches = pd.concat(all_matches_list, ignore_index=True) if all_matches_list else pd.DataFrame()
            
            # Resolve UOF conflicts (should be minimal now due to exclusion logic)
            if not final_resolved_matches.empty:
                final_resolved_matches, conflicted_ups = self._resolve_uof_conflicts(final_resolved_matches)

            # Create final links
            final_matches_df = self._create_final_matches_df(final_resolved_matches)

            print(f"\n🎉 LINKING PROCESS COMPLETE")
            print(f"Final result: {len(final_matches_df)} UOF-UP links created")
            print("="*60)
            
            results['links_df'] = final_matches_df
            results['success'] = True
            results['message'] = f"Successfully created {len(final_matches_df)} UOF-UP links for target date {target_date}."

            if save_to_db: #save to db if True
                print(f"Saving links to database...")
                self._save_links_to_database(final_matches_df)
            
            return results
            
        except Exception as e:
            results['message'] = f"Error in linking process: {e}"
            print(f"❌ Error in linking process: {e}")
            return results
        
    def _save_links_to_database(self, links_df: pd.DataFrame):
        """
        Save the provided links DataFrame to the configured database table.
        
        Parameters:
            links_df (pd.DataFrame): DataFrame containing the UP-UOF links to be saved.
        """
        try:

            engine = self.db_utils.create_engine(self.config.DATABASE_NAME)
            self.db_utils.write_table(engine, links_df, self.config.UP_UOF_VINCULACION_TABLE, if_exists='append')

        except Exception as e:
            print(f"❌ Error saving links to database: {e}")
            raise e
        
        finally:
            if engine:
                engine.dispose()




