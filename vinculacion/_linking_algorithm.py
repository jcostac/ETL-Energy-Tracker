import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path
import pretty_errors
import pytz
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utilidades.db_utils import DatabaseUtils
from vinculacion.configs.vinculacion_config import VinculacionConfig
from vinculacion._extract_transform_linking_data import VinculacionDataExtractor
from utilidades.progress_utils import with_progress  

class UOFUPLinkingAlgorithm:
    """Core algorithm for linking UOFs to UPs based on volume matching"""
    
    def __init__(self, database_name: str = None):
        self.config = VinculacionConfig()
        self.data_extractor = VinculacionDataExtractor()
        self.db_utils = DatabaseUtils()
        self.database_name = self.config.DATABASE_NAME if not database_name else database_name #energy_tracker == default bbdd name
        self._engine = None
        
    @property
    def engine(self):
        if self._engine is None:
            raise ValueError("Database engine not initialized. Please provide database_name in constructor.")
        else:
            return self._engine
    
    @engine.setter
    def engine(self, value):
        #test engine connection
        try:
            self._engine = value
        except Exception as e:
            raise ValueError(f"❌ Error setting engine: {e}")
        
    def _get_active_ups(self) -> pd.DataFrame:
        """
        Get all non-obsolete UPs from up_listado table
        
        Returns:
            pd.DataFrame: Active UPs
        """
        try:
            self.engine = self.db_utils.create_engine(self.database_name)
                
            # Use DatabaseUtils.read_table with where clause
            where_clause = "obsoleta != 1 OR obsoleta IS NULL"
            ups_df = self.db_utils.read_table(
                engine=self.engine,
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
            
    def _prepare_volume_data(self, omie_data: pd.DataFrame, 
                          i90_data: pd.DataFrame, target_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepares volume data for comparison by standardizing structure and filtering by target date
        
        Args:
            omie_data: OMIE processed data
            i90_data: I90 processed data
            target_date: Target date for filtering (YYYY-MM-DD)
            
        Returns:
            Tuple of prepared dataframes (omie_prepared, i90_prepared)
        """
        print("\n🔧 PREPARING VOLUME DATA FOR COMPARISON")
        print("-"*45)
        
        # Prepare OMIE data (UOFs)
        omie_prepared = omie_data.copy()
        if 'uof' in omie_prepared.columns and 'volumenes' in omie_prepared.columns:
            omie_prepared = omie_prepared[['datetime_utc', 'uof', 'volumenes']].copy()
            omie_prepared['hour'] = omie_prepared['datetime_utc'].dt.hour
            
            # Convert UTC datetime to local time, then extract date for filtering
            madrid_tz = pytz.timezone('Europe/Madrid')
            target_dt = pd.to_datetime(target_date).date()
            
            # Filter for target date using local time conversion
            omie_local_dates = omie_prepared['datetime_utc'].dt.tz_convert(madrid_tz).dt.date
            omie_prepared = omie_prepared[omie_local_dates == target_dt].copy()
            
            print(f"✅ OMIE data prepared: {len(omie_prepared)} records, {omie_prepared['uof'].nunique()} unique UOFs")
            print(f"   Unique hours in OMIE: {sorted(omie_prepared['hour'].unique())}")
        else:
            print("❌ OMIE data missing required columns")
            raise Exception("OMIE data missing required columns")
            
        # Prepare I90 data (UPs)
        i90_prepared = i90_data.copy()
        if 'up' in i90_prepared.columns and 'volumenes' in i90_prepared.columns:
            i90_prepared = i90_prepared[['datetime_utc', 'up', 'volumenes']].copy()
            i90_prepared['hour'] = i90_prepared['datetime_utc'].dt.hour
            
            # Filter for target date using local time conversion
            i90_local_dates = i90_prepared['datetime_utc'].dt.tz_convert(madrid_tz).dt.date
            i90_prepared = i90_prepared[i90_local_dates == target_dt].copy()
            
            print(f"✅ I90 data prepared: {len(i90_prepared)} records, {i90_prepared['up'].nunique()} unique UPs")
            print(f"   Unique hours in I90: {sorted(i90_prepared['hour'].unique())}")
        else:
            print("❌ I90 data missing required columns")
            raise Exception("I90 data missing required columns")
        
         #fill volumenes with 0 for hours with no data
        omie_prepared['volumenes'] = omie_prepared['volumenes'].fillna(0)
        i90_prepared['volumenes'] = i90_prepared['volumenes'].fillna(0)

         # Group by uof/up and hour, then sum volumenes (preserve the entity columns)
        omie_prepared = omie_prepared.groupby(['uof', 'hour'])['volumenes'].sum().reset_index()
        i90_prepared = i90_prepared.groupby(['up', 'hour'])['volumenes'].sum().reset_index()

        #round volumenes to 4 decimal places to handle float precision issues
        omie_prepared['volumenes'] = omie_prepared['volumenes'].round(4)
        i90_prepared['volumenes'] = i90_prepared['volumenes'].round(4)
            
        return omie_prepared, i90_prepared
    
    def _compute_hourly_hash(self, volume_list: List[float]) -> str:
        """
        Compute hash for a 24-hour volume profile
        
        Args:
            volume_list: List of hourly volumes (should be 24 values)
            
        Returns:
            str: MD5 hash of the volume profile
        """
        try:
            # Round to 4 decimal places to handle float precision issues
            rounded_volumes = [round(vol, 4) for vol in volume_list]
            volume_str = ','.join(map(str, rounded_volumes))
            return hashlib.md5(volume_str.encode()).hexdigest()
        except Exception as e:
            print(f"❌ Error computing hourly hash: {e}")
            raise e

    @with_progress(message="Creating volume profiles...", interval=2)
    def _create_volume_profiles(self, df: pd.DataFrame, up_or_uof: str) -> Tuple[Dict[str, List[float]], Dict[str, str]]:
        """
        Creates volume profiles for UOFs and UPs. A volume profile is a list of hourly values for a given entity.
        The hash is computed from the volume profile.
        
        Args:
            df: DataFrame containing volume data
            up_or_uof: Column name for the entity ('uof' or 'up')
            
        Returns:
            Tuple containing:
            - Dict mapping entity names to 24-hour volume profiles
            - Dict mapping entity names to their hash values
        """
        profiles = {}
        hashes = {}
        
        try:
            for up_uof in df[up_or_uof].unique():
                # Filter for entity (either uof or up) and sort by hour 
                up_uof_data = df[df[up_or_uof] == up_uof].sort_values('hour')
                
              
                # Create ordered volume profile
                volume_profile = []  # This list will hold the hourly volume data for the entity
                for hour in range(24):
                    # Filter the data for the current hour
                    hour_data = up_uof_data[up_uof_data['hour'] == hour]
                    if not hour_data.empty:
                        # Check if the volume for this hour is non-zero before appending
                        if hour_data['volumenes'].iloc[0] != 0:
                            #we only appned the volumnes if there is hour data and the volume is not 0
                            volume_profile.append(hour_data['volumenes'].iloc[0]) 
                        
                
                # Store volume profile and hash for entity in the dict with its name as key
                profiles[up_uof] = volume_profile 
                hashes[up_uof] = self._compute_hourly_hash(volume_profile) 

            return profiles, hashes

        except Exception as e:
            print(f"❌ Error creating volume profiles: {e}")
            raise e
        
    @with_progress(message="Finding hash matches...", interval=2)
    def _find_hash_matches(self, up_hashes: Dict[str, str], hash_to_uofs: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """
        Finds matches based on identical hashes for UOFs and UPs
        
        Args:
            up_hashes: Dict mapping UP names to their hash values ex: {'UP1': 'hash1', 'UP2': 'hash2'}
            hash_to_uofs: Dict mapping hash values to lists of UOF names ex: {'hash1': ['UOF1', 'UOF2'], 'hash2': ['UOF3', 'UOF4']}
            
        Returns:
            List of match dictionaries, i.e. exact_matches and ambiguous_matches
        """
        exact_matches = list()
        ambiguous_matches = list()
        
        for up, up_hash in up_hashes.items():
            #if the hash is in the hash_to_uofs dict, we have a match
            if up_hash in hash_to_uofs: 

                # get the list of uofs that are stored in the hash key ex: {'hash1': ['UOF1', 'UOF2'], 'hash2': ['UOF3', 'UOF4']}
                matching_uofs = hash_to_uofs[up_hash]
                
                #if there is only one uof, we have a perfect match
                if len(matching_uofs) == 1:
                    # Perfect 1:1 match
                    exact_matches.append({
                        'up': up,
                        'uof': matching_uofs[0],
                        'confidence': 1.0,
                        "match_type": "exact_unique"
                    })
                else:
                    # Ambiguous match - one UP hash matches multiple UOF hashes
                    for uof in matching_uofs:
                        ambiguous_matches.append({
                            'up': up,
                            'uof': uof,
                            'confidence': 1.0,
                            'match_type': 'exact_ambiguous'
                        })

        return exact_matches, ambiguous_matches
        
    def _find_volume_matches(self, omie_df: pd.DataFrame, i90_df: pd.DataFrame,
                          target_date: str) -> pd.DataFrame:
        """
        Finds potential UOF-UP matches based on volume patterns using hash-based comparison
        
        Args:
            omie_df: Prepared OMIE data (already filtered by target date)
            i90_df: Prepared I90 data (already filtered by target date)
            target_date: Target date for matching
            
        Returns:
            pd.DataFrame: Potential matches with confidence scores
        """
        print(f"\n🔍 FINDING VOLUME MATCHES FOR {target_date}")
        print("-"*50)
        
        if omie_df.empty or i90_df.empty:
            print("⚠️  No data available for target date")
            return pd.DataFrame()
            
        print(f"📊 Data for {target_date}:")
        print(f"   - OMIE UOFs: {omie_df['uof'].nunique()}")
        print(f"   - I90 UPs: {i90_df['up'].nunique()}")
    
        # Step 1: Create 24-hour volume profiles for UOFs (OMIE)
        print("\n🔧 Creating volume profiles for UOFs...")    
        uof_profiles, uof_hashes = self._create_volume_profiles(omie_df, 'uof')
        print(f"✅ Created {len(uof_profiles)} UOF volume profiles")
        
        # Step 2: Create 24-hour volume profiles for UPs (I90)
        print("\n🔧 Creating volume profiles for UPs...")
        up_profiles, up_hashes = self._create_volume_profiles(i90_df, 'up')
        print(f"✅ Created {len(up_profiles)} UP volume profiles")
        
        # Step 3: Create reverse mapping for efficient lookup
        hash_to_uofs = {}
        for uof, hash_val in uof_hashes.items(): 
            if hash_val not in hash_to_uofs:
                hash_to_uofs[hash_val] = []
            hash_to_uofs[hash_val].append(uof)
        
        # Step 4: Find exact matches based on identical hashes using the dedicated method
        print("\n🔍 Finding hash-based matches...")
        exact_matches, ambiguous_matches = self._find_hash_matches(up_hashes, hash_to_uofs)
        
        if not exact_matches and not ambiguous_matches:
            print("⚠️  No volume matches found")
            return pd.DataFrame(columns=['up', 'uof', 'confidence', 'match_type'])
        
        # Convert to DataFrame
        exact_matches_df = pd.DataFrame(exact_matches)
        ambiguous_matches_df = pd.DataFrame(ambiguous_matches)

        
        print(f"\n📊 MATCHING RESULTS:")
        print(f"   ✅ Unique exact matches: {len(exact_matches_df)}")
        print(f"   ⚠️  Ambiguous exact matches: {len(ambiguous_matches_df)}")
        print(f"   📈 Total matches: {len(exact_matches_df) + len(ambiguous_matches_df)}")
        
        if len(ambiguous_matches_df) > 0:
            # Show ambiguous matches summary
            ambiguous_ups = ambiguous_matches_df.groupby('up')['uof'].count()
            print(f"\n⚠️  AMBIGUOUS MATCHES DETECTED:")
            for up, count in ambiguous_ups.items():
                uofs = ambiguous_matches_df[ambiguous_matches_df['up'] == up]['uof'].tolist()
                print(f"   UP {up} matches {count} UOFs: {', '.join(uofs)}")
            breakpoint()
        
        return exact_matches_df, ambiguous_matches_df

    def _resolve_ambiguous_matches(self, exact_matches_df: pd.DataFrame, ambiguous_matches_df: pd.DataFrame, target_date: str) -> pd.DataFrame:
        """
        Resolves ambiguous matches by checking historical data n-94 days and intra markets
        
        Args:
            exact_matches_df: Exact matches from initial matching
            ambiguous_matches_df: Initial matches with potential ambiguities
            target_date: Target date
            
        Returns:
            pd.DataFrame: All resolved matches (exact + resolved ambiguous)
        """
        print(f"\n🔍 RESOLVING AMBIGUOUS MATCHES")
        print("-"*40)
        
        # Start with exact matches
        all_matches_df = exact_matches_df.copy() if not exact_matches_df.empty else pd.DataFrame()
        remaining_ambiguous_matches_df = ambiguous_matches_df.copy()
        
        if remaining_ambiguous_matches_df.empty:
            print("✅ No ambiguous matches to resolve")
            return all_matches_df

        # Try historical data (94 days back)
        print(f"\n📅 Checking historical data (-{self.config.HISTORICAL_CHECK_WINDOW} days)")
        target_date_yesterday = (pd.to_datetime(target_date) - 
                         timedelta(days=1)).strftime('%Y-%m-%d')
        
        try:
            self.data_extractor.extract_data_for_matching(target_date=target_date_yesterday)
            diario_data = self.data_extractor.transform_diario_data_for_initial_matching(target_date=target_date_yesterday)
            
            if 'omie_diario' in diario_data and 'i90_diario' in diario_data:
                omie_diario, i90_diario = self._prepare_volume_data(
                    diario_data['omie_diario'],
                    diario_data['i90_diario'],
                    target_date_yesterday
                )

                # Filter the dfs by the uofs and ups in the ambiguous_matches_df
                ambiguous_uofs = remaining_ambiguous_matches_df['uof'].unique()
                ambiguous_ups = remaining_ambiguous_matches_df['up'].unique()
                omie_diario_filtered = omie_diario[omie_diario['uof'].isin(ambiguous_uofs)]
                i90_diario_filtered = i90_diario[i90_diario['up'].isin(ambiguous_ups)]
                
                # Find the volume matches again but only for the ambiguous uofs and ups
                exact_matches_historical_df, still_ambiguous_matches_df = self._find_volume_matches(
                    omie_diario_filtered, i90_diario_filtered, target_date_yesterday
                )

                if not exact_matches_historical_df.empty:
                    print(f"✅ Resolved {len(exact_matches_historical_df)} matches using historical data")
                    all_matches_df = pd.concat([all_matches_df, exact_matches_historical_df], ignore_index=True)
                    remaining_ambiguous_matches_df = still_ambiguous_matches_df
                else:
                    print("⚠️ No matches resolved using historical data")

        except Exception as e:
            print(f"⚠️ Could not resolve using historical data: {e}")
        
        # Try intra markets if there are still ambiguous matches
        if not remaining_ambiguous_matches_df.empty:
            print(f"\n🔍 Checking intra markets for remaining {len(remaining_ambiguous_matches_df)} ambiguities on {target_date}")

            try:
                intra_data = self.data_extractor.transform_intra_data_for_ambiguous_matches(target_date)

                if 'omie_intra' in intra_data and 'i90_intra' in intra_data:
                    omie_intra = intra_data['omie_intra']
                    i90_intra = intra_data['i90_intra']
                else:
                    print("❌ No intra data available")
                    return all_matches_df
                
                # Try each session (1, 2, 3)
                for id_mercado in range(1, 4):                        
                    if remaining_ambiguous_matches_df.empty:
                        break
                        
                    print(f"   📊 Checking Intra Session {id_mercado}")
                    
                    # Filter intra data for current session
                    omie_intra_session = omie_intra[omie_intra['id_mercado'] == id_mercado]
                    i90_intra_session = i90_intra[i90_intra['id_mercado'] == id_mercado]
                    
                    if omie_intra_session.empty or i90_intra_session.empty:
                        print(f"   ⚠️ No data for session {id_mercado}, skipping")
                        continue
                    
                    # Filter by remaining ambiguous UOFs and UPs
                    ambiguous_uofs = remaining_ambiguous_matches_df['uof'].unique()
                    ambiguous_ups = remaining_ambiguous_matches_df['up'].unique()
                    omie_intra_filtered = omie_intra_session[omie_intra_session['uof'].isin(ambiguous_uofs)]
                    i90_intra_filtered = i90_intra_session[i90_intra_session['up'].isin(ambiguous_ups)]
                    
                    # Prepare data for this session
                    omie_prepared, i90_prepared = self._prepare_volume_data(
                        omie_intra_filtered,
                        i90_intra_filtered,
                        target_date
                    )
                    
                    # Find volume matches for this session
                    exact_matches_intra_df, remaining_ambiguous_matches_df = self._find_volume_matches(
                        omie_prepared, i90_prepared, target_date
                    )
                    
                    if not exact_matches_intra_df.empty:
                        print(f"   ✅ Found {len(exact_matches_intra_df)} exact matches in session {id_mercado}")
                        all_matches_df = pd.concat([all_matches_df, exact_matches_intra_df], ignore_index=True)
                    
                    print(f"   📊 Remaining ambiguous matches: {len(remaining_ambiguous_matches_df)}")
                
            except Exception as e:
                print(f"⚠️ Could not resolve using intra data: {e}")

        return all_matches_df

    def _create_final_matches_df(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates final matches DataFrame with only UP and UOF columns
        
        Args:
            matches_df: DataFrame with all matches including metadata
            
        Returns:
            pd.DataFrame: Final matches with columns [up, uof]
        """
        if matches_df.empty:
            return pd.DataFrame(columns=['up', 'uof'])
        
        # Keep only the essential columns and remove duplicates
        final_df = matches_df[['up', 'uof']].drop_duplicates().reset_index(drop=True)
        
        print(f"📊 Final matches summary:")
        print(f"   - Total unique UP-UOF pairs: {len(final_df)}")
        print(f"   - Unique UPs matched: {final_df['up'].nunique()}")
        print(f"   - Unique UOFs matched: {final_df['uof'].nunique()}")
        
        return final_df

    def _resolve_uof_conflicts(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Resolves conflicts where the same UOF is matched to multiple UPs
        
        Args:
            matches_df: DataFrame with potential UOF conflicts
            
        Returns:
            pd.DataFrame: Matches with UOF conflicts resolved
        """
        if matches_df.empty:
            return matches_df
        
        print(f"\n🔍 CHECKING FOR UOF CONFLICTS")
        print("-"*40)
        
        # Find UOFs that are matched to multiple UPs
        uof_counts = matches_df['uof'].value_counts()
        conflicted_uofs = uof_counts[uof_counts > 1].index.tolist()
        
        if not conflicted_uofs:
            print("✅ No UOF conflicts detected")
            return matches_df
        
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

    ### MAIN METHOD TO LINK UOFs TO UPs FOR A GIVEN DATE ###
    def link_uofs_to_ups(self, target_date: str, ups_to_link: List[str] = None) -> pd.DataFrame:
        """
        Main method to link UOFs to UPs for a given date
        
        Args:
            target_date: Target date for linking (YYYY-MM-DD)
            ups_to_link: List of UPs to link (optional), if not provided, all active UPs will be linked
        Returns:
            pd.DataFrame: Final links with columns [up, uof]
        """
        print(f"\n🚀 STARTING UOF-UP LINKING PROCESS")
        print(f"Target Date: {target_date} - ({self.config.DATA_DOWNLOAD_WINDOW} days window)")
        print("="*60)
        
        try:
            # Step 1: Get all active UPs if ups_to_link is not provided
            active_ups = self._get_active_ups()

            #if ups_to_link list is provided, filter the active ups to only the ones in the list
            if ups_to_link:
                active_ups = active_ups[active_ups['up'].isin(ups_to_link)]

                if active_ups.empty:
                    print("❌ UPs to link not found in the active UPs list")
                    raise ValueError("UPs to link not found in the active UPs list")
                
            # Step 2: Extract data and transform it for initial matching
            self.data_extractor.extract_data_for_matching(target_date)
            breakpoint()
            transformed_diario_data = self.data_extractor.transform_diario_data_for_initial_matching(target_date)
            
            if 'omie_diario' not in transformed_diario_data or 'i90_diario' not in transformed_diario_data:
                print("❌ Required data not available")
                raise ValueError("Required data not available")
            
         
            # Step 3: Prepare data by adding hour and date columns for easier matching
            omie_prepared, i90_prepared = self._prepare_volume_data(
                transformed_diario_data['omie_diario'],
                transformed_diario_data['i90_diario'],
                target_date
            )
            
            if omie_prepared.empty or i90_prepared.empty:
                print("❌ Data preparation failed")
                raise ValueError("Data preparation failed")
                
            # Filter for active UPs only
            i90_prepared_filtered = i90_prepared[i90_prepared['up'].isin(active_ups['up'])]
            
            # Step 4: Find initial matches (both exact and ambiguous)
            exact_matches_df, ambiguous_matches_df = self._find_volume_matches(
                omie_prepared, i90_prepared_filtered, target_date
            )
            
            if exact_matches_df.empty and ambiguous_matches_df.empty:
                print("❌ No matches found")
                return pd.DataFrame(columns=['up', 'uof'])
            
            breakpoint()
                
            # Step 5: Resolve ambiguities by comparing diario data 94 days ago and then if needed intra data
            all_resolved_matches = self._resolve_ambiguous_matches(
                exact_matches_df, ambiguous_matches_df, target_date
            ) 

            breakpoint()

            # Step 6: Resolve UOF conflicts
            all_resolved_matches, conflicted_ups = self._resolve_uof_conflicts(all_resolved_matches)

            # Step 6: Create final links
            final_matches_df = self._create_final_matches_df(all_resolved_matches)
            
            print(f"\n🎉 LINKING PROCESS COMPLETE")
            print(f"Final result: {len(final_matches_df)} UOF-UP links created")
            print("="*60)
            
            return final_matches_df
            
        except Exception as e:
            print(f"❌ Error in linking process: {e}")
            return pd.DataFrame(columns=['up', 'uof'])

   
def example_usage():
    # Initialize the algorithm
    algorithm = UOFUPLinkingAlgorithm()
    #get the target date
    target_date = algorithm.config.get_linking_target_date()
    links_df = algorithm.link_uofs_to_ups(target_date, ups_to_link= None)
    print(links_df)

if __name__ == "__main__":
    example_usage()



