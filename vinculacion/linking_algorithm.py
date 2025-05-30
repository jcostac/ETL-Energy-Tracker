import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path
import pretty_errors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utilidades.db_utils import DatabaseUtils
from vinculacion.configs.vinculacion_config import VinculacionConfig
from vinculacion.ET_volume_data import VinculacionDataExtractor

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
                          i90_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepares volume data for comparison by standardizing structure ie adding hour and date columns
        
        Args:
            omie_data: OMIE processed data
            i90_data: I90 processed data
            
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
            omie_prepared['date'] = omie_prepared['datetime_utc'].dt.date
            print(f"✅ OMIE data prepared: {len(omie_prepared)} records, {omie_prepared['uof'].nunique()} unique UOFs")
        else:
            print("❌ OMIE data missing required columns")
            return pd.DataFrame(), pd.DataFrame()
            
        # Prepare I90 data (UPs)
        i90_prepared = i90_data.copy()
        if 'up' in i90_prepared.columns and 'volumenes' in i90_prepared.columns:
            i90_prepared = i90_prepared[['datetime_utc', 'up', 'volumenes']].copy()
            i90_prepared['hour'] = i90_prepared['datetime_utc'].dt.hour
            i90_prepared['date'] = i90_prepared['datetime_utc'].dt.date
            print(f"✅ I90 data prepared: {len(i90_prepared)} records, {i90_prepared['up'].nunique()} unique UPs")
        else:
            print("❌ I90 data missing required columns")
            return pd.DataFrame(), pd.DataFrame()
            
        return omie_prepared, i90_prepared
        
    def _find_volume_matches(self, omie_df: pd.DataFrame, i90_df: pd.DataFrame,
                          target_date: str) -> pd.DataFrame:
        """
        Finds potential UOF-UP matches based on volume patterns
        
        Args:
            omie_df: Prepared OMIE data
            i90_df: Prepared I90 data  
            target_date: Target date for matching
            
        Returns:
            pd.DataFrame: Potential matches with confidence scores
        """
        print(f"\n🔍 FINDING VOLUME MATCHES FOR {target_date}")
        print("-"*50)
        
        target_dt = pd.to_datetime(target_date).date()
        
        # Filter data for target date
        omie_day = omie_df[omie_df['date'] == target_dt].copy()
        i90_day = i90_df[i90_df['date'] == target_dt].copy()
        
        if omie_day.empty or i90_day.empty:
            print("⚠️  No data available for target date")
            return pd.DataFrame()
            
        print(f"📊 Data for {target_date}:")
        print(f"   - OMIE UOFs: {omie_day['uof'].nunique()}")
        print(f"   - I90 UPs: {i90_day['up'].nunique()}")
        
        matches = []
        exact_matches = []
        
        # Get unique UOFs and UPs
        unique_uofs = omie_day['uof'].unique()
        unique_ups = i90_day['up'].unique()
        
        for uof in unique_uofs:
            uof_volumes = omie_day[omie_day['uof'] == uof].set_index('hour')['volumenes'].sort_index()
            
            for up in unique_ups:
                up_volumes = i90_day[i90_day['up'] == up].set_index('hour')['volumenes'].sort_index()
                
                # Find common hours
                common_hours = uof_volumes.index.intersection(up_volumes.index)
                
                if len(common_hours) < self.config.MIN_MATCHING_HOURS:
                    continue
                    
                # Compare volumes for common hours
                uof_common = uof_volumes.loc[common_hours]
                up_common = up_volumes.loc[common_hours]
                
                # Calculate volume differences
                volume_diffs = abs(uof_common - up_common)
                matching_hours = (volume_diffs <= self.config.VOLUME_TOLERANCE).sum()
                total_hours = len(common_hours)
                
                # Calculate confidence score
                confidence = matching_hours / total_hours if total_hours > 0 else 0
                
                # Calculate correlation for additional validation
                if len(uof_common) > 1 and uof_common.var() > 0 and up_common.var() > 0:
                    correlation = uof_common.corr(up_common)
                else:
                    correlation = 0
                    
                # Store potential match
                match_data = {
                    'uof': uof,
                    'up': up,
                    'target_date': target_date,
                    'matching_hours': matching_hours,
                    'total_hours': total_hours,
                    'confidence': confidence,
                    'correlation': correlation,
                    'avg_volume_diff': volume_diffs.mean(),
                    'max_volume_diff': volume_diffs.max(),
                    'is_exact_match': False
                }
                
                # Check if this is an "exact" match
                if (confidence >= self.config.EXACT_MATCH_CONFIDENCE_THRESHOLD and 
                    correlation >= self.config.EXACT_MATCH_CORRELATION_THRESHOLD):
                    match_data['is_exact_match'] = True
                    exact_matches.append(match_data)
                    print(f"🎯 EXACT MATCH: {up} ↔ {uof} (conf: {confidence:.3f}, corr: {correlation:.3f})")
                
                matches.append(match_data)
                
        matches_df = pd.DataFrame(matches)
        
        if not matches_df.empty:
            # Prioritize exact matches
            exact_matches_df = matches_df[matches_df['is_exact_match'] == True]
            
            print(f"🎯 Found {len(exact_matches_df)} EXACT matches")
            print(f"📊 Total potential matches: {len(matches_df)}")
            
            # If we have exact matches, filter to only include those and high-confidence ones
            if not exact_matches_df.empty:
                high_quality_matches = matches_df[
                    (matches_df['is_exact_match'] == True) | 
                    ((matches_df['confidence'] >= 0.9) & (matches_df['correlation'] >= 0.9))
                ]
                return high_quality_matches.sort_values(['is_exact_match', 'confidence'], ascending=[False, False])
            else:
                # Fallback to high-confidence matches
                high_confidence_matches = matches_df[
                    (matches_df['confidence'] >= 0.9) & 
                    (matches_df['correlation'] >= 0.9)  # Lowered from 1.0 to 0.9
                ]
                return high_confidence_matches.sort_values('confidence', ascending=False)
        else:
            print("⚠️  No volume matches found")
            return pd.DataFrame()
            
    def _resolve_ambiguous_matches(self, matches_df: pd.DataFrame, 
                                target_date: str) -> pd.DataFrame:
        """
        Resolves ambiguous matches by checking historical data and intra markets
        
        Args:
            matches_df: Initial matches with potential ambiguities
            target_date: Target date
            
        Returns:
            pd.DataFrame: Resolved matches
        """
        print(f"\n🔍 RESOLVING AMBIGUOUS MATCHES")
        print("-"*40)
        
        # Identify ambiguous matches (UPs matching multiple UOFs or vice versa)
        up_counts = matches_df.groupby('up')['uof'].nunique()
        uof_counts = matches_df.groupby('uof')['up'].nunique()
        
        ambiguous_ups = up_counts[up_counts > 1].index.tolist()
        ambiguous_uofs = uof_counts[uof_counts > 1].index.tolist()
        
        if not ambiguous_ups and not ambiguous_uofs:
            print("✅ No ambiguous matches found")
            return matches_df
            
        print(f"⚠️  Found ambiguities:")
        print(f"   - UPs with multiple UOF matches: {len(ambiguous_ups)}")
        print(f"   - UOFs with multiple UP matches: {len(ambiguous_uofs)}")
        
        # Try historical data (94 days back)
        print(f"\n📅 Checking historical data (-{self.config.HISTORICAL_CHECK_WINDOW} days)")
        historical_date = (pd.to_datetime(target_date) - 
                         timedelta(days=self.config.HISTORICAL_CHECK_WINDOW)).strftime('%Y-%m-%d')
        
        try:
            self.data_extractor.extract_data_for_matching(
                    target_date=historical_date
                ) #extract data for 94 days ago
            diario_data = self.data_extractor.transform_diario_data_for_initial_matching(target_date = historical_date)
            
            if 'omie_diario' in diario_data and 'i90_diario' in diario_data:
                omie_hist, i90_hist = self._prepare_volume_data(
                    diario_data['omie_diario'],
                    diario_data['i90_diario']
                )
                
                historical_matches = self._find_volume_matches(
                    omie_hist, i90_hist, historical_date
                )
                
                if not historical_matches.empty:
                    # Use historical matches to resolve ambiguities
                    matches_df = self._apply_historical_resolution(
                        matches_df, historical_matches, ambiguous_ups, ambiguous_uofs
                    )
                    
        except Exception as e:
            print(f"⚠️  Could not resolve using historical data: {e}")
            
        # If still ambiguous, try intra markets
        remaining_ambiguous = self._check_remaining_ambiguities(matches_df)
        
        if remaining_ambiguous:
            print(f"\n🔍 Checking intra markets for remaining ambiguities")
            try:
                intra_data = self.data_extractor.transform_intra_data_for_ambiguous_matches(target_date = historical_date)
                
                # Try each session (1, 2, 3)
                for session_num in range(1, 4):
                    i90_key = f'i90_intra_{session_num}'
                    omie_key = f'omie_intra_{session_num}'
                    
                    if (i90_key in intra_data and omie_key in intra_data and remaining_ambiguous):
                        print(f"   📊 Checking Intra Session {session_num}")
                        matches_df = self._resolve_with_intra_data(
                            matches_df, 
                            intra_data[i90_key], 
                            intra_data[omie_key],
                            target_date, 
                            session_num
                        )
                        remaining_ambiguous = self._check_remaining_ambiguities(matches_df)
                
            except Exception as e:
                print(f"⚠️  Could not resolve using intra data: {e}")
                
        return matches_df
        
    def _apply_historical_resolution(self, current_matches: pd.DataFrame,
                                   historical_matches: pd.DataFrame,
                                   ambiguous_ups: List[str],
                                   ambiguous_uofs: List[str]) -> pd.DataFrame:
        """Apply historical matching patterns to resolve current ambiguities"""
        
        resolved_matches = current_matches.copy()
        
        # Create historical mapping
        hist_mapping = {}
        hist_high_conf = historical_matches[historical_matches['confidence'] >= 0.9]
        
        for _, row in hist_high_conf.iterrows():
            hist_mapping[row['up']] = row['uof']
            
        # Apply historical resolution
        for up in ambiguous_ups:
            if up in hist_mapping:
                target_uof = hist_mapping[up]
                # Keep only the historically validated match
                mask = (resolved_matches['up'] == up) & (resolved_matches['uof'] == target_uof)
                if mask.any():
                    # Remove other matches for this UP
                    resolved_matches = resolved_matches[
                        (resolved_matches['up'] != up) | 
                        (resolved_matches['uof'] == target_uof)
                    ]
                    print(f"   ✅ Resolved {up} -> {target_uof} using historical data")
                    
        return resolved_matches
        
    def _check_remaining_ambiguities(self, matches_df: pd.DataFrame) -> bool:
        """Check if there are still ambiguous matches"""
        up_counts = matches_df.groupby('up')['uof'].nunique()
        uof_counts = matches_df.groupby('uof')['up'].nunique()
        
        return (up_counts > 1).any() or (uof_counts > 1).any()
        
    def _resolve_with_intra_data(self, matches_df: pd.DataFrame,
                               i90_intra_data: pd.DataFrame, 
                               omie_intra_data: pd.DataFrame,
                               target_date: str, 
                               session_num: int) -> pd.DataFrame:
        """
        Resolve ambiguities by comparing hourly volume patterns between I90 and OMIE intra data
        
        Args:
            matches_df: Current matches with ambiguities
            i90_intra_data: I90 intra data for specific session
            omie_intra_data: OMIE intra data for specific session
            target_date: Target date
            session_num: Session number (1, 2, or 3)
            
        Returns:
            pd.DataFrame: Matches with some ambiguities potentially resolved
        """
        if i90_intra_data.empty or omie_intra_data.empty:
            print(f"      ⚠️  No intra data available for session {session_num}")
            return matches_df
        
        target_dt = pd.to_datetime(target_date).date()
        
        # Prepare intra data
        i90_prepared = i90_intra_data.copy()
        i90_prepared['hour'] = i90_prepared['datetime_utc'].dt.hour
        i90_prepared['date'] = i90_prepared['datetime_utc'].dt.date
        i90_day = i90_prepared[i90_prepared['date'] == target_dt]
        
        omie_prepared = omie_intra_data.copy()
        omie_prepared['hour'] = omie_prepared['datetime_utc'].dt.hour
        omie_prepared['date'] = omie_prepared['datetime_utc'].dt.date
        omie_day = omie_prepared[omie_prepared['date'] == target_dt]
        
        if i90_day.empty or omie_day.empty:
            print(f"      ⚠️  No intra data for {target_date} in session {session_num}")
            return matches_df
        
        # Identify ambiguous matches
        up_counts = matches_df.groupby('up')['uof'].nunique()
        ambiguous_ups = up_counts[up_counts > 1].index.tolist()
        
        resolved_matches = matches_df.copy()
        resolution_count = 0
        
        # For each ambiguous UP, compare volume patterns with candidate UOFs
        for up in ambiguous_ups:
            up_candidates = matches_df[matches_df['up'] == up]['uof'].tolist()
            
            # Get UP's volume pattern from I90 intra
            up_volumes = i90_day[i90_day['up'] == up]
            if up_volumes.empty:
                continue
            
            up_volume_pattern = up_volumes.set_index('hour')['volumenes'].sort_index()
            
            best_uof = None
            best_confidence = -1
            
            # Compare with each candidate UOF's volume pattern from OMIE intra
            for candidate_uof in up_candidates:
                uof_volumes = omie_day[omie_day['uof'] == candidate_uof]
                if uof_volumes.empty:
                    continue
                
                uof_volume_pattern = uof_volumes.set_index('hour')['volumenes'].sort_index()
                
                # Find common hours and compare patterns
                common_hours = up_volume_pattern.index.intersection(uof_volume_pattern.index)
                
                if len(common_hours) < self.config.MIN_MATCHING_HOURS:
                    continue
                
                up_common = up_volume_pattern.loc[common_hours]
                uof_common = uof_volume_pattern.loc[common_hours]
                
                # Calculate volume differences and confidence
                volume_diffs = abs(up_common - uof_common)
                matching_hours = (volume_diffs <= self.config.VOLUME_TOLERANCE).sum()
                confidence = matching_hours / len(common_hours)
                
                # Calculate correlation
                if len(up_common) > 1 and up_common.var() > 0 and uof_common.var() > 0:
                    correlation = up_common.corr(uof_common)
                else:
                    correlation = 0
                
                # Combined score (you can adjust weights)
                combined_score = (confidence * 0.7) + (max(0, correlation) * 0.3)
                
                if combined_score > best_confidence:
                    best_confidence = combined_score
                    best_uof = candidate_uof
                
            # If we found a good match, resolve the ambiguity
            if best_uof and best_confidence >= 0.8:  # Threshold for intra resolution
                resolved_matches = resolved_matches[
                    (resolved_matches['up'] != up) | 
                    (resolved_matches['uof'] == best_uof)
                ]
                resolution_count += 1
                print(f"      ✅ Resolved {up} -> {best_uof} using session {session_num} data (confidence: {best_confidence:.3f})")
            
        print(f"      📊 Resolved {resolution_count} ambiguities using session {session_num} data")
        return resolved_matches
        
    def _create_final_matches_df(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates the final UOF-UP links DataFrame
        
        Args:
            matches_df: Resolved matches
            
        Returns:
            pd.DataFrame: Final links with columns [up, uof]
        """
        print(f"\n📋 CREATING FINAL LINKS")
        print("-"*25)
        
        if matches_df.empty:
            print("⚠️  No matches to create links from")
            return pd.DataFrame(columns=['up', 'uof'])
            
        # Take only the highest confidence match for each UP
        final_links = (matches_df
                      .sort_values('confidence', ascending=False)
                      .groupby('up')
                      .first()
                      .reset_index()
                      [['up', 'uof']])
        
        print(f"✅ Created {len(final_links)} final UOF-UP links")
        print(f"📊 Coverage:")
        print(f"   - Unique UPs linked: {final_links['up'].nunique()}")
        print(f"   - Unique UOFs linked: {final_links['uof'].nunique()}")
        
        return final_links
        
    def link_uofs_to_ups(self, target_date: str) -> pd.DataFrame:
        """
        Main method to link UOFs to UPs for a given date
        
        Args:
            target_date: Target date for linking (YYYY-MM-DD)
          
            
        Returns:
            pd.DataFrame: Final links with columns [up, uof]
        """
        print(f"\n🚀 STARTING UOF-UP LINKING PROCESS")
        print(f"Target Date: {target_date}")
        print("="*60)
        
        try:
            # Step 1: Get active UPs
            # Step 1: Get active UPs
            active_ups = self._get_active_ups()
            if active_ups.empty:
                print("❌ No active UPs found")
                return pd.DataFrame(columns=['up', 'uof'])
                
            # Step 2: Extract data
            self.data_extractor.extract_data_for_matching(target_date)
            breakpoint()
            transformed_diario_data = self.data_extractor.transform_diario_data_for_initial_matching(target_date)
            
            if 'omie_diario' not in transformed_diario_data or 'i90_diario' not in transformed_diario_data:
                print("❌ Required data not available")
                raise ValueError("Required data not available")
                
            # Step 3: Prepare data by adding hour and date columns
            omie_prepared, i90_prepared = self._prepare_volume_data(
                transformed_diario_data['omie_diario'],
                transformed_diario_data['i90_diario']
            )
            
            if omie_prepared.empty or i90_prepared.empty:
                print("❌ Data preparation failed")
                raise ValueError("Data preparation failed")
                
            # Filter for active UPs only
            i90_prepared_filtered = i90_prepared[i90_prepared['up'].isin(active_ups['up'])]
            
            # Step 4: Find matches
            matches = self._find_volume_matches(omie_prepared, i90_prepared_filtered, target_date)
            
            if matches.empty:
                print("❌ No matches found")
                return pd.DataFrame(columns=['up', 'uof'])
                
            # Step 5: Resolve ambiguities by comparing diario data 94 days ago and then if needed intra data
            resolved_matches = self._resolve_ambiguous_matches(matches, target_date)
            
            # Step 6: Create final links
            final_matches_df = self._create_final_matches_df(resolved_matches)
            
            print(f"\n🎉 LINKING PROCESS COMPLETE")
            print(f"Final result: {len(final_matches_df)} UOF-UP links created")
            print("="*60)
            
            return final_matches_df
            
        except Exception as e:
            print(f"❌ Error in linking process: {e}")
            return pd.DataFrame(columns=['up', 'uof'])
        
        finally:
            # Cleanup all files downloaded by the data extractor
            self.data_extractor.cleanup_extraction()
            
    def save_links_to_database(self, links_df: pd.DataFrame, table_name: str, 
                              if_exists: str = 'append') -> bool:
        """
        Save the generated links to database using DatabaseUtils
        
        Args:
            links_df: DataFrame with UOF-UP links
            table_name: Target table name
            if_exists: How to behave if table exists ('fail', 'replace', 'append')
            
        Returns:
            bool: Success status
        """
        try:
            if not self.engine:
                raise ValueError("Database engine not initialized")
                
            if links_df.empty:
                print("⚠️  No links to save")
                return False
                
            self.db_utils.write_table(
                engine=self.engine,
                df=links_df,
                table_name=table_name,
                if_exists=if_exists,
                index=False
            )
            
            print(f"✅ Successfully saved {len(links_df)} links to {table_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving links to database: {e}")
            return False
            
    def update_links_in_database(self, links_df: pd.DataFrame, table_name: str,
                               key_columns: List[str] = ['up']) -> bool:
        """
        Update existing links in database using DatabaseUtils
        
        Args:
            links_df: DataFrame with updated UOF-UP links
            table_name: Target table name
            key_columns: Columns to match for updates
            
        Returns:
            bool: Success status
        """
        try:
            if not self.engine:
                raise ValueError("Database engine not initialized")
                
            if links_df.empty:
                print("⚠️  No links to update")
                return False
                
            self.db_utils.update_table(
                engine=self.engine,
                df=links_df,
                table_name=table_name,
                key_columns=key_columns
            )
            
            print(f"✅ Successfully updated {len(links_df)} links in {table_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error updating links in database: {e}")
            return False 
        

def main():
    # Initialize the algorithm
    algorithm = UOFUPLinkingAlgorithm()
    #get the target date
    target_date = algorithm.config.get_linking_target_date()
    links_df = algorithm.link_uofs_to_ups(target_date)
    print(links_df)

if __name__ == "__main__":
    main()

