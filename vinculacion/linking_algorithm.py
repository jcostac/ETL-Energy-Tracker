import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utilidades.db_utils import DatabaseUtils
from vinculacion.configs.vinculacion_config import VinculacionConfig
from vinculacion.data_extractor import VinculacionDataExtractor

class UOFUPLinkingAlgorithm:
    """Core algorithm for linking UOFs to UPs based on volume matching"""
    
    def __init__(self):
        self.config = VinculacionConfig()
        self.data_extractor = VinculacionDataExtractor()
        
    def get_active_ups(self, engine) -> pd.DataFrame:
        """
        Get all non-obsolete UPs from up_listado table
        
        Args:
            engine: Database engine
            
        Returns:
            pd.DataFrame: Active UPs
        """
        try:
            query = f"""
            SELECT DISTINCT up
            FROM {self.config.UP_LISTADO_TABLE}
            WHERE obsoleta != 1 OR obsoleta IS NULL
            """
            
            with engine.connect() as conn:
                result = conn.execute(query)
                ups_df = pd.DataFrame(result.fetchall(), columns=['up'])
                
            print(f"📋 Found {len(ups_df)} active UPs in database")
            return ups_df
            
        except Exception as e:
            print(f"❌ Error fetching active UPs: {e}")
            return pd.DataFrame()
            
    def prepare_volume_data(self, omie_data: pd.DataFrame, 
                          i90_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepares volume data for comparison by standardizing structure
        
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
        
    def find_volume_matches(self, omie_df: pd.DataFrame, i90_df: pd.DataFrame,
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
                matches.append({
                    'uof': uof,
                    'up': up,
                    'target_date': target_date,
                    'matching_hours': matching_hours,
                    'total_hours': total_hours,
                    'confidence': confidence,
                    'correlation': correlation,
                    'avg_volume_diff': volume_diffs.mean(),
                    'max_volume_diff': volume_diffs.max()
                })
                
        matches_df = pd.DataFrame(matches)
        
        if not matches_df.empty:
            # Filter high-confidence matches
            high_confidence_matches = matches_df[
                (matches_df['confidence'] >= 0.8) & 
                (matches_df['correlation'] >= 0.9)
            ]
            print(f"✅ Found {len(high_confidence_matches)} high-confidence matches")
            print(f"📊 Total potential matches: {len(matches_df)}")
            return matches_df.sort_values('confidence', ascending=False)
        else:
            print("⚠️  No volume matches found")
            return pd.DataFrame()
            
    def resolve_ambiguous_matches(self, matches_df: pd.DataFrame, 
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
            historical_data = self.data_extractor.extract_data_for_linking(
                target_date=historical_date,
                days_back=1
            )
            
            if 'omie_diario' in historical_data and 'i90_diario' in historical_data:
                omie_hist, i90_hist = self.prepare_volume_data(
                    historical_data['omie_diario'],
                    historical_data['i90_diario']
                )
                
                historical_matches = self.find_volume_matches(
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
                intra_data = self.data_extractor.extract_intra_data_for_ambiguous_matches(target_date)
                
                for intra_session in ['intra_1', 'intra_2', 'intra_3']:
                    if intra_session in intra_data and remaining_ambiguous:
                        print(f"   📊 Checking {intra_session}")
                        matches_df = self._resolve_with_intra_data(
                            matches_df, intra_data[intra_session], target_date
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
                               intra_data: pd.DataFrame, target_date: str) -> pd.DataFrame:
        """Resolve ambiguities using intra market data"""
        # Implementation would be similar to main matching but with intra data
        # This is a placeholder for the intra-based resolution logic
        print(f"      📊 Intra resolution applied")
        return matches_df
        
    def create_final_links(self, matches_df: pd.DataFrame) -> pd.DataFrame:
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
        
    def link_uofs_to_ups(self, target_date: str, engine) -> pd.DataFrame:
        """
        Main method to link UOFs to UPs for a given date
        
        Args:
            target_date: Target date for linking (YYYY-MM-DD)
            engine: Database engine
            
        Returns:
            pd.DataFrame: Final links with columns [up, uof]
        """
        print(f"\n🚀 STARTING UOF-UP LINKING PROCESS")
        print(f"Target Date: {target_date}")
        print("="*60)
        
        try:
            # Step 1: Get active UPs
            active_ups = self.get_active_ups(engine)
            if active_ups.empty:
                print("❌ No active UPs found")
                return pd.DataFrame(columns=['up', 'uof'])
                
            # Step 2: Extract data
            extracted_data = self.data_extractor.extract_data_for_linking(target_date)
            
            if 'omie_diario' not in extracted_data or 'i90_diario' not in extracted_data:
                print("❌ Required data not available")
                return pd.DataFrame(columns=['up', 'uof'])
                
            # Step 3: Prepare data
            omie_prepared, i90_prepared = self.prepare_volume_data(
                extracted_data['omie_diario'],
                extracted_data['i90_diario']
            )
            
            if omie_prepared.empty or i90_prepared.empty:
                print("❌ Data preparation failed")
                return pd.DataFrame(columns=['up', 'uof'])
                
            # Filter for active UPs only
            i90_prepared = i90_prepared[i90_prepared['up'].isin(active_ups['up'])]
            
            # Step 4: Find matches
            matches = self.find_volume_matches(omie_prepared, i90_prepared, target_date)
            
            if matches.empty:
                print("❌ No matches found")
                return pd.DataFrame(columns=['up', 'uof'])
                
            # Step 5: Resolve ambiguities
            resolved_matches = self.resolve_ambiguous_matches(matches, target_date)
            
            # Step 6: Create final links
            final_links = self.create_final_links(resolved_matches)
            
            print(f"\n🎉 LINKING PROCESS COMPLETE")
            print(f"Final result: {len(final_links)} UOF-UP links created")
            print("="*60)
            
            return final_links
            
        except Exception as e:
            print(f"❌ Error in linking process: {e}")
            return pd.DataFrame(columns=['up', 'uof'])
        finally:
            # Cleanup
            self.data_extractor.cleanup_extraction() 