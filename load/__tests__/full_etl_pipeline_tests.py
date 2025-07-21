import pytest
from datetime import datetime, timedelta
import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# ESIOS imports
from extract.esios_extractor import ESIOSPreciosExtractor
from transform.esios_transform import TransformadorESIOS

# OMIE imports
from extract.omie_extractor import OMIEExtractor
from transform.omie_transform import TransformadorOMIE

# I90 imports (testing precios as representative; add volumenes if needed)
from extract.i90_extractor import I90PreciosExtractor
from transform.i90_transform import TransformadorI90

# I3 imports
from extract.i3_extractor import I3Extractor
from transform.i3_transform import TransformadorI3

# Curtailments imports
from transform.curtailment_transform import CurtailmentTransformer

# Load import (common for all)
from load.data_lake_loader import DataLakeLoader


@pytest.fixture
def test_date():
    """Fixture for a fixed test date where data should be available."""
    return '2024-05-22'


@pytest.mark.parametrize("test_date", [
    (datetime.now() - timedelta(days=93)).strftime('%Y-%m-%d'),  # Recent date
    '2024-03-31',  # DST change date in Spain (clocks go forward)
])

def test_esios_full_etl(test_date):
    """Test full ETL pipeline for ESIOS prices with assertions on return structures."""
    with pytest.subTest("Extraction"):
        extractor = ESIOSPreciosExtractor()
        extract_result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
        
        # Assert on return dict structure
        assert isinstance(extract_result, dict), "Extraction result is not a dict"
        assert "success" in extract_result, "'success' key missing in extract result"
        assert "details" in extract_result, "'details' key missing in extract result"
        assert isinstance(extract_result["details"], dict), "'details' is not a dict"
        assert "markets_downloaded" in extract_result["details"], "'markets_downloaded' missing"
        assert "markets_failed" in extract_result["details"], "'markets_failed' missing"
        assert "date_range" in extract_result["details"], "'date_range' missing"
        
        # Check for partial failures
        if extract_result["details"]["markets_failed"]:
            pytest.fail(f"Extraction had failures: {extract_result['details']['markets_failed']}")
        
        assert extract_result['success'], f"ESIOS extraction failed for {test_date}"

    with pytest.subTest("Transformation"):
        transformer = TransformadorESIOS()
        transform_result = transformer.transform_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
        
        # Assert on return dict structure (based on ESIOS transform format)
        assert isinstance(transform_result, dict), "Transform result is not a dict"
        assert "data" in transform_result, "'data' key missing in transform result"
        assert "status" in transform_result, "'status' key missing in transform result"
        assert isinstance(transform_result["status"], dict), "'status' is not a dict"
        assert "success" in transform_result["status"], "'success' missing in status"
        assert "details" in transform_result["status"], "'details' missing in status"
        details = transform_result["status"]["details"]
        assert "markets_processed" in details, "'markets_processed' missing"
        assert "markets_failed" in details, "'markets_failed' missing"
        assert "mode" in details, "'mode' missing"
        assert "date_range" in details, "'date_range' missing"
        
        # Check for partial failures
        if details["markets_failed"]:
            pytest.fail(f"Transformation had failures: {details['markets_failed']}")
        
        assert transform_result['status']['success'], f"ESIOS transformation failed for {test_date}"
        
        # Verify transformed data (e.g., non-empty for at least one market, expected columns)
        for market, df in transform_result["data"].items():
            assert isinstance(df, pd.DataFrame), f"Transformed data for {market} is not a DataFrame"
            if not df.empty:
                # Example: Check for expected columns (customize based on your schema)
                expected_columns = {'datetime_utc', 'precio', 'id_mercado'}  # Adjust as per your data
                assert expected_columns.issubset(df.columns), f"Missing columns in {market} DF: {expected_columns - set(df.columns)}"

    with pytest.subTest("Loading"):
        loader = DataLakeLoader()
        for market, df in transform_result["data"].items():
            if df.empty:
                continue  # Skip empty DFs, but in real test, you might want to assert non-empty
            load_result = loader.load_transformed_data_esios(df)
            
            # Assert on load result structure
            assert isinstance(load_result, dict), "Load result is not a dict"
            assert "success" in load_result, "'success' key missing in load result"
            
            assert load_result['success'], f"ESIOS load failed for market {market} on {test_date}"

def test_omie_full_etl(test_date):
    """Test full ETL for OMIE volumes."""
    extractor = OMIEExtractor()
    extract_result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
    assert extract_result['success'], "OMIE extraction failed"

    transformer = TransformadorOMIE()
    transform_result = transformer.transform_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
    assert transform_result['status']['success'], "OMIE transformation failed"

    loader = DataLakeLoader()
    for market in transform_result["data"].keys():
        load_result = loader.load_transformed_data_volumenes_omie(transform_result["data"][market])
        assert load_result['success'], "OMIE load failed"

def test_i90_full_etl(test_date):
    """Test full ETL for I90 prices (as representative; extend for volumes if needed)."""
    extractor = I90PreciosExtractor()
    extract_result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
    assert extract_result['success'], "I90 extraction failed"

    transformer = TransformadorI90()
    transform_result = transformer.transform_data_for_all_markets(fecha_inicio=test_date, mercados_lst=["diario", "intra"])
    transform_result["data"] = transform_result["data"].head(10)
    assert transform_result['status']['success'], "I90 transformation failed"

    loader = DataLakeLoader()
    for market in transform_result["data"].keys():
        load_result = loader.load_transformed_data_precios_i90(transform_result["data"][market])
        assert load_result['success'], "I90 load failed"

def test_i3_full_etl(test_date):
    """Test full ETL for I3 volumes."""
    extractor = I3Extractor()
    extract_result = extractor.extract_data_for_all_markets(test_date, test_date)
    assert extract_result['success'], "I3 extraction failed"

    transformer = TransformadorI3()
    transform_result = transformer.transform_data_for_all_markets(test_date, test_date)
    transform_result["data"] = transform_result["data"].head(10)
    assert transform_result['status']['success'], "I3 transformation failed"

    loader = DataLakeLoader()
    for market in transform_result["data"].keys():
        load_result = loader.load_transformed_data_volumenes_i3(transform_result["data"][market])
        assert load_result['success'], "I3 load failed"

def test_curtailments_full_etl_i90(test_date):
    """Test full ETL for curtailments."""
    transformer = CurtailmentTransformer()
    transform_result = transformer.transform_curtailment_i90(fecha_inicio=test_date, fecha_fin=test_date)
    assert transform_result['status']['success'], "Curtailments transformation failed"

    loader = DataLakeLoader()
    for market in transform_result["data"].keys():
        load_result = loader.load_transformed_data_curtailments_i90(transform_result["data"][market])
        assert load_result['success'], "Curtailments load failed"

def test_curtailments_full_etl_i3(test_date):
    """Test full ETL for curtailments."""
    transformer = CurtailmentTransformer()
    transform_result = transformer.transform_curtailment_i3(fecha_inicio=test_date, fecha_fin=test_date)
    assert transform_result['status']['success'], "Curtailments transformation failed"

    loader = DataLakeLoader()
    for market in transform_result["data"].keys():
        load_result = loader.load_transformed_data_curtailments_i3(transform_result["data"][market])
        assert load_result['success'], "Curtailments load failed"


if __name__ == "__main__":
    test_esios_full_etl("2025-01-01")