import pytest
from datetime import datetime, timedelta

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

# Load import (common for all)
from load.local_data_lake_loader import LocalDataLakeLoader


@pytest.fixture
def test_date():
    """Fixture for a fixed test date where data should be available."""
    return '2024-01-01'


def test_esios_full_etl(test_date):
    """Test full ETL for ESIOS prices."""
    extractor = ESIOSPreciosExtractor()
    extract_result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
    assert extract_result['success'], "ESIOS extraction failed"

    transformer = TransformadorESIOS()
    transform_result = transformer.transform_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
    transform_result["data"] = transform_result["data"].head(10)
    assert transform_result['status']['success'], "ESIOS transformation failed"

    loader = LocalDataLakeLoader()
    for market in transform_result["data"].keys():
        load_result = loader.load_transformed_data_esios(transform_result["data"][market])
        assert load_result['success'], "ESIOS load failed"


def test_omie_full_etl(test_date):
    """Test full ETL for OMIE volumes."""
    extractor = OMIEExtractor()
    extract_result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
    assert extract_result['success'], "OMIE extraction failed"

    transformer = TransformadorOMIE()
    transform_result = transformer.transform_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
    assert transform_result['status']['success'], "OMIE transformation failed"

    loader = LocalDataLakeLoader()
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

    loader = LocalDataLakeLoader()
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

    loader = LocalDataLakeLoader()
    for market in transform_result["data"].keys():
        load_result = loader.load_transformed_data_volumenes_i3(transform_result["data"][market])
        assert load_result['success'], "I3 load failed"