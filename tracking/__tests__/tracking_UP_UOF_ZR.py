import sys 
import os 

from tracking.UOF_tracking import UOFTracker, download_uofs_from_omie
from tracking.UP_tracking import UPTracker
from tracking.ZR_tracking import ZRTracker

def test_tracking_zr():
    """
    Execute the Regulation Zones (ZR) synchronization workflow using predefined ESIOS CSV and BSP Excel files.
    
    This function instantiates a ZRTracker, sets up input file paths for ESIOS and BSP data, and processes them to update and synchronize regulation zones and their mappings in the database.
    """
    tracker = ZRTracker()   
        
    esios_csv = os.path.join(os.path.dirname(__file__), "..", "data_lake", "temporary", "export_unidades-de-programacion_2025-05-13_13_42.csv")
    bsp_path = os.path.join(os.path.dirname(__file__), "..", "data_lake", "temporary", "BSP-aFRR_ 01_03_2025.xlsx")
    
    tracker.process_zonas(esios_csv, bsp_path)


def test_tracking_up():
    """
    Synchronizes and updates Programming Units (UP) data in the database using a specified CSV file.
    
    This function instantiates a UPTracker, locates the UP data CSV file, and processes it to ensure the database reflects the latest UP information from the source file.
    """

    # Example usage
    tracker = UPTracker()
    #use realtive path
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data_lake", "temporary", "export_unidades-de-programacion_2025-03-12_16_16.csv")
    tracker.process_ups(csv_path)


def test_tracking_uof():
    """
    Downloads the latest UOF data from OMIE and processes it to update the database with new, obsolete, and changed UOF records.
    
    This function serves as the main workflow for synchronizing UOF data by downloading the most recent file, then invoking the tracker to process and apply updates.
    """

    print("--- UOF Tracker Script ---")
    # Resolve the path properly using abspath
    download_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data_lake", "temporary"))
    download_uofs_from_omie(download_dir)
    tracker = UOFTracker()
    tracker.process_uofs(omie_path=os.path.join(download_dir, "listado_unidades.xlsx"))