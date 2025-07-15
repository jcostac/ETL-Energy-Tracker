import sys 
import os 

from tracking.UOF_tracking import UOFTracker, download_uofs_from_omie
from tracking.UP_tracking import UPTracker
from tracking.ZR_tracking import ZRTracker

def test_tracking_zr():
    """
    Runs the Regulation Zones update workflow using hardcoded file paths for ESIOS and BSP data sources.
    
    Instantiates the ZRTracker, sets the input file paths, and executes the full process to synchronize zones and mappings with the database.
    """
    tracker = ZRTracker()   
        
    esios_csv = os.path.join(os.path.dirname(__file__), "..", "data_lake", "temporary", "export_unidades-de-programacion_2025-05-13_13_42.csv")
    bsp_path = os.path.join(os.path.dirname(__file__), "..", "data_lake", "temporary", "BSP-aFRR_ 01_03_2025.xlsx")
    
    tracker.process_zonas(esios_csv, bsp_path)


def test_tracking_up():
    """
    Runs the UP synchronization workflow using a specified CSV file.
    
    This function instantiates the UPTracker, sets the path to the ESIOS UP list CSV file, and initiates the process to synchronize and update UP data between the CSV source and the database.
    """

    # Example usage
    tracker = UPTracker()
    #use realtive path
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data_lake", "temporary", "export_unidades-de-programacion_2025-03-12_16_16.csv")
    tracker.process_ups(csv_path)


def test_tracking_uof():
    """
    Entry point for the UOF tracking script.
    
    Downloads the latest UOF data from OMIE, then processes and updates the database with new, obsolete, and changed UOFs. 
    """

    print("--- UOF Tracker Script ---")
    # Resolve the path properly using abspath
    download_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data_lake", "temporary"))
    download_uofs_from_omie(download_dir)
    tracker = UOFTracker()
    tracker.process_uofs(omie_path=os.path.join(download_dir, "listado_unidades.xlsx"))