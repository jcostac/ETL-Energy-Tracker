"""
Script to download and track UPs and Zonas de Regulación using Selenium (designed to be run as a cron job)
"""
import os
from datetime import datetime
import logging
from selenium_descarga_UP_list import descargador_UP_list
from UP_tracking import UPTracker
from ZR_tracking import ZRTracker
from selenium_descarga_uofs_omie import download_uofs_from_omie
from UOF_tracking import UOFTracker
import pymsteams
import sys
import importlib.util
import pretty_errors

# Get the path to the config.py file
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
config_path = os.path.join(root_dir, 'config.py')

# Ensure the directory containing config.py is in the path
sys.path.insert(0, root_dir)
# Load the module directly from the file path
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

# Now use the variable from the loaded module
pymsteams_connector = config.pymsteams_connector

def configure_logging():
    """Configure logging for the UP processor"""
    # Create logs directory if it doesn't exist
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'logs'))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Configure logger
    logger = logging.getLogger('up_zr_uof_tracker')
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(os.path.join(log_dir, 'selenium_up_zr_uof_tracker.log'))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
    
    return logger

def setup_directories():
    """Create necessary directories if they don't exist"""
    download_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../downloads/tracking'))
    if not os.path.exists(download_dir):
        os.makedirs(download_dir, exist_ok=True)
    return download_dir

def run_tracking():
    """Main function to download and process UPs, ZRs and UOFs using Selenium"""
    # Configure logger inside the function
    logger = configure_logging()
    
    try:
        logger.info("Starting Selenium UP processing job")
        
        # Setup directories
        download_dir = setup_directories()
        
        # Download latest UP list
        logger.info("Downloading UP list...")

        try:
            descargador_UP_list(download_dir)
            download_uofs_from_omie(download_dir)
            #download_bsp_list(download_dir) WIP
            
        except Exception as e:
            logger.error(f"Error downloading UP list: {str(e)}", exc_info=True)
            raise e
        
        # Find the most recent downloaded file
        esios_files = [f for f in os.listdir(download_dir) if f.startswith('export_unidades-de-programacion') and f.endswith('.csv')]
        bsp_files = [f for f in os.listdir(download_dir) if f.startswith('BSP-aFRR') and (f.endswith('.csv') or f.endswith('.xlsx'))]
        uof_files = [f for f in os.listdir(download_dir) if f.startswith('listado_unidades') and f.endswith('.xlsx')]

        if not esios_files:
            raise FileNotFoundError("No UP export files found in downloads directory")
        if not bsp_files:
            raise FileNotFoundError("No BSP file found in downloads directory")
        if not uof_files:
            raise FileNotFoundError("No UOF file found in downloads directory")
        
        #get latest esios path
        latest_esios_file = max(esios_files, key=lambda x: os.path.getctime(os.path.join(download_dir, x)))
        esios_csv_path = os.path.join(download_dir, latest_esios_file)

        #get latest bsp path
        latest_bsp_file = max(bsp_files, key=lambda x: os.path.getctime(os.path.join(download_dir, x)))
        bsp_path = os.path.join(download_dir, latest_bsp_file)

        #get latest uof path
        latest_uof_file = max(uof_files, key=lambda x: os.path.getctime(os.path.join(download_dir, x)))
        uof_xlsx_path = os.path.join(download_dir, latest_uof_file)
        
        # Process UPs
        logger.info("Processing UPs...")
        up_tracker = UPTracker()
        up_tracker.process_ups(esios_csv_path)

        logger.info("UP processing job completed successfully")

        # Process Zonas de Regulación
        logger.info("Processing Zonas de Regulación...")
        zr_tracker = ZRTracker()
        zr_tracker.process_zonas(esios_csv_path, bsp_path)

        logger.info("Zonas de Regulación processing job completed successfully")

        # Process UOFs
        logger.info("Processing UOFs...")
        uof_tracker = UOFTracker()
        uof_tracker.process_uofs(uof_xlsx_path)

        logger.info("UOF processing job completed successfully")
        
    except Exception as e:
        logger.error(f"Error in UP processing job: {str(e)}", exc_info=True)
        raise e

    finally:
        # Remove the csv file
        if os.path.exists(esios_csv_path):
            os.remove(esios_csv_path)
        if os.path.exists(uof_xlsx_path):
            os.remove(uof_xlsx_path)

if __name__ == "__main__":
    #Connection for warning messages
    myTeamsMessage = pymsteams.connectorcard(pymsteams_connector)
    myTeamsMessage.color("#339CFF")

    try:
        run_tracking()
        success_message = f"UP/ZR/UOF tracking job completed successfully on {datetime.now().strftime('%Y-%m-%d')}"
        print(success_message)
        #myTeamsMessage.text(success_message)
        #myTeamsMessage.send()
    except Exception as e:
        error_message = f"Error in UP/ZR/UOF tracking job on {datetime.now().strftime('%Y-%m-%d')}: {str(e)}"
        print(error_message)
        myTeamsMessage.text(error_message)
        myTeamsMessage.send() 