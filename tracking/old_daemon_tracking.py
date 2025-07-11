"""
Script to download and track UPs and Zonas de Regulación (designed to be run as a cron job) 
"""
import os
from datetime import datetime
import logging
from descarga_up_list import descargador_UP_list
from UP_tracking import UPTracker
from ZR_tracking import ZRTracker
from descarga_uofs_omie import download_uofs_from_omie
from UOF_tracking import UOFTracker
import sys
import importlib.util
import pretty_errors
import asyncio


# Get the path to the config.py file
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
config_path = os.path.join(root_dir, 'config.py')

# Ensure the directory containing config.py is in the path
sys.path.insert(0, root_dir)  # Insert at the beginning of sys.path for priority
# Load the module directly from the file path
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

# Now use the variable from the loaded module
pymsteams_connector = config.pymsteams_connector

def configure_logging():
    """
    Sets up and returns a logger for tracking UP, ZR, and UOF processing activities.
    
    Creates a dedicated logs directory if it does not exist, configures a logger with both file and console handlers, and ensures log messages are timestamped and formatted consistently.
    
    Returns:
        logger (logging.Logger): Configured logger instance for the tracking process.
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'logs'))  # Updated to use os.path.abspath and os.path.join
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Configure logger
    logger = logging.getLogger('up_zr_uof_tracker')
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(os.path.join(log_dir, 'up_zr_uof_tracker.log')) #takes directory and name of log file
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')) #format of log message
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')) #format of log message
        logger.addHandler(console_handler)
    
    return logger

def setup_directories():
    """
    Ensures the existence of the tracking downloads directory and returns its absolute path.
    
    Returns:
        str: The absolute path to the tracking downloads directory.
    """
    download_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../downloads/tracking'))
    if not os.path.exists(download_dir):
        os.makedirs(download_dir, exist_ok=True)
    return download_dir

async def run_tracking():
    """
    Coordinates the end-to-end workflow for downloading, verifying, and processing UPs (Unidades de Programación), Zonas de Regulación, and UOFs (Unidades de Oferta Flexible).
    
    This asynchronous function sets up logging and directories, downloads the latest required data files, identifies the most recent files for each category, and processes them using their respective tracker classes. It ensures all required files are present, logs progress and errors, and removes downloaded files after processing. Raises exceptions if any critical step fails.
    """
    # Configure logger inside the function
    logger = configure_logging()
    
    try:
        logger.info("Starting UP processing job")
        
        # Setup directories
        download_dir = setup_directories()
        
        # Download latest UP list
        logger.info("Downloading UP list...")

        try:
            await descargador_UP_list(download_dir)
            await download_uofs_from_omie(download_dir)
            #download_bsp_list(download_dir) WIP de momento no automatico
            
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
        up_tracker.process_ups(esios_csv_path)  # Set dev=False for production

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
            os.remove(esios_csv_path)
            #os.remove(bsp_path)
            os.remove(uof_xlsx_path)


