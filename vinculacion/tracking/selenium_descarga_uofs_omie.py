import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import os
import pretty_errors

def download_listado_unidades(download_dir: str) -> str:
    """
    Downloads the 'Listado de unidades' Excel file from OMIE to the specified directory.

    Args:
        download_dir (str): The directory where the file will be saved.
    """
    url = "https://www.omie.es/es/listado-de-agentes"
    
    # Configure Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run in headless mode
    chrome_options.add_argument('--disable-gpu')  # Required for some systems
    chrome_options.add_argument('--no-sandbox')  # Required for some systems
    chrome_options.add_argument('--disable-dev-shm-usage')  # Required for some systems
    chrome_options.add_experimental_option("prefs", {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
    })
    
    # Initialize the driver
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        # Navigate to the page
        print(f"Navigating to {url}")
        driver.get(url)
        
        # Wait for and handle the cookie popup
        wait = WebDriverWait(driver, 20)
      
        try:
            # Now proceed with the download
            listado_unidades_excel = wait.until(
                EC.presence_of_element_located((By.XPATH, "//a[@data-entity-type='file' and contains(@href, 'LISTA_UNIDADES.XLS')]"))
            )
            
            print("Found download link, clicking...")

        except Exception as e:
            print(f"Error finding download link: {str(e)}")
            raise e
        
        try:
            # Use JavaScript click as a fallback
            driver.execute_script("arguments[0].click();", listado_unidades_excel)
            print("Clicked download link")

        except Exception as e:
            print(f"Error clicking download link: {str(e)}")
            raise e
        
        # Wait for download to complete
        print("Waiting for download to complete...")
        time.sleep(5)
        
        # Get the downloaded file path
        downloaded_file = os.path.join(download_dir, "LISTA_UNIDADES.XLS")
        print(f"Downloaded to {downloaded_file}")
        
        return downloaded_file
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e
    
    finally:
        driver.quit()

def download_uofs_from_omie(download_dir: str) -> pd.DataFrame:
    """
    Download the UOFs from OMIE and convert them to a df. Data starts at row 4. And drops all empty cols.
    """
    excel_path = download_listado_unidades(download_dir)
    df = pd.read_excel(excel_path, header=3)
    df = df.dropna(axis=1, how='all')
    
    # Drop 'DESCRIPCIÓN' and 'PORCENTAJE \nPROPIEDAD' columns
    df = df.drop(columns=['DESCRIPCIÓN', 'PORCENTAJE \nPROPIEDAD'], errors='ignore')

    # Remove accent from 'TECNOLOGÍA'
    df = df.rename(columns={'TECNOLOGÍA': "tecnologia"})
    
    # Make all columns lower case
    df.columns = df.columns.str.lower()
    
    # Rename columns
    df = df.rename(columns={
        'codigo': 'UOF',
        'zona/frontera': 'zona',
        'agente propietario': 'agente_propietario',
        "tipo unidad": "tipo_unidad",
        "tecnologia": "tecnologia"
    })
    
    # Save as excel in download_dir
    df.to_excel(f"{download_dir}/listado_unidades.xlsx", index=False)

    #delete LISTA_UNIDADES.XLS
    os.remove(f"{download_dir}/LISTA_UNIDADES.XLS")

    return df

if __name__ == "__main__":
    download_dir = "/Users/jjcosta/Desktop/optimize/Energy_tracker_scripts/scripts/Spain/downloads/tracking"
    df = download_uofs_from_omie(download_dir) 