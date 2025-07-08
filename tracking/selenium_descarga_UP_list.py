import os
import glob
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import pretty_errors
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException

def setup_chrome_options(download_dir: str) -> Options:
    """
    Set up Chrome options for downloading files
    
    Args:
        download_dir (str): Directory path where files will be downloaded
        
    Returns:
        Options: Configured Chrome options
    """
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run in headless mode
    chrome_options.add_argument('--disable-gpu')  # Required for some systems
    chrome_options.add_argument('--no-sandbox')  # Required for some systems
    chrome_options.add_argument('--disable-dev-shm-usage')  # Required for some systems
    chrome_options.add_experimental_option("prefs", {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "directory_upgrade": True
    })
    return chrome_options

# Function to delete existing files matching the pattern
def delete_existing_files(pattern, directory):
    files_to_delete = glob.glob(os.path.join(directory, pattern))
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"🗑️ Deleted: {file_path}")
        except Exception as e:
            print(f"⚠️ Error deleting {file_path}: {e}")

# Function to wait for the new file and return its name
def wait_for_new_file(directory, file_pattern, timeout=30):
    end_time = time.time() + timeout
    while time.time() < end_time:
        # Ensure the directory path is correct
        print(f"Checking directory: {directory}")  # Debugging: Print directory path
        files = os.listdir(directory)  # This should be a valid directory path
        print(f"Files found: {files}")  # Debugging: Print found files
        files = glob.glob(os.path.join(directory, file_pattern))
        print(f"Files found: {files}")  # Debugging: Print found files
        if files:
            # Assuming the most recent file is the downloaded one
            latest_file = max(files, key=os.path.getctime)
            print(f"Latest file: {latest_file}")  # Debugging: Print latest file
            # Check if download is complete (e.g., no .crdownload extension)
            if not latest_file.endswith(".crdownload"):
                return latest_file
        time.sleep(1)
    return None

def descargador_UP_list(download_dir: str):
    """
    Descarga la lista de UPs desde el sitio web de ESIOS, como un CSV
    """
    # Initialize WebDriver
    service = Service()  # Ensure chromedriver is in PATH or specify path here
    chrome_options = setup_chrome_options(download_dir)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Step 1: Delete existing files before download
        file_pattern = "export_unidades-de-programacion*.csv"
        delete_existing_files(file_pattern, download_dir)

        # Step 2: Open the webpage
        driver.get("https://www.esios.ree.es/es/unidades-de-programacion")
        wait = WebDriverWait(driver, 15)

        # Step 3: Accept cookies by clicking "Permitir todas las cookies" or "Allow all cookies" if in english
        cookies_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Permitir todas las cookies') or contains(text(), 'Allow all cookies')]")))
        cookies_button.click()
        print("✅ Cookies accepted.")

        # Step 4: Wait for "EXPORTAR CSV" button
        try:
            export_button = wait.until(EC.visibility_of_element_located((By.LINK_TEXT, "EXPORTAR CSV")))
            time.sleep(1)
            wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "EXPORTAR CSV")))  # Ensure it's clickable
             # Use JavaScript to click the button
            driver.execute_script("arguments[0].click();", export_button)
            print("✅ 'EXPORTAR CSV' clicked, download should begin.")

        except TimeoutException:
            print("❌ 'EXPORTAR CSV' button not found. Please check the website.")
            raise
        except ElementClickInterceptedException:
            print("⚠️ 'EXPORTAR CSV' button is not clickable. There might be an overlay or another issue.")
            raise

        # Step 5: Wait for the file to appear
        downloaded_file = wait_for_new_file(download_dir, file_pattern, timeout=60)

        if downloaded_file:
            print(f"📥 Download complete: {os.path.basename(downloaded_file)}")
        else:
            print("❌ Download failed or timed out.")
            raise

    finally:
        # Close the browser
        driver.quit()

if __name__ == "__main__":
    download_dir = "/Users/jjcosta/Desktop/optimize/Energy_tracker_scripts/scripts/Spain/downloads/tracking"
    descargador_UP_list(download_dir) 