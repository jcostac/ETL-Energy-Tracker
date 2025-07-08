import os
import glob
import time
import asyncio # Required for Playwright's async nature
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError
import pretty_errors

# Function to delete existing files matching the pattern
def delete_existing_files(pattern, directory):
    files_to_delete = glob.glob(os.path.join(directory, pattern))
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"🗑️ Deleted: {file_path}")
        except Exception as e:
            print(f"⚠️ Error deleting {file_path}: {e}")

async def descargador_UP_list(download_dir: str):
    """
    Descarga la lista de UPs desde el sitio web de ESIOS, como un CSV, usando Playwright.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True) # You can set headless=False for debugging
        # Create a context that will automatically accept downloads
        # and save them to the specified directory.
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        try:
            # Step 1: Delete existing files before download
            file_pattern = "export_unidades-de-programacion*.csv"
            delete_existing_files(file_pattern, download_dir)

            # Step 2: Open the webpage
            await page.goto("https://www.esios.ree.es/es/unidades-de-programacion", wait_until="networkidle")
            
            # Step 3: Accept cookies
            try:
                # Using a more robust selector that handles both Spanish and English text
                cookie_button_selector = "button:has-text('Permitir todas las cookies'), button:has-text('Allow all cookies')"
                await page.wait_for_selector(cookie_button_selector, timeout=15000)
                await page.click(cookie_button_selector)
                print("✅ Cookies accepted.")
            except PlaywrightTimeoutError:
                print("⚠️ Cookie consent button not found or timed out. Assuming cookies are already accepted or not present.")

            # Step 4: Click "EXPORTAR CSV" button and handle download
            try:
                # Start waiting for the download before clicking the button
                async with page.expect_download(timeout=60000) as download_info:
                    export_button_selector = "a:has-text('EXPORTAR CSV')"
                    await page.wait_for_selector(export_button_selector, timeout=15000, state="visible")
                    await page.click(export_button_selector)
                
                download = await download_info.value
                
                # Construct the full path for saving the file
                # Playwright downloads to a temporary path first
                # We then save it to our desired location with the original name
                original_filename = download.suggested_filename
                download_path = os.path.join(download_dir, original_filename)
                await download.save_as(download_path)

                if os.path.exists(download_path) and not download_path.endswith(".crdownload"):
                     print(f"📥 Download complete: {os.path.basename(download_path)}")
                else:
                    print(f"❌ Download failed or file not fully saved to {download_path}.")
                    # Check if a temp file still exists, indicating an issue
                    temp_file = await download.path()
                    if temp_file and os.path.exists(temp_file):
                        print(f"⚠️ Temporary download file exists at: {temp_file}")
                    raise PlaywrightError("Download failed or timed out.")

            except PlaywrightTimeoutError:
                print("❌ 'EXPORTAR CSV' button not found or interaction timed out. Please check the website.")
                raise
            except PlaywrightError as e: # Catching generic Playwright errors for click/download issues
                print(f"⚠️ An error occurred while trying to click 'EXPORTAR CSV' or during download: {e}")
                raise

        finally:
            # Close the browser
            await browser.close()

async def main():
    # Ensure download_dir is an absolute path and exists
    # Using a relative path for the example, adjust as needed.
    # For your original path: "C:\\Users\\Usuario\\OneDrive - OPTIMIZE ENERGY\\Escritorio\\Optimize Energy\\Energy_tracker_scripts\\scripts\\Spain\\data"
    # It's better to construct paths using os.path.join for cross-platform compatibility
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "downloads")) # Example: navigating up to a 'data' folder
    # If you want to keep your specific Windows path:
    # download_dir = r"C:\Users\Usuario\OneDrive - OPTIMIZE ENERGY\Escritorio\Optimize Energy\Energy_tracker_scripts\scripts\Spain\data"
    
    # For this example, let's assume 'data' directory is at the same level as 'Spain'
    # scripts/Spain/data
    download_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "downloads", "tracking")
    
    if not os.path.exists(download_dir):
        os.makedirs(download_dir, exist_ok=True)
        print(f"📂 Created download directory: {download_dir}")
    
    await descargador_UP_list(download_dir)

if __name__ == "__main__":
    asyncio.run(main()) 

