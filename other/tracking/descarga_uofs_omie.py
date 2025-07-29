import asyncio
from playwright.async_api import async_playwright
import pretty_errors
import pandas as pd

async def download_listado_unidades(download_dir: str):
    """
    Asynchronously downloads the 'Listado de unidades' Excel file from the OMIE website and saves it to the specified directory.
    
    Parameters:
        download_dir (str): Directory path where the downloaded Excel file will be saved.
    
    Returns:
        str: Full path to the saved Excel file.
    """
    url = "https://www.omie.es/es/listado-de-agentes"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        # Go to the page
        await page.goto(url)

        # Find the "Listado de unidades" Excel link and click it
        # The second "aquí" link under "Listado de unidades" is for Excel
        # This selector finds the link with text "aquí" that is after "Listado de unidades" 
        listado_unidades_excel = page.locator("text=Listado de unidades").locator("xpath=following::a[1]")

        # Wait for the download to start after clicking
        async with page.expect_download() as download_info:
            await listado_unidades_excel.click()
        download = await download_info.value

        # Save the file to the specified directory
        await download.save_as(f"{download_dir}/listado_unidades.xlsx")

        print(f"Downloaded to {download_dir}/listado_unidades.xlsx")
        await browser.close()

        return f"{download_dir}/listado_unidades.xlsx"
    

async def download_uofs_from_omie(download_dir: str) -> pd.DataFrame:
    """
    Download and process the OMIE UOFs Excel file, returning a cleaned pandas DataFrame.
    
    The function downloads the "Listado de unidades" Excel file from OMIE, reads its data starting from the fourth row, removes empty columns and specific unwanted columns, standardizes column names, and saves the cleaned data back to Excel in the specified directory.
    
    Parameters:
        download_dir (str): Directory where the Excel file will be downloaded and saved.
    
    Returns:
        pd.DataFrame: The processed DataFrame containing UOFs data.
    """
    excel_path = await download_listado_unidades(download_dir)
    df = pd.read_excel(excel_path, header=3)
    df = df.dropna(axis=1, how='all')
    
    # Drop 'DESCRIPCIÓN' and 'PORCENTAJE \nPROPIEDAD' columns
    df = df.drop(columns=['DESCRIPCIÓN', 'PORCENTAJE \nPROPIEDAD'], errors='ignore')

    # Remove accent from 'TECNOLOGÍA'
    df = df.rename(columns={'TECNOLOGÍA': "tecnologia"})
    
    # Make all columns lower case
    df.columns = df.columns.str.lower()
    
    # Rename 'CODIGO' to 'uof' and 'ZONA/FRONTERA' to 'zona'
    df = df.rename(columns={'codigo': 'UOF', 'zona/frontera': 'zona', 'agente propietario': 'agente_propietario', "tipo unidad": "tipo_unidad", "tecnologia": "tecnologia"})
    
    #save as csv in download_dir
    df.to_excel(f"{download_dir}/listado_unidades.xlsx", index=False)

    return df
    
    

if __name__ == "__main__":
    download_dir = "C:/Users/Usuario/OneDrive - OPTIMIZE ENERGY/Escritorio/Optimize Energy/Energy_tracker_scripts/scripts/Spain/data"
    df = asyncio.run(download_uofs_from_omie(download_dir))