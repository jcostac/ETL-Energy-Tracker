from playwright.sync_api import sync_playwright
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization.pkcs12 import load_key_and_certificates
import os
import tempfile

class PariticpaEsiosBot:
    def __init__(self):
        """
        Initialize the PariticpaEsiosBot with default settings for client certificate, login URL, and temporary file tracking.
        """
        self._client_cert = {
            "certPath": None,  # Initialize as None; will be set later
            "keyPath": None
        }
        self._url = "https://participa.esios.ree.es/esiosqhws/login"
        self._temp_files = []  # Track temporary files for cleanup

    @property
    def client_cert(self):
        """
        Get the current client certificate and key file paths used for authentication.
        
        Returns:
            dict: Dictionary containing paths to the client certificate and key files.
        """
        return self._client_cert

    @client_cert.setter
    def client_cert(self, client_cert_paths: list[str]):
        """
        Set the client certificate and key file paths for authentication.
        
        Parameters:
            client_cert_paths (list[str]): List containing the certificate path and optionally the key path. The first element must be the certificate file path; the second element, if provided, is the key file path.
        
        Raises:
            ValueError: If the input is not a list of up to two paths.
            FileNotFoundError: If the certificate or key file does not exist.
        """

        if not isinstance(client_cert_paths, list) or len(client_cert_paths) > 2:
            raise ValueError("client_cert_paths must be a list of two paths: [cert_path, key_path]")
        
        if not os.path.exists(client_cert_paths[0]):
            raise FileNotFoundError(f"Certificate path not found: {client_cert_paths[0]}")
        
        if client_cert_paths[1] and not os.path.exists(client_cert_paths[1]):
            raise FileNotFoundError(f"Key path not found: {client_cert_paths[1]}")
        
        if client_cert_paths[1] is None: #if key path is not provided, do not set it
            self._client_cert = {"certPath": client_cert_paths[0]}
        else: #if key path is provided, set it
            self._client_cert = {"certPath": client_cert_paths[0], "keyPath": client_cert_paths[1]}

    @property
    def url(self):
        """
        Get the current URL configured for the ESIOS Participa login page.
        """
        return self._url

    @url.setter
    def url(self, url: str):
        """
        Set the target URL for the ESIOS Participa website.
        
        Raises:
            ValueError: If the provided URL is not a string or does not start with 'https://'.
        """
        if not isinstance(url, str) or not url.startswith("https://"):
            raise ValueError("URL must be a valid HTTPS string")
        self._url = url

    def _convert_p12_to_pem(self, certificate_path, certificate_password=None) -> tuple[str, str]:
        """
        Converts a PKCS#12 (.p12) certificate file to PEM format and writes the certificate and private key to temporary files.
        
        Parameters:
            certificate_path (str): Path to the .p12 certificate file.
            certificate_password (str, optional): Password for the .p12 file, if required.
        
        Returns:
            tuple[str, str]: The certificate and private key in PEM format as strings.
        
        Raises:
            ValueError: If the .p12 file cannot be loaded, typically due to an incorrect or missing password.
            Exception: For any other unexpected errors during conversion.
        """
        try:
            with open(certificate_path, "rb") as f:
                certificate_data = f.read()

            private_key, certificate, additional_certificates = load_key_and_certificates(
                certificate_data, certificate_password.encode("utf-8") if certificate_password else None
            )

            cert_pem = certificate.public_bytes(encoding=serialization.Encoding.PEM).decode("utf-8")
            key_pem = None
            if private_key:
                key_pem = private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ).decode("utf-8")

            # Create temporary files with the converted PEM content
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as cert_file:
                cert_file.write(cert_pem.encode("utf-8"))
                cert_path = cert_file.name
                self._temp_files.append(cert_path)  # Track for cleanup

            key_path = None
            if key_pem:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as key_file:
                    key_file.write(key_pem.encode("utf-8"))
                    key_path = key_file.name
                    self._temp_files.append(key_path)  # Track for cleanup

            self.client_cert = [cert_path, key_path] #set the client certificate and key paths as class attributes 

            return cert_pem, key_pem

        except ValueError as e:
            raise ValueError(f"Failed to load .p12 file: {e}. Check if the password is correct or missing.")
        except Exception as e:
            raise Exception(f"Unexpected error loading .p12 file: {e}")

    def _check_certificate_format(self, certificate_path, certificate_password=None) -> bool:
        """
        Determines if a certificate file is in PEM format, converting from .p12 to PEM if necessary.
        
        Returns:
            True if the certificate is already in PEM format or is successfully converted from .p12; False otherwise.
        """
        try:
            with open(certificate_path, "r") as f:
                content = f.read()
                if content.startswith("-----BEGIN CERTIFICATE-----"):
                    return True

            # If not PEM, assume it's .p12 and try to convert
            self._convert_p12_to_pem(certificate_path, certificate_password)
            return True
        except Exception as e:
            print(f"Error checking/converting certificate format: {e}")
            return False

    def login(self, client_cert_paths_lst: list[str], certificate_password: str = None):
        """
        Authenticate to the ESIOS Participa website using the provided client certificate and launch a Playwright browser session.
        
        Parameters:
            client_cert_paths_lst (list[str]): List containing paths to the client certificate and key files.
            certificate_password (str, optional): Password for the certificate file, if required.
        
        Returns:
            tuple: A tuple containing the Playwright page, context, and browser objects for further interaction and cleanup.
        
        Raises:
            ValueError: If the certificate format is invalid or conversion fails.
        """
        self.client_cert = client_cert_paths_lst #set the client certificate and key paths as class attributes 

        if not self._check_certificate_format(self._client_cert["certPath"], certificate_password):
            raise ValueError("Certificate format is not valid or conversion failed.")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            context = browser.new_context(
                client_certificates=[self._client_cert],
                ignore_https_errors=True
            )
            page = context.new_page()
            page.goto(self._url)
            return page, context, browser  # Return context and browser for proper cleanup

    def download_bsp_list(self, download_path=None):
        """
        Downloads the BSP LSI file from the ESIOS Participa website using client certificate authentication.
        
        If no download path is specified, the file is saved to the current working directory. Ensures cleanup of browser resources and any temporary certificate files after the download.
        
        Parameters:
            download_path (str, optional): Directory where the downloaded file will be saved.
        
        Returns:
            str: Full path to the downloaded BSP LSI file.
        
        Raises:
            ValueError: If the BSP LSI download link is not found on the page.
        """
        download_path = download_path or os.getcwd() #if no download path is provided, use the current directory
        if not os.path.exists(download_path):
            os.makedirs(download_path)

        page, context, browser = self.login()

        try:
            # Locate and click the BSP LSI link
            bsp_lsi_link = page.locator("text=BSP LSI")
            if not bsp_lsi_link.is_visible():
                raise ValueError("BSP LSI link not found on the page.")

            # Use expect_download to capture the download
            with page.expect_download() as download_info:
                bsp_lsi_link.click()
            download = download_info.value

            # Save the file to the specified path
            file_path = os.path.join(download_path, download.suggested_filename)
            download.save_as(file_path)

            print(f"File downloaded to: {file_path}")
            return file_path

        finally:
            # Clean up browser resources
            page.close()
            context.close()
            browser.close()

            # Clean up temporary certificate files
            for temp_file in self._temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            self._temp_files.clear()
            print("Cleaned up temporary certificate files.")