import json
import random
import time
from typing import List, Dict, Optional
import requests
from dotenv import load_dotenv
import os
load_dotenv()

class ProxyManager:
    def __init__(self, blacklist_timeout: int = 600):
        """
        Initializes the ProxyManager. Fetches proxies from Webshare API if a token is provided,
        otherwise attempts to load from proxies.json.

        Args:
            blacklist_timeout: seconds to keep a proxy blacklisted (default 600s = 10min)
            webshare_api_token: Your Webshare API token.
        """
        self.proxies: List[Dict[str, str]] = []
        self.webshare_api_token = os.getenv("WEBSHARE_API_KEY")

        if self.webshare_api_token:
            self.proxies = self._fetch_proxies_from_webshare_api()

            if not self.proxies: # If API fetch failed or returned empty list
                    print("API fetch failed or returned no proxies. Falling back to loading from proxies.json")
                    self._load_from_file()
        else:
            print("No Webshare API token provided. Loading from proxies.json")
            self._load_from_file()

        self.index = 0
        self.blacklist = {}  # {proxy_str: blacklist_until_timestamp}
        self.blacklist_timeout = blacklist_timeout # seconds to keep a proxy blacklisted (default 600s = 10min)

    def _fetch_proxies_from_webshare_api(self) -> List[Dict[str, str]]:
        """
        Fetches the proxy list from the Webshare API.

        Args:
            webshare_api_token: Your Webshare API token.

        Returns:
            A list of proxy dictionaries in the format {"ip_address": ..., "port": ...},
            or an empty list if an error occurs or no valid proxies are found.
        """
        headers = {"Authorization": f"Token {self.webshare_api_token}"}
        # Consider making page_size configurable or handling pagination for > 100 proxies
        api_url = "https://proxy.webshare.io/api/v2/proxy/list/?mode=direct&page=1&page_size=100"
        fetched_proxies = []

        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            data = response.json()

            # Transform the API response to the expected format
            fetched_proxies = [
                {"ip_address": p.get("proxy_address"), "port": p.get("port")}
                for p in data.get("results", [])
                if p.get("valid") and p.get("proxy_address") and p.get("port") # Ensure essential fields exist and proxy is valid
            ] 

            #fetched proxies format = [{'ip_address': '185.100.85.10', 'port': '8000'} etc.]
            print(f"Successfully fetched {len(fetched_proxies)} valid proxies from Webshare API.")

            # TODO: Implement pagination handling by checking data.get('next')
            # If data.get('next') is not None, make another request to that URL

        except requests.exceptions.RequestException as e:
            print(f"Error fetching proxies from Webshare API: {e}")
        except json.JSONDecodeError:
            print("Error decoding JSON response from Webshare API.")

        return fetched_proxies

    def _load_from_file(self):
        """Loads proxies from the local proxies.json file. This file contains a list of free proxies from https://proxycompass.com/es/free-proxy/"""
        try:
            # Ensure path is correct relative to where the script is run
            with open('utilidades/proxies.json', 'r') as f:
                loaded_data = json.load(f)
                # Basic validation if loading from file
                if isinstance(loaded_data, list):
                    self.proxies = [p for p in loaded_data if isinstance(p, dict) and 'ip_address' in p and 'port' in p]
                    print(f"Loaded {len(self.proxies)} proxies from utilidades/proxies.json")
                else:
                     print("Invalid format in proxies.json. Expected a list of dictionaries.")
                     self.proxies = []
        except FileNotFoundError:
            print("utilidades/proxies.json not found. No proxies loaded.")
            self.proxies = []
        except json.JSONDecodeError:
            print("Error decoding JSON from proxies.json. No proxies loaded.")
            self.proxies = []


    def _proxy_to_str(self, proxy: Dict[str, str]) -> str:
        """convert proxy dict to string
            Args:
                proxy: proxy dict
            Returns:
                proxy string in format "ip:port"
        """
        # Added check for missing keys just in case
        ip = proxy.get('ip_address', '')
        port = proxy.get('port', '')
        return f"{ip}:{port}"

    def _is_blacklisted(self, proxy: Dict[str, str]) -> bool:
        """check if proxy is blacklisted
            Args:
                proxy: proxy dict
            Returns:
                True if proxy is blacklisted (found in blacklist dict), False otherwise
        """
        proxy_str = self._proxy_to_str(proxy)
        if not proxy_str or ':' not in proxy_str: # Handle potential empty/invalid proxies
             return True # Treat invalid proxy string as blacklisted

        until = self.blacklist.get(proxy_str)
        if until is None:
            return False
        if time.time() > until:
            # Blacklist expired
            del self.blacklist[proxy_str]
            return False
        return True

    def get_next_proxy(self) -> Optional[Dict[str, str]]:
        """get next proxy from list of proxies
            Returns:
                next proxy from list of proxies in format {"ip_address": ..., "port": ...}
        """
        if not self.proxies:
            # print("Warning: Proxy list is empty.") # Optional: Add logging
            return None

        start_index = self.index
        list_len = len(self.proxies)

        for _ in range(list_len): # Iterate at most once through the list
            proxy = self.proxies[self.index]
            self.index = (self.index + 1) % list_len
            if not self._is_blacklisted(proxy):
                return proxy

        # If loop completes, all proxies are blacklisted
        # print("Warning: All proxies are currently blacklisted.") # Optional: Add logging
        return None

    def get_random_proxy(self) -> Optional[Dict[str, str]]:
        """get random proxy from list of proxies
            Returns:
                random proxy from list of proxies in format {"ip_address": ..., "port": ...}
        """
        available = [p for p in self.proxies if not self._is_blacklisted(p)]
        if not available:
            # print("Warning: No available (non-blacklisted) proxies.") # Optional: Add logging
            return None
        return random.choice(available)

    def format_proxy(self, proxy: Dict[str, str]) -> Optional[Dict[str, str]]:
        """format proxy to be used in requests
            Args:
                proxy: proxy dict with 'ip_address' and 'port' keys
            Returns:
                proxy dict with http and https keys, or None if input is invalid
        """
        ip = proxy.get('ip_address')
        port = proxy.get('port')

        if not ip or not port:
            print(f"Warning: Invalid proxy format for formatting: {proxy}")
            return None

        proxy_url = f"http://{ip}:{port}" # Assuming HTTP proxies, adjust if needed
        return {
            "http": proxy_url,
            "https": proxy_url # Often the same for HTTP proxies, check if HTTPS url needed
        }

    def mark_bad_proxy(self, proxy: Dict[str, str]):
        """mark proxy as bad
            Args:
                proxy: proxy dict
        """
        proxy_str = self._proxy_to_str(proxy)
        if proxy_str and ':' in proxy_str: # Only blacklist valid-looking proxies
            self.blacklist[proxy_str] = time.time() + self.blacklist_timeout
        else:
            print(f"Warning: Attempted to mark an invalid proxy as bad: {proxy}")


if __name__ == "__main__":

    proxy_manager = ProxyManager()

    print(f"Total proxies loaded: {len(proxy_manager.proxies)}")

    next_proxy = proxy_manager.get_next_proxy()
    if next_proxy:
        print("Next proxy:", next_proxy)
        formatted = proxy_manager.format_proxy(next_proxy)
        if formatted:
            print("Formatted proxy:", formatted)
        proxy_manager.mark_bad_proxy(next_proxy)
        print("Marked proxy as bad.")
        next_proxy_after_blacklist = proxy_manager.get_next_proxy()
        print("Next proxy after marking bad:", next_proxy_after_blacklist)
    else:
        print("Could not get a proxy (list empty or all blacklisted).")

    random_proxy = proxy_manager.get_random_proxy()
    if random_proxy:
        print("Random proxy:", random_proxy)
    else:
        print("Could not get a random proxy (list empty or all blacklisted).")
    
