

import requests
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

def scrape_url(url: str) -> str:
    """
    Fetches content from a URL and extracts clean text using BeautifulSoup.
    
    Args:
        url (str): The URL of the webpage to scrape.
        
    Returns:
        str: The extracted text content from the webpage.
        
    Raises:
        Exception: If the request fails or content cannot be parsed.
    """
    try:
        logger.info(f"Attempting to scrape URL: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
            
        # Get text and clean it up
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = '\n'.join(chunk for chunk in chunks if chunk)
        
        logger.info(f"Successfully scraped and cleaned text from {url}")
        return clean_text
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL {url}: {e}")
        raise Exception(f"Could not fetch content from URL: {url}") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred while scraping {url}: {e}")
        raise Exception("Failed to scrape and parse the webpage.") from e