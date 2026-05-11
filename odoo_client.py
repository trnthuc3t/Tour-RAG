"""Odoo API client"""
from __future__ import annotations
import requests
import logging
from typing import List, Optional, Generator
from config import ODOO_URL
from models import TourData

logger = logging.getLogger(__name__)


class OdooClient:
    def __init__(self):
        self.base_url = ODOO_URL

    def get_tours_paginated(self, limit: int = 50) -> Generator[TourData, None, None]:
        offset = 0
        page = 1
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        while True:
            try:
                # Use dedicated chunking endpoint (no images, minimal data)
                url = f"{self.base_url}/api/tours-for-chunking"
                params = {'limit': limit, 'offset': offset}
                
                logger.info(f"Fetching chunking page {page} (limit={limit}, offset={offset})")
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                if data.get('code') != 200:
                    error_msg = data.get('message', 'Unknown error')
                    logger.error(f"Odoo API error: {error_msg}")
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"Stopping after {consecutive_errors} consecutive errors")
                        break
                    break
                
                consecutive_errors = 0  # Reset on success
                
                tours_data = data.get('response', {}).get('tours', [])
                if not tours_data:
                    logger.info(f"No more tours after page {page - 1}")
                    break
                
                logger.info(f"Fetched chunking page {page}: {len(tours_data)} tours")
                
                # Yield each tour to avoid keeping all in RAM
                for tour_dict in tours_data:
                    try:
                        tour = TourData(**tour_dict)
                        yield tour
                    except Exception as e:
                        logger.warning(f"Error parsing tour: {e}")
                        continue
                
                # Check for more pages
                if not data.get('response', {}).get('has_more', False):
                    logger.info(f"Reached last page ({page})")
                    break
                
                offset += limit
                page += 1
                
            except Exception as e:
                logger.error(f"Error fetching chunking page {page}: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Stopping after {consecutive_errors} consecutive errors")
                    break

    def get_tours(self) -> List[TourData]:
        try:
            url = f"{self.base_url}/api/tours"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if data.get('code') != 200:
                logger.error(f"Odoo API error: {data.get('message')}")
                return []
            
            tours = []
            for tour_dict in data.get('response', {}).get('tours', []):
                tour = TourData(**tour_dict)
                tours.append(tour)
            
            logger.info(f"Fetched {len(tours)} tours from Odoo")
            return tours
        except Exception as e:
            logger.error(f"Error fetching tours from Odoo: {e}")
            return []

    def get_tour_by_id(self, tour_id: int) -> Optional[TourData]:
        """Placeholder for single-tour lookup when a dedicated endpoint exists."""
        try:
            url = f"{self.base_url}/api/products/{tour_id}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if data.get('code') != 200:
                logger.error(f"Odoo API error: {data.get('message')}")
                return None
            
            # Current endpoint shape does not match TourData yet.
            return None
        except Exception as e:
            logger.error(f"Error fetching tour {tour_id} from Odoo: {e}")
            return None
