"""Odoo API client"""
from __future__ import annotations
import requests
import logging
from typing import List, Optional
from config import ODOO_URL
from models import TourData

logger = logging.getLogger(__name__)


class OdooClient:
    def __init__(self):
        self.base_url = ODOO_URL

    def get_tours(self) -> List[TourData]:
        """Fetch all tours from Odoo API"""
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
        """Fetch single tour by ID"""
        try:
            url = f"{self.base_url}/api/products/{tour_id}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if data.get('code') != 200:
                logger.error(f"Odoo API error: {data.get('message')}")
                return None
            
            # This endpoint returns product details, not the full tour data
            # We'll need to map it
            return None
        except Exception as e:
            logger.error(f"Error fetching tour {tour_id} from Odoo: {e}")
            return None
