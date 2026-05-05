"""Data ingest script to load tours into vector database"""
from __future__ import annotations
import logging
import asyncio
from typing import Tuple
from odoo_client import OdooClient
from chunking import chunk_tour_data
from embeddings import get_embeddings_batch
from db import Database
import json

logger = logging.getLogger(__name__)


async def ingest_tours(db: Database, skip_existing: bool = True) -> Tuple[int, int]:
    """Fetch tours from Odoo, embed chunks, and upsert into vector storage."""
    try:
        client = OdooClient()
        tours = client.get_tours()
        
        if not tours:
            logger.warning("No tours found in Odoo")
            return 0, 0
        
        logger.info(f"Starting to ingest {len(tours)} tours")
        
        existing_tour_ids = set(db.get_all_tour_ids()) if skip_existing else set()
        
        tours_to_process = [t for t in tours if t.id not in existing_tour_ids]
        logger.info(f"Processing {len(tours_to_process)} tours (skipped {len(existing_tour_ids)} existing)")
        
        total_chunks = 0
        
        for tour in tours_to_process:
            try:
                if not skip_existing:
                    db.clear_embeddings(tour.id)
                
                chunks = chunk_tour_data(tour.name, tour.description, tour.detail_information)
                
                texts = [chunk['chunk_text'] for chunk in chunks]
                
                embeddings = get_embeddings_batch(texts)
                
                for chunk, embedding in zip(chunks, embeddings):
                    if embedding:
                        metadata = {
                            'category': tour.category,
                            'price': tour.price,
                            'currency': tour.currency,
                            'image_url': tour.image_url,
                            'created_at': tour.created_at,
                            'updated_at': tour.updated_at
                        }
                        
                        db.save_embedding(
                            tour_id=tour.id,
                            tour_name=tour.name,
                            chunk_index=chunk['chunk_index'],
                            chunk_text=chunk['chunk_text'],
                            embedding=embedding,
                            metadata=metadata
                        )
                        total_chunks += 1
                
                logger.info(f"Ingested tour {tour.id} ({tour.name}) with {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Error ingesting tour {tour.id}: {e}")
                continue
        
        logger.info(f"Ingest completed: {len(tours_to_process)} tours, {total_chunks} chunks")
        return len(tours_to_process), total_chunks
    except Exception as e:
        logger.error(f"Error in ingest_tours: {e}")
        raise


def run_ingest_sync():
    """CLI entry point for tour ingestion."""
    db = Database()
    try:
        tours_processed, chunks_created = asyncio.run(ingest_tours(db, skip_existing=True))
        print(f"\n✓ Ingest completed!")
        print(f"  Tours processed: {tours_processed}")
        print(f"  Chunks created: {chunks_created}")
    except Exception as e:
        print(f"\n✗ Ingest failed: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    run_ingest_sync()
