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
    try:
        client = OdooClient()
        
        
        existing_tour_ids = set(db.get_all_tour_ids()) if skip_existing else set()
        
        tours_processed = 0
        total_chunks = 0
        failed_tours = []
        
        for tour in client.get_tours_paginated(limit=50):
            try:
                if skip_existing and tour.id in existing_tour_ids:
                    logger.debug(f"Skipping existing tour {tour.id}")
                    continue
                
                if not skip_existing:
                    db.clear_embeddings(tour.id)
                
                full_text = f"{tour.name}\n\n{tour.description}\n\n{tour.detail_information}".strip()
                
                # Chunk the tour data
                chunks = chunk_tour_data(tour.name, tour.description, tour.detail_information)
                
                if not chunks:
                    logger.warning(f"No chunks created for tour {tour.id}")
                    tours_processed += 1
                    continue
                
                # Extract chunk texts for embedding
                texts = [chunk['chunk_text'] for chunk in chunks]
                
                # Get embeddings in batches (batch size=20 to prevent API throttling)
                embeddings = get_embeddings_batch(texts)
                
                if not embeddings or len(embeddings) != len(chunks):
                    logger.warning(f"Embedding mismatch for tour {tour.id}: got {len(embeddings)} embeddings for {len(chunks)} chunks")
                    failed_tours.append(tour.id)
                    continue
                
                chunks_saved = 0
                for chunk, embedding in zip(chunks, embeddings):
                    if embedding and len(embedding) == 768:  # Validate embedding
                        metadata = {
                            'category': tour.category,
                            'price': tour.price,
                            'currency': tour.currency,
                            # Note: image_url is None/absent in chunking endpoint
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
                        chunks_saved += 1
                    else:
                        logger.warning(f"Invalid embedding for tour {tour.id} chunk {chunk['chunk_index']}")
                
                if chunks_saved > 0:
                    logger.info(f"✓ Tour {tour.id}: {tour.name[:50]} - {chunks_saved} chunks saved")
                    total_chunks += chunks_saved
                    tours_processed += 1
                else:
                    failed_tours.append(tour.id)
                
            except Exception as e:
                failed_tours.append(tour.id)
                continue
        
        if failed_tours:
            logger.warning(f"Failed tour IDs: {failed_tours}")
        
        return tours_processed, total_chunks
    except Exception as e:
        logger.error(f"Fatal error in ingest_tours: {e}", exc_info=True)
        raise


def run_ingest_sync():
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
