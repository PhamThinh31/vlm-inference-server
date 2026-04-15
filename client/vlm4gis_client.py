# client/vlm4gis_client.py
"""
VLM4GIS Python Client
Async client library for interacting with VLM4GIS server
"""

import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp


def _encode_image(path: str) -> Dict[str, str]:
    """Read a file from disk and return the client-side payload shape."""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return {"filename": Path(path).name, "data": data}


def _encode_pair(garment_path: str, body_path: str) -> Dict[str, Dict[str, str]]:
    return {"garment": _encode_image(garment_path), "body": _encode_image(body_path)}


logger = logging.getLogger(__name__)

@dataclass
class VLM4GISConfig:
    """Client configuration"""
    base_url: str = "http://0.0.0.0:8000"
    timeout: int = 300
    max_retries: int = 3

class VLM4GISClient:
    """Async client for VLM4GIS server"""

    def __init__(self, config: Optional[VLM4GISConfig] = None):
        self.config = config or VLM4GISConfig()
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(
            total=self.config.timeout))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def health_check(self) -> Dict[str, Any]:
        """Check server health"""
        async with self.session.get(f"{self.config.base_url}/health") as resp:
            return await resp.json()

    # ===== Session Management =====

    async def create_session(self,
                             name: Optional[str] = None,
                             metadata: Optional[Dict] = None) -> str:
        """Create a new processing session"""
        payload = {"session_name": name, "metadata": metadata or {}}

        async with self.session.post(f"{self.config.base_url}/session/create",
                                     json=payload) as resp:
            data = await resp.json()
            return data["session_id"]

    async def upload_images(self, session_id: str,
                            image_paths: List[str]) -> Dict[str, Any]:
        """Upload images to a session"""
        data = aiohttp.FormData()

        for path_str in image_paths:
            path = Path(path_str)
            if not path.exists():
                logger.warning(f"File not found: {path}")
                continue

            with open(path, 'rb') as f:
                data.add_field('files',
                               f.read(),
                               filename=path.name,
                               content_type='image/jpeg')

        async with self.session.post(
                f"{self.config.base_url}/session/{session_id}/upload",
                data=data) as resp:
            return await resp.json()

    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get session information"""
        async with self.session.get(
                f"{self.config.base_url}/session/{session_id}") as resp:
            return await resp.json()

    async def delete_session(self, session_id: str) -> Dict[str, Any]:
        """Delete a session"""
        async with self.session.delete(
                f"{self.config.base_url}/session/{session_id}") as resp:
            return await resp.json()

    # ===== Task Execution =====

    async def classify_images(
            self,
            session_id: str,
            image_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run Task 1: Image classification"""
        payload = {"session_id": session_id, "image_files": image_files}

        async with self.session.post(f"{self.config.base_url}/task/classify",
                                     json=payload) as resp:
            return await resp.json()

    async def extract_attributes(
            self,
            session_id: str,
            pairs: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Any]:
        """Run Task 2: Garment attributes extraction"""
        payload = {"session_id": session_id, "pairs": pairs}

        async with self.session.post(f"{self.config.base_url}/task/attributes",
                                     json=payload) as resp:
            return await resp.json()

    async def validate_pairs(self,
                             session_id: str,
                             pairs: Optional[List[Tuple[str, str]]] = None,
                             max_pairs: int = 10) -> Dict[str, Any]:
        """Run Task 3: Pair validation"""
        payload = {
            "session_id": session_id,
            "pairs": pairs,
            "max_pairs": max_pairs
        }

        async with self.session.post(f"{self.config.base_url}/task/validate",
                                     json=payload) as resp:
            return await resp.json()

    async def run_gis_pipeline(self,
                               session_id: str,
                               max_pairs_task2: int = 5,
                               max_pairs_task3: int = 10,
                               use_cache: bool = True) -> Dict[str, Any]:
        """Run complete GIS pipeline"""
        payload = {
            "session_id": session_id,
            "max_pairs_task2": max_pairs_task2,
            "max_pairs_task3": max_pairs_task3,
            "use_cache": use_cache
        }

        async with self.session.post(
                f"{self.config.base_url}/task/gis-pipeline",
                json=payload) as resp:
            return await resp.json()

    async def process_folder(
            self,
            folder_path: str,
            session_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process all images in a folder through the complete pipeline
        
        Args:
            folder_path: Path to folder containing images
            session_name: Optional session name
            
        Returns:
            Pipeline results dictionary
        """

        # Find all images in folder
        folder = Path(folder_path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        image_paths = [
            str(p) for p in folder.glob('*')
            if p.suffix.lower() in image_extensions
        ]

        if not image_paths:
            raise ValueError(f"No images found in {folder_path}")

        logger.info(f"Found {len(image_paths)} images in {folder_path}")

        # Create session
        session_id = await self.create_session(
            name=session_name or f"Folder: {folder.name}")

        try:
            # Upload images
            logger.info(f"Uploading images to session {session_id}")
            upload_result = await self.upload_images(session_id, image_paths)

            if upload_result["total_uploaded"] == 0:
                raise ValueError("No images were successfully uploaded")

            # Run GIS pipeline
            logger.info("Running GIS pipeline")
            pipeline_result = await self.run_gis_pipeline(session_id)

            return pipeline_result

        finally:
            # Clean up session
            await self.delete_session(session_id)

    ###
    ### Direct Inference and Server Path Methods Task 1
    ###
    #Task 1
    async def classify_images_direct(self,
                                     image_paths: List[str]) -> Dict[str, Any]:
        """Run Task 1 with direct image paths (base64 upload)."""
        payload = {"image_data_list": [_encode_image(p) for p in image_paths]}
        async with self.session.post(f"{self.config.base_url}/task/classify",
                                     json=payload) as resp:
            return await resp.json()

    async def classify_images_server_paths(
            self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Run Task 1 using images that already exist on the server filesystem.
        
        Args:
            image_paths: List of absolute paths on the SERVER machine.
        """
        payload = {"image_paths_list": image_paths}

        async with self.session.post(f"{self.config.base_url}/task/classify",
                                     json=payload) as resp:
            return await resp.json()

    #Task 2

    async def extract_attributes_direct(
            self, image_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Run Task 2 with direct image pairs (base64 upload)."""
        payload = {"image_data_pairs": [_encode_pair(g, b) for g, b in image_pairs]}
        async with self.session.post(f"{self.config.base_url}/task/attributes",
                                     json=payload) as resp:
            return await resp.json()

    async def extract_attributes_server_paths(
            self, image_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Run Task 2 using images that already exist on the server filesystem.
        
        Args:
            image_pairs: List of (garment_path, body_path) tuples on SERVER machine.
        """
        payload = {"image_pairs": image_pairs}

        async with self.session.post(f"{self.config.base_url}/task/attributes",
                                     json=payload) as resp:
            return await resp.json()
    #Task 3

    async def validate_pairs_direct(
            self, image_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Run Task 3 with direct image pairs (base64 upload)."""
        payload = {"image_data_pairs": [_encode_pair(g, b) for g, b in image_pairs]}
        async with self.session.post(f"{self.config.base_url}/task/validate",
                                     json=payload) as resp:
            result = await resp.json()
            if "detail" in result:
                logger.error("validate_pairs_direct failed: %s", result["detail"])
            return result

    async def validate_pairs_server_paths(
            self, image_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Run Task 3 using images that already exist on the server filesystem.
        
        Args:
            image_pairs: List of (garment_path, body_path) tuples on SERVER machine.
        """
        payload = {"image_pairs": image_pairs}

        async with self.session.post(f"{self.config.base_url}/task/validate",
                                     json=payload) as resp:
            return await resp.json()
# ===== Example Usage Functions =====

async def example_basic_usage():
    """Example: Basic usage"""

    async with VLM4GISClient() as client:
        # Check health
        health = await client.health_check()
        print(f"Server status: {health['status']}")

        # Create session
        session_id = await client.create_session(name="Test Session")
        print(f"Created session: {session_id}")

        # Upload images
        image_paths = [
            "images/garment1.jpg",
            "images/body1.jpg",
            "images/garment2.jpg",
            "images/body2.jpg"
        ]
        upload_result = await client.upload_images(session_id, image_paths)
        print(f"Uploaded {upload_result['total_uploaded']} images")

        # Run Task 1
        task1_result = await client.classify_images(session_id)
        print(f"Classified {task1_result['total_images']} images")

        # Run Task 2
        task2_result = await client.extract_attributes(session_id)
        print(f"Extracted attributes for {task2_result['pairs_processed']} pairs")

        # Run Task 3
        task3_result = await client.validate_pairs(session_id, max_pairs=10)
        print(f"Validated {task3_result['total_validated']} pairs")
        print(f"Valid: {task3_result['valid_count']}, Invalid: {task3_result['invalid_count']}")

        # Clean up
        await client.delete_session(session_id)

async def example_folder_processing():
    """Example: Process entire folder"""

    async with VLM4GISClient() as client:
        # Process entire folder
        result = await client.process_folder(
            folder_path="./test_images/sample_folder",
            session_name="Test Folder Processing"
        )
        print(json.dumps(result, indent=2))

        # Print summary
        print("\n=== GIS Pipeline Results ===")
        print(f"Session ID: {result['session_id']}")
        print(f"Total processing time: {result['total_processing_time']:.2f}s")
        print(f"Cache hits: {result['cache_hits']}")

        print("\n=== Summary ===")
        summary = result['summary']

        # Image statistics
        img_stats = summary['image_statistics']
        print(f"\nImages processed: {img_stats['total']}")
        print(f"  - Garments: {img_stats['garments']}")
        print(f"  - Bodies: {img_stats['bodies']}")
        print(f"  - Unknown: {img_stats['unknown']}")

        # Task 2 statistics
        task2_stats = summary['task2_statistics']
        print(f"\nTask 2 - Attributes extracted: {task2_stats['pairs_analyzed']}")
        print(f"Categories found: {task2_stats['categories']}")

        # Task 3 statistics
        task3_stats = summary['task3_statistics']
        print(f"\nTask 3 - Pairs validated: {task3_stats['pairs_validated']}")
        print(f"  - Valid: {task3_stats['valid_pairs']} ({task3_stats['validation_rate']:.1f}%)")
        print(f"  - Invalid: {task3_stats['invalid_pairs']}")

        # Best matches
        print("\nBest Matches:")
        for match in summary['best_matches'][:3]:
            print(f"  - {match['garment']} + {match['body']} (confidence: {match['confidence']:.2f})")

        # Performance
        perf = summary['performance']
        print(f"\nPerformance:")
        print(f"  - Task 1: {perf['task1_time']}")
        print(f"  - Task 2: {perf['task2_time']}")
        print(f"  - Task 3: {perf['task3_time']}")
        print(f"  - Total: {perf['total_time']}")

async def example_custom_pipeline():
    """Example: Custom pipeline with specific pairs"""

    async with VLM4GISClient() as client:
        # Create session and upload images
        session_id = await client.create_session(name="Custom Pipeline")

        image_paths = ["images/g1.jpg", "images/g2.jpg", "images/b1.jpg", "images/b2.jpg"]
        await client.upload_images(session_id, image_paths)

        # Run Task 1 to classify images
        task1_result = await client.classify_images(session_id)

        # Extract specific pairs for Task 2
        specific_pairs = [
            ("g1.jpg", "b1.jpg"),
            ("g2.jpg", "b2.jpg")
        ]

        task2_result = await client.extract_attributes(session_id, pairs=specific_pairs)

        # Validate all possible combinations
        task3_result = await client.validate_pairs(session_id, max_pairs=20)

        # Print results
        print(json.dumps(task3_result, indent=2))

        # Clean up
        await client.delete_session(session_id)

if __name__ == "__main__":
    # Run examples
    asyncio.run(example_folder_processing())
