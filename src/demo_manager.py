"""
Modular Demo Manager for Audio Intelligence System

This module handles downloading, preprocessing, and caching of demo audio files
for the web application. It provides a clean interface for managing demo content
and ensures fast response times for users.
"""

import os
import json
import asyncio
import aiohttp
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class DemoFile:
    """Represents a demo audio file with metadata."""
    id: str
    display_name: str
    filename: str
    language: str
    description: str
    duration: str
    url: str
    local_path: Optional[str] = None
    processed: bool = False
    result_path: Optional[str] = None
    download_status: str = "pending"  # pending, downloading, completed, failed
    error_message: Optional[str] = None


class DemoManager:
    """
    Manages demo audio files including downloading, preprocessing, and caching.
    
    Features:
    - Automatic download of demo files from URLs
    - Background preprocessing for fast response
    - Caching of processed results
    - Error handling and retry logic
    - Configuration-driven file management
    """
    
    def __init__(self, config_path: str = "demo_config.json"):
        """
        Initialize the Demo Manager.
        
        Args:
            config_path (str): Path to demo configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.demo_files: Dict[str, DemoFile] = {}
        self.download_semaphore = asyncio.Semaphore(
            self.config["settings"]["max_concurrent_downloads"]
        )
        
        # Create directories
        self.demo_audio_dir = Path(self.config["settings"]["demo_audio_dir"])
        self.demo_results_dir = Path(self.config["settings"]["demo_results_dir"])
        self._ensure_directories()
        
        # Initialize demo files
        self._initialize_demo_files()
        
        logger.info(f"DemoManager initialized with {len(self.demo_files)} demo files")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load demo configuration from JSON file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Demo config loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load demo config: {e}")
            # Return default config
            return {
                "demo_files": [],
                "settings": {
                    "demo_audio_dir": "demo_audio",
                    "demo_results_dir": "demo_results",
                    "auto_preprocess": True,
                    "max_concurrent_downloads": 2,
                    "download_timeout": 300
                }
            }
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        self.demo_audio_dir.mkdir(exist_ok=True)
        self.demo_results_dir.mkdir(exist_ok=True)
        logger.debug(f"Directories ensured: {self.demo_audio_dir}, {self.demo_results_dir}")
    
    def _initialize_demo_files(self):
        """Initialize DemoFile objects from configuration."""
        for file_config in self.config["demo_files"]:
            demo_file = DemoFile(
                id=file_config["id"],
                display_name=file_config["display_name"],
                filename=file_config["filename"],
                language=file_config["language"],
                description=file_config["description"],
                duration=file_config["duration"],
                url=file_config["url"]
            )
            
            # Check if file exists locally
            local_path = self.demo_audio_dir / file_config["filename"]
            if local_path.exists():
                demo_file.local_path = str(local_path)
                demo_file.download_status = "completed"
                
                # Check if already processed
                result_path = self.demo_results_dir / f"{file_config['id']}_results.json"
                if result_path.exists():
                    demo_file.processed = True
                    demo_file.result_path = str(result_path)
            
            self.demo_files[demo_file.id] = demo_file
    
    async def download_all_demo_files(self) -> Dict[str, str]:
        """
        Download all demo files that don't exist locally.
        
        Returns:
            Dict[str, str]: Mapping of file ID to download status
        """
        download_tasks = []
        
        for demo_file in self.demo_files.values():
            if demo_file.download_status != "completed":
                task = self._download_demo_file(demo_file)
                download_tasks.append(task)
        
        if download_tasks:
            logger.info(f"Starting download of {len(download_tasks)} demo files")
            results = await asyncio.gather(*download_tasks, return_exceptions=True)
            
            # Process results
            status_map = {}
            for demo_file, result in zip([f for f in self.demo_files.values() if f.download_status != "completed"], results):
                if isinstance(result, Exception):
                    demo_file.download_status = "failed"
                    demo_file.error_message = str(result)
                    status_map[demo_file.id] = "failed"
                    logger.error(f"Download failed for {demo_file.id}: {result}")
                else:
                    status_map[demo_file.id] = "completed"
            
            return status_map
        
        return {file_id: "already_exists" for file_id in self.demo_files.keys()}
    
    async def _download_demo_file(self, demo_file: DemoFile) -> str:
        """
        Download a single demo file or check if local file exists.
        
        Args:
            demo_file (DemoFile): Demo file to download
            
        Returns:
            str: Download status
        """
        async with self.download_semaphore:
            try:
                # Check if it's a local file (already exists)
                if demo_file.url == "local":
                    local_path = self.demo_audio_dir / demo_file.filename
                    if local_path.exists():
                        demo_file.local_path = str(local_path)
                        demo_file.download_status = "completed"
                        demo_file.error_message = None
                        logger.info(f"✅ Local file found: {demo_file.filename}")
                        return "completed"
                    else:
                        raise Exception(f"Local file not found: {local_path}")
                
                demo_file.download_status = "downloading"
                logger.info(f"Downloading {demo_file.filename} from {demo_file.url}")
                
                timeout = aiohttp.ClientTimeout(total=self.config["settings"]["download_timeout"])
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(demo_file.url) as response:
                        if response.status == 200:
                            # Save file
                            local_path = self.demo_audio_dir / demo_file.filename
                            with open(local_path, 'wb') as f:
                                async for chunk in response.content.iter_chunked(8192):
                                    f.write(chunk)
                            
                            demo_file.local_path = str(local_path)
                            demo_file.download_status = "completed"
                            demo_file.error_message = None
                            
                            logger.info(f"Successfully downloaded {demo_file.filename}")
                            return "completed"
                        else:
                            raise Exception(f"HTTP {response.status}: {response.reason}")
                            
            except Exception as e:
                demo_file.download_status = "failed"
                demo_file.error_message = str(e)
                logger.error(f"Failed to download {demo_file.filename}: {e}")
                raise
    
    def get_demo_file_info(self, file_id: str) -> Optional[DemoFile]:
        """Get information about a specific demo file."""
        return self.demo_files.get(file_id)
    
    def get_all_demo_files(self) -> List[DemoFile]:
        """Get all demo files."""
        return list(self.demo_files.values())
    
    def get_available_demo_files(self) -> List[DemoFile]:
        """Get demo files that are available for processing."""
        return [f for f in self.demo_files.values() if f.download_status == "completed"]
    
    def get_processed_demo_files(self) -> List[DemoFile]:
        """Get demo files that have been processed."""
        return [f for f in self.demo_files.values() if f.processed]
    
    def mark_as_processed(self, file_id: str, result_path: str):
        """Mark a demo file as processed."""
        if file_id in self.demo_files:
            self.demo_files[file_id].processed = True
            self.demo_files[file_id].result_path = result_path
            logger.info(f"Marked {file_id} as processed")
    
    def get_demo_file_path(self, file_id: str) -> Optional[str]:
        """Get the local path of a demo file."""
        demo_file = self.demo_files.get(file_id)
        return demo_file.local_path if demo_file else None
    
    def get_demo_result_path(self, file_id: str) -> Optional[str]:
        """Get the result path of a processed demo file."""
        demo_file = self.demo_files.get(file_id)
        return demo_file.result_path if demo_file else None
    
    def get_demo_file_by_filename(self, filename: str) -> Optional[DemoFile]:
        """Find a demo file by its filename."""
        for demo_file in self.demo_files.values():
            if demo_file.filename == filename:
                return demo_file
        return None
    
    def get_demo_files_by_language(self, language: str) -> List[DemoFile]:
        """Get demo files filtered by language."""
        return [f for f in self.demo_files.values() if f.language == language]
    
    def get_download_status_summary(self) -> Dict[str, int]:
        """Get a summary of download statuses."""
        statuses = {}
        for demo_file in self.demo_files.values():
            status = demo_file.download_status
            statuses[status] = statuses.get(status, 0) + 1
        return statuses
    
    def get_processing_status_summary(self) -> Dict[str, int]:
        """Get a summary of processing statuses."""
        total = len(self.demo_files)
        processed = len(self.get_processed_demo_files())
        available = len(self.get_available_demo_files())
        
        return {
            "total": total,
            "processed": processed,
            "available": available,
            "pending": total - available
        }
    
    def cleanup_failed_downloads(self):
        """Remove failed download entries and reset status."""
        for demo_file in self.demo_files.values():
            if demo_file.download_status == "failed":
                demo_file.download_status = "pending"
                demo_file.error_message = None
                logger.info(f"Reset download status for {demo_file.id}")
    
    def validate_file_integrity(self, file_id: str) -> bool:
        """
        Validate that a downloaded file is not corrupted.
        
        Args:
            file_id (str): ID of the demo file to validate
            
        Returns:
            bool: True if file is valid, False otherwise
        """
        demo_file = self.demo_files.get(file_id)
        if not demo_file or not demo_file.local_path:
            return False
        
        try:
            local_path = Path(demo_file.local_path)
            if not local_path.exists():
                return False
            
            # Basic file size check (should be > 1KB for audio files)
            if local_path.stat().st_size < 1024:
                logger.warning(f"File {file_id} is too small, may be corrupted")
                return False
            
            # Check file extension
            valid_extensions = {'.mp3', '.wav', '.ogg', '.m4a', '.flac'}
            if local_path.suffix.lower() not in valid_extensions:
                logger.warning(f"File {file_id} has invalid extension: {local_path.suffix}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating file {file_id}: {e}")
            return False
    
    def get_demo_file_metadata(self, file_id: str) -> Dict[str, Any]:
        """
        Get comprehensive metadata for a demo file.
        
        Args:
            file_id (str): ID of the demo file
            
        Returns:
            Dict[str, Any]: File metadata
        """
        demo_file = self.demo_files.get(file_id)
        if not demo_file:
            return {}
        
        metadata = {
            "id": demo_file.id,
            "display_name": demo_file.display_name,
            "filename": demo_file.filename,
            "language": demo_file.language,
            "description": demo_file.description,
            "duration": demo_file.duration,
            "url": demo_file.url,
            "local_path": demo_file.local_path,
            "processed": demo_file.processed,
            "result_path": demo_file.result_path,
            "download_status": demo_file.download_status,
            "error_message": demo_file.error_message
        }
        
        # Add file size if available
        if demo_file.local_path and Path(demo_file.local_path).exists():
            try:
                file_size = Path(demo_file.local_path).stat().st_size
                metadata["file_size_bytes"] = file_size
                metadata["file_size_mb"] = round(file_size / (1024 * 1024), 2)
            except Exception:
                pass
        
        return metadata
    
    def export_config(self, output_path: str = None):
        """
        Export current demo configuration to JSON file.
        
        Args:
            output_path (str, optional): Output file path
        """
        if output_path is None:
            output_path = f"demo_config_export_{int(time.time())}.json"
        
        export_data = {
            "demo_files": [],
            "settings": self.config["settings"]
        }
        
        for demo_file in self.demo_files.values():
            export_data["demo_files"].append({
                "id": demo_file.id,
                "display_name": demo_file.display_name,
                "filename": demo_file.filename,
                "language": demo_file.language,
                "description": demo_file.description,
                "duration": demo_file.duration,
                "url": demo_file.url
            })
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Demo configuration exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export demo configuration: {e}")


# Convenience functions for easy usage
def create_demo_manager(config_path: str = "demo_config.json") -> DemoManager:
    """Create and return a DemoManager instance."""
    return DemoManager(config_path)


async def download_demo_files(config_path: str = "demo_config.json") -> Dict[str, str]:
    """Download all demo files from a configuration."""
    manager = DemoManager(config_path)
    return await manager.download_all_demo_files()


if __name__ == "__main__":
    # Test the demo manager
    async def test():
        manager = DemoManager()
        print(f"Initialized with {len(manager.demo_files)} demo files")
        
        # Download files
        results = await manager.download_all_demo_files()
        print(f"Download results: {results}")
        
        # Show status
        print(f"Download status: {manager.get_download_status_summary()}")
        print(f"Processing status: {manager.get_processing_status_summary()}")
    
    asyncio.run(test())
