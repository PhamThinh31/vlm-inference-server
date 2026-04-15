# app/utils/image_mapping.py
"""
Image name mapping utilities for clean inference
"""

import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ImageNameMapper:
    """Handle mapping between original and simplified names"""
    
    def __init__(self):
        self.original_to_simple: Dict[str, str] = {}
        self.simple_to_original: Dict[str, str] = {}
        self.counter = 0
    
    def reset(self):
        """Reset all mappings"""
        self.original_to_simple.clear()
        self.simple_to_original.clear()
        self.counter = 0
    
    def add_mapping(self, original_name: str) -> str:
        """
        Add a new mapping and return simplified name
        
        Args:
            original_name: Original filename
            
        Returns:
            Simplified filename (e.g., "1.png")
        """
        if original_name in self.original_to_simple:
            return self.original_to_simple[original_name]
        
        self.counter += 1
        # Keep original extension if it's an image
        ext = Path(original_name).suffix.lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
            ext = '.png'  # Default to png
        
        simple_name = f"{self.counter}{ext}"
        
        self.original_to_simple[original_name] = simple_name
        self.simple_to_original[simple_name] = original_name
        
        logger.debug(f"Mapped {original_name} -> {simple_name}")
        
        return simple_name
    
    def get_simple_name(self, original_name: str) -> Optional[str]:
        """Get simplified name for original"""
        return self.original_to_simple.get(original_name)
    
    def get_original_name(self, simple_name: str) -> Optional[str]:
        """Get original name from simplified"""
        # Try exact match first
        if simple_name in self.simple_to_original:
            return self.simple_to_original[simple_name]
        
        # Try without extension
        base_name = Path(simple_name).stem
        for simple, original in self.simple_to_original.items():
            if Path(simple).stem == base_name:
                return original
        
        return None
    
    def map_list(self, original_names: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """
        Map a list of names
        
        Returns:
            Tuple of (simple_names, mapping_dict)
        """
        simple_names = []
        mapping = {}
        
        for original in original_names:
            simple = self.add_mapping(original)
            simple_names.append(simple)
            mapping[simple] = original
        
        return simple_names, mapping
    
    def remap_results(self, results: List[Dict], key: str = 'filename') -> List[Dict]:
        """
        Remap results back to original names
        
        Args:
            results: List of result dictionaries
            key: Key containing the filename
            
        Returns:
            Results with original filenames
        """
        remapped = []
        
        for result in results:
            result_copy = result.copy()
            if key in result_copy:
                simple_name = result_copy[key]
                original_name = self.get_original_name(simple_name)
                if original_name:
                    result_copy[key] = original_name
                    result_copy['simple_name'] = simple_name  # Keep reference
            remapped.append(result_copy)
        
        return remapped