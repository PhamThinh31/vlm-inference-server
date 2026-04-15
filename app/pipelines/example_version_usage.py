# app/pipelines/example_version_usage.py
"""
Example demonstrating how to use different prompt versions
"""

from .task1_classifier import Task1Classifier
from .task2_attributes import Task2AttributesExtractor
from .task3_validation import Task3Validator


# =============================================================================
# Example 1: Using v1.0 (default)
# =============================================================================

# Task 1 with v1.0
classifier_v1 = Task1Classifier(version="v1.0")

# Task 2 with v1.0
extractor_v1 = Task2AttributesExtractor(version="v1.0")

# Task 2 with v1.0 alternative format (no description)
extractor_v1_alt = Task2AttributesExtractor(version="v1.0", use_alt=True)

# Task 3 with v1.0
validator_v1 = Task3Validator(version="v1.0")


# =============================================================================
# Example 2: Using v1.1
# =============================================================================

# Task 1 with v1.1
classifier_v1_1 = Task1Classifier(version="v1.1")

# Task 2 with v1.1
extractor_v1_1 = Task2AttributesExtractor(version="v1.1")

# Task 3 with v1.1
validator_v1_1 = Task3Validator(version="v1.1")


# =============================================================================
# Example 3: Default usage (v1.0)
# =============================================================================

# Without specifying version, it defaults to v1.0
classifier_default = Task1Classifier()
extractor_default = Task2AttributesExtractor()
validator_default = Task3Validator()


# =============================================================================
# Example 4: Switching versions at runtime
# =============================================================================

def create_pipeline(version: str = "v1.0"):
    """
    Create a pipeline with specified version

    Args:
        version: Prompt version to use ('v1.0' or 'v1.1')

    Returns:
        Tuple of (classifier, extractor, validator)
    """
    classifier = Task1Classifier(version=version)
    extractor = Task2AttributesExtractor(version=version)
    validator = Task3Validator(version=version)

    return classifier, extractor, validator


# Usage:
# pipeline_v1 = create_pipeline("v1.0")
# pipeline_v1_1 = create_pipeline("v1.1")
