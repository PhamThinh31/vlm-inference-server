# app/pipelines/prompt_templates.py
"""
Centralized prompt templates for all VLM4GIS tasks
Supports versioning for easy prompt management and experimentation
"""

class PromptTemplates:
    """Centralized storage for all prompt templates with versioning"""

    # =============================================================================
    # TASK 1: Image Type Classification
    # =============================================================================

    TASK1_V1_0 = """<image>
Images: {image_list}

[Task 1 v1.0 prompt — classify each image into predefined categories.
[To keep the sensitive information from private project] Replace this with your own classification prompt and output schema.]"""

    TASK1_V1_1 = """<image>
Images: {image_list}

[Task 1 v1.1 prompt — improved version of classification prompt.
[To keep the sensitive information from private project] Replace this with your own classification prompt and output schema.]"""

    # =============================================================================
    # TASK 2: Garment Attributes Extraction
    # =============================================================================

    TASK2_V1_0 = """<image>
<image>
Images: {garment_file}, {body_file}

[Task 2 v1.0 prompt — extract structured attributes from image pair.
[To keep the sensitive information from private project] Replace this with your own attribute extraction prompt and output schema.]"""

    TASK2_V1_1 = """<image>
<image>
Images: {garment_file}, {body_file}

[Task 2 v1.1 prompt — improved attribute extraction.
[To keep the sensitive information from private project] Replace this with your own attribute extraction prompt and output schema.]"""

    TASK2_V1_0_ALT = """<image>
<image>
Images: {garment_file}, {body_file}

[Task 2 v1.0 alternative format — shorter output variant.
[To keep the sensitive information from private project] Replace this with your own attribute extraction prompt and output schema.]"""

    # =============================================================================
    # TASK 3: Pair Validation
    # =============================================================================

    TASK3_V1_0 = """<image>
<image>
Images: {garment_file}, {body_file}

[Task 3 v1.0 prompt — validate whether image pair meets quality criteria.
[To keep the sensitive information from private project] Replace this with your own validation prompt.
Output format: Yes or No]"""

    TASK3_V1_1 = """<image>
<image>
Images: {garment_file}, {body_file}

[Task 3 v1.1 prompt — extended validation with quality scoring.
[To keep the sensitive information from private project] Replace this with your own validation prompt.
Output format:
Pose Quality: [Integer 0-100]
Valid Pair: [Yes/No]]"""

    # =============================================================================
    # Version Selection
    # =============================================================================

    @classmethod
    def get_task1_prompt(cls, version: str = "v1.0") -> str:
        """Get Task 1 prompt by version"""
        if version == "v1.0":
            return cls.TASK1_V1_0
        elif version == "v1.1":
            return cls.TASK1_V1_1
        else:
            raise ValueError(f"Unknown version: {version}")

    @classmethod
    def get_task2_prompt(cls, version: str = "v1.0", use_alt: bool = False) -> str:
        """Get Task 2 prompt by version"""
        if version == "v1.0":
            return cls.TASK2_V1_0_ALT if use_alt else cls.TASK2_V1_0
        elif version == "v1.1":
            return cls.TASK2_V1_1
        else:
            raise ValueError(f"Unknown version: {version}")

    @classmethod
    def get_task3_prompt(cls, version: str = "v1.0") -> str:
        """Get Task 3 prompt by version"""
        if version == "v1.0":
            return cls.TASK3_V1_0
        elif version == "v1.1":
            return cls.TASK3_V1_1
        else:
            raise ValueError(f"Unknown version: {version}")


# =============================================================================
# Convenience Functions
# =============================================================================

def get_prompt(task: str, version: str = "v1.0", **kwargs) -> str:
    """
    Get a prompt template for a specific task and version

    Args:
        task: Task name ('task1', 'task2', 'task3')
        version: Version string ('v1.0', 'v1.1')
        **kwargs: Additional arguments (e.g., use_alt=True for task2)

    Returns:
        Prompt template string

    Example:
        >>> prompt = get_prompt('task1', version='v1.0')
        >>> prompt = get_prompt('task2', version='v1.1')
    """
    task = task.lower()

    if task == 'task1':
        return PromptTemplates.get_task1_prompt(version)
    elif task == 'task2':
        return PromptTemplates.get_task2_prompt(version, **kwargs)
    elif task == 'task3':
        return PromptTemplates.get_task3_prompt(version)
    else:
        raise ValueError(f"Unknown task: {task}")