"""
Task Type Detector Agent

This module contains the TaskTypeDetector class responsible for determining
the specific evaluation approach based on task type.
"""

from eval_types import SamplerBase
from data_models import DatasetAnalysis


class TaskTypeDetector:
    """Agent for determining the specific evaluation approach based on task type"""
    
    def __init__(self, detector_sampler: SamplerBase):
        self.detector_sampler = detector_sampler
    
    def determine_eval_approach(self, analysis: DatasetAnalysis) -> str:
        """Determine which existing evaluation approach to base the new eval on"""
        
        # Map task types to existing evaluation templates
        type_mapping = {
            "multiple_choice": "mmlu_style",
            "math": "math_style",
            "classification": "mmlu_style",  # Can be adapted
            "reading_comprehension": "drop_style",
            "code_generation": "humaneval_style",
            "factual_qa": "simpleqa_style",
            "text_generation": "simpleqa_style",
            "translation": "simpleqa_style",
            "summarization": "simpleqa_style",
        }
        
        return type_mapping.get(analysis.task_type, "generic_style") 