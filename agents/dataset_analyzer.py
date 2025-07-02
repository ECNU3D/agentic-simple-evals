"""
Dataset Analyzer Agent

This module contains the DatasetAnalyzer class responsible for analyzing
HuggingFace datasets and determining their structure and content.
"""

import json
import re
from typing import Any, Dict, List, Optional
from datasets import load_dataset, get_dataset_config_names

from eval_types import SamplerBase
from data_models import DatasetAnalysis


class DatasetAnalyzer:
    """Agent for analyzing HuggingFace dataset structure and content"""
    
    def __init__(self, analyzer_sampler: SamplerBase):
        self.analyzer_sampler = analyzer_sampler
    
    def analyze_dataset(self, dataset_name: str, subset: Optional[str] = None, 
                       split: str = "test", max_samples: int = 10) -> DatasetAnalysis:
        """Analyze a HuggingFace dataset and return structured analysis"""
        
        requires_config = False
        actual_subset = subset
        
        # Try to load dataset and detect if config is required
        try:
            if subset:
                dataset = load_dataset(dataset_name, subset, split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)
        except Exception as e:
            # Check if error indicates missing config
            error_msg = str(e).lower()
            if "config name is missing" in error_msg or "pick one among" in error_msg:
                requires_config = True
                available_configs = []
                # Try to extract config names from error message first
                if "pick one among" in error_msg:
                    config_match = re.findall(r"'([^']+)'", str(e))
                    available_configs.extend(config_match)
                # Fallback: use huggingface hub to get config names
                if not available_configs:
                    try:
                        available_configs = get_dataset_config_names(dataset_name)
                    except Exception:
                        available_configs = []

                if available_configs:
                    # Choose the most appropriate config for Q&A tasks
                    preferred_configs = [
                        "question-answer",
                        "qa",
                        "test",
                        "train",
                        "validation",
                    ]
                    actual_subset = None
                    for preferred in preferred_configs:
                        for config in available_configs:
                            if preferred in config.lower():
                                actual_subset = config
                                break
                        if actual_subset:
                            break

                    if not actual_subset:
                        actual_subset = available_configs[0]

                    print(
                        f"📋 Dataset requires config. Using '{actual_subset}' from available: {available_configs}"
                    )
                    try:
                        dataset = load_dataset(dataset_name, actual_subset, split=split)
                    except Exception as retry_e:
                        # If split fails, try common split names
                        common_splits = ["test", "train", "validation", "dev"]
                        dataset = None
                        for test_split in common_splits:
                            try:
                                dataset = load_dataset(dataset_name, actual_subset, split=test_split)
                                print(
                                    f"   → Using split '{test_split}' instead of '{split}'"
                                )
                                split = test_split
                                break
                            except Exception:
                                continue
                        if dataset is None:
                            raise ValueError(
                                f"Could not load dataset {dataset_name} with config {actual_subset}: {retry_e}"
                            )
                else:
                    raise ValueError(
                        f"Dataset {dataset_name} requires config but none provided and no configs discovered. Error: {e}"
                    )
            else:
                # Try to auto-discover configs and splits using the hub as a last resort
                try:
                    config_names = get_dataset_config_names(dataset_name)
                except Exception:
                    config_names = []
                if config_names:
                    actual_subset = config_names[0]
                    try:
                        dataset = load_dataset(dataset_name, actual_subset, split=split)
                    except Exception as retry_e:
                        common_splits = ["test", "train", "validation", "dev"]
                        dataset = None
                        for test_split in common_splits:
                            try:
                                dataset = load_dataset(dataset_name, actual_subset, split=test_split)
                                print(
                                    f"   → Using split '{test_split}' instead of '{split}'"
                                )
                                split = test_split
                                break
                            except Exception:
                                continue
                        if dataset is None:
                            raise ValueError(
                                f"Could not load dataset {dataset_name} with auto-discovered config {actual_subset}: {retry_e}"
                            )
                else:
                    raise ValueError(f"Could not load dataset {dataset_name}: {e}")
        
        # Get basic info
        columns = dataset.column_names
        num_examples = len(dataset)
        sample_data = [dataset[i] for i in range(min(max_samples, num_examples))]
        
        # Use LLM to analyze the dataset structure
        analysis_prompt = self._create_analysis_prompt(dataset_name, columns, sample_data)
        analysis_response = self.analyzer_sampler([{"role": "user", "content": analysis_prompt}])
        
        # Parse the analysis response
        analysis_dict = self._parse_analysis_response(analysis_response)
        
        # Validate and fix column mappings
        validated_analysis = self._validate_column_mappings(analysis_dict, columns)
        
        return DatasetAnalysis(
            name=dataset_name,
            num_examples=num_examples,
            columns=columns,
            sample_data=sample_data,
            task_type=validated_analysis.get("task_type", "unknown"),
            input_columns=validated_analysis.get("input_columns", []),
            target_columns=validated_analysis.get("target_columns", []),
            metadata=validated_analysis.get("metadata", {}),
            subset=actual_subset,
            requires_config=requires_config
        )
    
    def _validate_column_mappings(self, analysis_dict: Dict[str, Any], actual_columns: List[str]) -> Dict[str, Any]:
        """Validate that column mappings exist in actual dataset"""
        
        # Ensure input_columns are valid
        input_cols = analysis_dict.get("input_columns", [])
        valid_input_cols = [col for col in input_cols if col in actual_columns]
        
        # If no valid input columns found, try to infer common ones
        if not valid_input_cols:
            common_input_names = [
                'question',
                'prompt',
                'input',
                'text',
                'context',
                'query',
                'source',
                'sentence',
            ]
            for name in common_input_names:
                if name in actual_columns:
                    valid_input_cols = [name]
                    break
        
        # Ensure target_columns are valid
        target_cols = analysis_dict.get("target_columns", [])
        valid_target_cols = [col for col in target_cols if col in actual_columns]
        
        # If no valid target columns found, try to infer common ones
        if not valid_target_cols:
            common_target_names = [
                'answer',
                'target',
                'label',
                'output',
                'solution',
                'reference',
                'ground_truth',
            ]
            for name in common_target_names:
                if name in actual_columns:
                    valid_target_cols = [name]
                    break
        
        # Update analysis with validated columns
        analysis_dict["input_columns"] = valid_input_cols
        analysis_dict["target_columns"] = valid_target_cols
        
        return analysis_dict
    
    def _create_analysis_prompt(self, dataset_name: str, columns: List[str], 
                               sample_data: List[Dict[str, Any]]) -> str:
        """Create a prompt for analyzing the dataset"""
        
        sample_str = "\n".join([f"Example {i+1}: {json.dumps(sample, indent=2)}" 
                               for i, sample in enumerate(sample_data[:5])])
        
        return f"""
Analyze the following HuggingFace dataset and determine its structure and task type.

Dataset Name: {dataset_name}
Columns: {columns}

Sample Data:
{sample_str}

Please analyze this dataset and respond with a JSON object containing:
1. "task_type": One of ["multiple_choice", "text_generation", "classification", "reading_comprehension", "code_generation", "math", "factual_qa", "translation", "summarization", "other"]
2. "input_columns": List of column names that contain the input/question
3. "target_columns": List of column names that contain the correct answer/target
4. "metadata": Object with additional relevant information like:
   - "choices_column": If multiple choice, which column has the choices
   - "context_column": If reading comprehension, which column has the context
   - "language": If multilingual
   - "domain": Subject domain (math, science, general, etc.)
   - "answer_format": How answers are formatted

Respond with only the JSON object, no additional text.
"""
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM analysis response"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback parsing
                return self._fallback_parse(response)
        except json.JSONDecodeError:
            return self._fallback_parse(response)
    
    def _fallback_parse(self, response: str) -> Dict[str, Any]:
        """Fallback parsing if JSON extraction fails"""
        # Simple heuristic-based parsing
        result = {
            "task_type": "other",
            "input_columns": [],
            "target_columns": [],
            "metadata": {}
        }
        
        # Look for common patterns
        if "multiple choice" in response.lower() or "choice" in response.lower():
            result["task_type"] = "multiple_choice"
        elif "classification" in response.lower():
            result["task_type"] = "classification"
        elif "math" in response.lower():
            result["task_type"] = "math"
        elif "code" in response.lower():
            result["task_type"] = "code_generation"
        elif "reading" in response.lower() or "comprehension" in response.lower():
            result["task_type"] = "reading_comprehension"
        elif "translation" in response.lower() or "translate" in response.lower():
            result["task_type"] = "translation"
        elif "summary" in response.lower() or "summarization" in response.lower():
            result["task_type"] = "summarization"
        
        return result 