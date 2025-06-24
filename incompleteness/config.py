from dataclasses import dataclass
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
import json

load_dotenv()

@dataclass
class ColumnConfig:
    severity: str = "Low"
    is_identifier: bool = False
    is_required: bool = True

@dataclass
class SystemConfig:
    name: str
    connection_string: str
    tables: List[str]
    completeness_threshold: float = 0.95
    column_configs: Dict[str, ColumnConfig] = None

@dataclass
class PipelineConfig:
    systems: Dict[str, SystemConfig]
    max_workers: int = 4

# Dynamic LLM-driven system configuration
# This will be populated based on LLM analysis of the dataset
DYNAMIC_SYSTEMS_CONFIG = {}

def create_dynamic_system_config(system_name: str, detected_columns: List[str], llm) -> SystemConfig:
    """Create a dynamic system configuration based on LLM analysis of the dataset"""
    
    # Use LLM to analyze columns and determine their importance
    prompt = f"""
    You are an expert pharmaceutical data analyst. Analyze these columns from a {system_name} system:
    {detected_columns}
    
    For each column, determine:
    1. Severity level (High/Medium/Low) for data completeness
    2. Whether it's an identifier column (True/False)
    3. Whether it's required (True/False)
    
    Return ONLY a JSON object with column configurations:
    {{
        "column_configs": {{
            "column_name": {{
                "severity": "High/Medium/Low",
                "is_identifier": true/false,
                "is_required": true/false
            }}
        }}
    }}
    """
    
    try:
        response = llm.invoke(prompt)
        response_content = response.content.strip()
        
        # Parse JSON response
        if response_content.startswith('```json'):
            response_content = response_content.replace('```json', '').replace('```', '').strip()
        elif response_content.startswith('```'):
            response_content = response_content.replace('```', '').strip()
        
        config_data = json.loads(response_content)
        
        # Convert to ColumnConfig objects
        column_configs = {}
        for col_name, config in config_data.get('column_configs', {}).items():
            if col_name in detected_columns:  # Only include columns that actually exist
                column_configs[col_name] = ColumnConfig(
                    severity=config.get('severity', 'Low'),
                    is_identifier=config.get('is_identifier', False),
                    is_required=config.get('is_required', True)
                )
        
        return SystemConfig(
            name=system_name,
            connection_string="",  # Not needed for CSV-based analysis
            tables=["dynamic_table"],
        completeness_threshold=0.95,
            column_configs=column_configs
        )
        
    except Exception as e:
        print(f"⚠️ LLM system config generation failed: {e}, using fallback")
        # Fallback: create basic config for all columns
        column_configs = {}
        for col in detected_columns:
            column_configs[col] = ColumnConfig(
                severity="Medium",
                is_identifier=col.lower().endswith('_id') or col.lower().endswith('id'),
                is_required=True
            )
        
        return SystemConfig(
            name=system_name,
            connection_string="",
            tables=["dynamic_table"],
            completeness_threshold=0.95,
            column_configs=column_configs
        )

# Incompleteness-specific configurations
INCOMPLETENESS_CONFIG = {
    "output_dir": "data_quality_reports/incompleteness",
    "report_format": "csv",
    "severity_weights": {
        "High": 1.0,
        "Medium": 0.7,
        "Low": 0.3
    }
}

def get_identifier_columns(system_config) -> list:
    """
    Return a list of column names that are marked as identifier columns in the given SystemConfig.
    """
    if hasattr(system_config, 'column_configs') and system_config.column_configs:
        return [col for col, config in system_config.column_configs.items() if getattr(config, 'is_identifier', False)]
    return [] 