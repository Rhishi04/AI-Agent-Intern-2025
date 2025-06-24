import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import json

@dataclass
class ColumnConfig:
    is_identifier: bool = False

@dataclass
class SystemConfig:
    name: str
    connection_string: str
    tables: List[str]
    consistency_threshold: float = 0.90
    column_configs: Dict[str, ColumnConfig] = None

@dataclass
class PipelineConfig:
    systems: Dict[str, SystemConfig]
    max_workers: int = 4

# Dynamic LLM-driven system configuration
# This will be populated based on LLM analysis of the dataset
DYNAMIC_SYSTEMS_CONFIG = {}

# Dynamic field mappings - will be generated based on LLM analysis
DYNAMIC_FIELD_MAPPINGS = []

def create_dynamic_system_config(system_name: str, detected_columns: List[str], llm) -> SystemConfig:
    """Create a dynamic system configuration based on LLM analysis of the dataset"""
    
    # Use LLM to analyze columns and determine which are identifiers
    prompt = f"""
    You are an expert pharmaceutical data analyst. Analyze these columns from a {system_name} system:
    {detected_columns}
    
    For each column, determine if it's an identifier column (True/False).
    Identifier columns are typically unique keys, IDs, or primary identifiers.
    
    Return ONLY a JSON object with column configurations:
    {{
        "column_configs": {{
            "column_name": {{
                "is_identifier": true/false
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
                    is_identifier=config.get('is_identifier', False)
                )
        
        return SystemConfig(
            name=system_name,
            connection_string="",  # Not needed for CSV-based analysis
            tables=["dynamic_table"],
        consistency_threshold=0.90,
            column_configs=column_configs
        )
        
    except Exception as e:
        print(f"⚠️ LLM system config generation failed: {e}, using fallback")
        # Fallback: create basic config for all columns
        column_configs = {}
        for col in detected_columns:
            column_configs[col] = ColumnConfig(
                is_identifier=col.lower().endswith('_id') or col.lower().endswith('id')
            )
        
        return SystemConfig(
            name=system_name,
            connection_string="",
            tables=["dynamic_table"],
        consistency_threshold=0.90,
            column_configs=column_configs
        )

def create_dynamic_field_mappings(system1_name: str, system1_columns: List[str], 
                                system2_name: str, system2_columns: List[str], llm) -> Dict:
    """Create dynamic field mappings between two systems using LLM analysis"""
    
    prompt = f"""
    You are an expert pharmaceutical data analyst. Analyze columns from two systems:
    
    {system1_name} columns: {system1_columns}
    {system2_name} columns: {system2_columns}
    
    Find semantically similar columns that should be compared for data consistency.
    Return ONLY a JSON object with field mappings:
    {{
        "source_system": "{system1_name}",
        "target_system": "{system2_name}",
        "source_table": "dynamic_table",
        "target_table": "dynamic_table",
        "field_map": {{
            "source_column": "target_column"
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
        
        mapping_data = json.loads(response_content)
        
        # Validate that mapped columns exist in both systems
        field_map = {}
        for source_col, target_col in mapping_data.get('field_map', {}).items():
            if source_col in system1_columns and target_col in system2_columns:
                field_map[source_col] = target_col
        
        mapping_data['field_map'] = field_map
        return mapping_data
        
    except Exception as e:
        print(f"⚠️ LLM field mapping generation failed: {e}, using fallback")
        # Fallback: find exact name matches
        field_map = {}
        for col1 in system1_columns:
            if col1 in system2_columns:
                field_map[col1] = col1
        
        return {
            "source_system": system1_name,
            "target_system": system2_name,
            "source_table": "dynamic_table",
            "target_table": "dynamic_table",
            "field_map": field_map
        }

# Inconsistency-specific configurations
INCONSISTENCY_CONFIG = {
    "output_dir": "data_quality_reports/inconsistency",
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