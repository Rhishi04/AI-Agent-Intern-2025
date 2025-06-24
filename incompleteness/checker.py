import pandas as pd
from typing import List, Dict
from datetime import datetime
from .config import get_identifier_columns, ColumnConfig

class IncompletenessChecker:
    def __init__(self, system_name: str, config: Dict):
        self.system_name = system_name
        self.config = config
        
    def _is_null_value(self, value) -> bool:
        """Check if a value is null or null-like"""
        # First check for pandas null values (np.nan, None, pd.NA)
        if pd.isna(value):
            return True
            
        # For string values, check for null-like strings
        if isinstance(value, str):
            # Convert to lowercase and strip whitespace for string comparison
            value = value.lower().strip()
            
            # Check for null-like strings
            null_like_values = {
                # Standard null representations
                "n/a", "na", "null", "none", "nil",
                # Empty or whitespace
                "", " ",
                # Common placeholders
                "-", "--", ".",
                # Additional variations
                "n.a.", "n.a", "n/a.", "na.", "null.", "none.",
                # String representations of NaN
                "nan"
            }
            
            # Check if the value is in our null set
            if value in null_like_values or value.isspace():
                return True
                
        return False
    
def check_completeness(self, df: pd.DataFrame, table_name: str, batch_id: str) -> List[Dict]:
    findings = []
    identifier_columns = get_identifier_columns(self.config)
    
    print(f"   📊 Analyzing {len(df.columns)} columns in {table_name}...")
    
    for column in df.columns:
        # Check ALL columns for missing values, not just required ones
        column_config = self.config.column_configs.get(column)
        if not column_config:
            # If no config exists for this column, create a default one
            column_config = ColumnConfig(severity="Medium", is_required=True)
        
        # Work on a copy, do not modify df in place!
        col_values = df[column].copy()
        null_mask = col_values.apply(self._is_null_value)
        missing_count = null_mask.sum()
        total_count = len(df)
        missing_percentage = (missing_count / total_count) * 100 if total_count > 0 else 0
        
        # Determine severity based on missing percentage and column config
        if missing_percentage >= 50:
            severity = "High"
        elif missing_percentage >= 20:
            severity = "Medium"
        else:
            severity = "Low"
        
        # Override with column config severity if it's higher
        if column_config.severity == "High" and severity != "High":
            severity = "High"
        elif column_config.severity == "Medium" and severity == "Low":
            severity = "Medium"
        
        # Print progress for columns with missing data
        if missing_count > 0:
            status_icon = "🔴" if severity == "High" else "🟡" if severity == "Medium" else "🟢"
            print(f"     {status_icon} {column}: {missing_count}/{total_count} missing ({missing_percentage:.1f}%) - {severity}")
        
            # Get ALL rows with missing values
            missing_rows = df[null_mask]

            for idx, row in missing_rows.iterrows():
                row_identifiers = {id_col: str(row[id_col]) for id_col in identifier_columns if id_col in df.columns}
                findings.append({
                    "type": "Data Incompleteness",
                    "system": self.system_name,
                    "description": f"Missing or null-like value in {column}",
                    "severity": severity,
                    "column_name": column,
                    "missing_count": missing_count,
                    "total_count": total_count,
                    "missing_percentage": missing_percentage,
                    "row_identifiers": row_identifiers,
                    "timestamp": datetime.now().isoformat()
                })
    
    return findings
