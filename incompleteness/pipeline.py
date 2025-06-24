import pandas as pd
from typing import Dict, List
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
from .checker import IncompletenessChecker
from .config import DYNAMIC_SYSTEMS_CONFIG, PipelineConfig, get_identifier_columns
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IncompletenessPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.findings = []
        self.output_dir = "data_quality_reports/incompleteness"
        self.systems_data = {}  # Initialize systems_data as instance variable
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _generate_finding_id(self, batch_id: str, index: int) -> str:
        """Generate a finding ID using batch ID and index"""
        return f"INC-{batch_id}-{index:04d}"
        
    def _load_system_data(self, system_name: str) -> Dict[str, pd.DataFrame]:
        """Load data from a specific system's tables"""
        system_config = self.config.systems[system_name]
        data = {}
        
        for table in system_config.tables:
            try:
                # Check if we have a DataFrame already loaded (from monkey-patched function)
                if hasattr(self, '_patched_data') and system_name in self._patched_data:
                    data[table] = self._patched_data[system_name].get(table, pd.DataFrame())
                else:
                    # Fallback to database loading (for real implementations)
                    df = pd.read_sql(f"SELECT * FROM {table}", system_config.connection_string)
                    data[table] = df
            except Exception as e:
                logger.error(f"Error loading data from {system_name}.{table}: {str(e)}")
                # Return empty DataFrame instead of None
                data[table] = pd.DataFrame()
                
        return data
    
    def _save_findings_to_csv(self, findings_df: pd.DataFrame):
        """Save only the detailed findings to CSV file (no summary CSV)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detailed_filename = f"{self.output_dir}/incompleteness_detailed_{timestamp}.csv"
        try:
            findings_df.to_csv(detailed_filename, index=False)
            logger.info(f"Detailed findings saved to {detailed_filename}")
            print(f"\n📄 **CSV Report Generated:**")
            print(f"   📋 Detailed Report: {detailed_filename}")
        except Exception as e:
            logger.error(f"Error saving detailed findings: {str(e)}")
    
    def _cross_system_identifier_check(self, batch_id: str) -> List[Dict]:
        findings = []
        system_names = list(self.config.systems.keys())
        # For each pair of systems
        for i in range(len(system_names)):
            for j in range(i + 1, len(system_names)):
                sys1 = system_names[i]
                sys2 = system_names[j]
                config1 = self.config.systems[sys1]
                config2 = self.config.systems[sys2]
                df1 = next(iter(self.systems_data[sys1].values())) if self.systems_data[sys1] else None
                df2 = next(iter(self.systems_data[sys2].values())) if self.systems_data[sys2] else None
                if df1 is None or df2 is None or df1.empty or df2.empty:
                    continue
                ids1 = set(get_identifier_columns(config1))
                ids2 = set(get_identifier_columns(config2))
                common_ids = ids1 & ids2
                
                # Check if common identifiers are present in both systems
                for id_col in common_ids:
                    if id_col in df1.columns and id_col in df2.columns:
                        missing_in_sys1 = df1[id_col].isnull().sum()
                        missing_in_sys2 = df2[id_col].isnull().sum()
                        
                        if missing_in_sys1 > 0 or missing_in_sys2 > 0:
                            findings.append({
                                'finding_id': self._generate_finding_id(batch_id, len(findings) + 1),
                                'type': 'Cross-System Identifier Incompleteness',
                                'system': f'{sys1} ↔ {sys2}',
                                'description': f"Common identifier '{id_col}' has missing values: {sys1}={missing_in_sys1}, {sys2}={missing_in_sys2}",
                                'table_name': f'{sys1}_table ↔ {sys2}_table',
                                'column_name': id_col,
                            'severity': 'High',
                            'timestamp': datetime.now().isoformat()
                        })
        
        return findings

    def run_pipeline(self) -> List[Dict]:
        """Execute the incompleteness pipeline"""
        start_time = datetime.now()
        batch_id = start_time.strftime("%Y%m%d%H%M%S")  # Use timestamp as batch ID
        
        print(f"\n📊 **Analyzing data for incompleteness issues...**")
        
        # Load data from all systems
        self.systems_data = {}  # Reset systems_data
        for system_name in self.config.systems:
            self.systems_data[system_name] = self._load_system_data(system_name)
        
        # Run completeness checks in parallel
        all_findings = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            quality_futures = []
            for system_name, tables_data in self.systems_data.items():
                checker = IncompletenessChecker(system_name, self.config.systems[system_name])
                for table_name, df in tables_data.items():
                    if not df.empty:  # Only check non-empty DataFrames
                        print(f"   🔍 Checking {system_name}.{table_name} ({len(df)} records, {len(df.columns)} columns)")
                        # Run completeness check
                        future = executor.submit(checker.check_completeness, df, table_name, batch_id)
                        quality_futures.append(future)
            # Collect all findings
            for future in quality_futures:
                all_findings.extend(future.result())
        
        # Add cross-system identifier presence findings
        all_findings.extend(self._cross_system_identifier_check(batch_id))
        
        # Assign finding IDs in a single thread-safe loop
        for idx, finding in enumerate(all_findings, start=1):
            finding['finding_id'] = self._generate_finding_id(batch_id, idx)
        
        self.findings = all_findings
        
        # Generate summary statistics
        if all_findings:
            # Count by severity
            severity_counts = {}
            for finding in all_findings:
                severity = finding.get('severity', 'Low')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            print(f"\n📊 **Incompleteness Summary**")
            print("=" * 60)
            print(f"   📈 Total Findings: {len(all_findings)}")
            print(f"   🔴 High Severity Issues: {severity_counts.get('High', 0)}")
            print(f"   🟡 Medium Severity Issues: {severity_counts.get('Medium', 0)}")
            print(f"   🟢 Low Severity Issues: {severity_counts.get('Low', 0)}")
            
            # Calculate quality score
            total_columns = len(set(finding.get('column_name', '') for finding in all_findings))
            high_severity = severity_counts.get('High', 0)
            medium_severity = severity_counts.get('Medium', 0)
            low_severity = severity_counts.get('Low', 0)
            
            quality_score = 100 - (high_severity * 10) - (medium_severity * 5) - (low_severity * 2)
            quality_score = max(0, min(100, quality_score))
            print(f"   📊 Overall Quality Score: {quality_score}/100")
        
        # Save findings to CSV
        findings_df = pd.DataFrame(self.findings)
        if not findings_df.empty:
            self._save_findings_to_csv(findings_df)
        
        return self.findings
    
    def get_findings_summary(self) -> pd.DataFrame:
        """Return a summary of findings in a DataFrame format"""
        if not self.findings:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.findings)
        # Sort by system and timestamp
        return df.sort_values(['system', 'timestamp'], ascending=[True, False]) 