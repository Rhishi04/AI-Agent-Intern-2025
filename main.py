import pandas as pd
import os  # Ensure this is present and not overwritten
import json
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# LangChain imports
from langchain.agents import initialize_agent, AgentType, create_openai_functions_agent
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

# Local imports
from tools import (
    llm_cycle_time_analysis,
    llm_root_cause_analysis,
    llm_kpi_analysis,
    llm_forecast_analysis,
    llm_optimization_analysis,
    llm_visualize_data,
    llm_generate_analysis,
    llm_generate_visualization_code,
    llm_track_pending_approvals,
    select_multiple_datasets,
    set_tool_data,
    get_tool_data
)

try:
    from langchain.agents import create_openai_functions_agent
    from langchain.agents.openai_functions_agent.base import OPENAI_FUNCTIONS_AGENT_PROMPT
    USE_CREATE_AGENT = True

except ImportError:
    from langchain.agents import AgentType
    USE_CREATE_AGENT = False

except Exception as e:
    print(f"An error occurred: {e}")

# Use Azure OpenAI
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import os
import ast
from email_utils import (
    send_email, send_email_with_attachments, save_avg_cycle_time_chart, save_df_to_csv,
    save_forecast_chart, save_delays_by_department_chart, save_kpi_summary_csv, save_forecast_csv
)
import difflib
from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION
)
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import sys
from langchain.prompts import PromptTemplate
from incompleteness.pipeline import IncompletenessPipeline
from incompleteness.config import PipelineConfig as IncompletenessPipelineConfig, create_dynamic_system_config
from inconsistency.pipeline import InconsistencyPipeline
from inconsistency.config import PipelineConfig as InconsistencyPipelineConfig, create_dynamic_system_config as create_inconsistency_system_config, create_dynamic_field_mappings, DYNAMIC_FIELD_MAPPINGS

# Global variables for tool access
current_df = None
column_mapping_manager = None

def quit_check(user_input):
    """Check if user wants to quit and handle exit gracefully"""
    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("👋 Thank you for using the Pharma Change Control Agentic AI System!")
        try:
            if 'column_mapping_manager' in globals() and column_mapping_manager:
                column_mapping_manager.save_mappings()
        except Exception:
            pass
        print("💾 Your column mappings have been saved for future sessions.")
        sys.exit(0)
    return user_input

# 1. Initialize LLM with your Azure OpenAI credentials
llm = AzureChatOpenAI(
    openai_api_version=AZURE_OPENAI_API_VERSION,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    temperature=0.1,  # Slightly higher for more creative responses
)

# 2. Ingest, configure, and prepare data using the LLM
# current_df, config = ingest_data(llm)

# 2.5. Dynamic Dataset Selection and Auto-Mapping
def select_dataset():
    """🔍 Dynamic Dataset Selection with Auto-Mapping"""
    print("\n🔍 **Dynamic Dataset Selection**")
    print("=" * 50)
    
    # Look for CSV files in current directory
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ No CSV files found in current directory.")
        print("💡 Please place your dataset CSV file in this directory and restart.")
        return None, None
    
    print("📁 Available datasets:")
    print("-" * 50)
    
    # Analyze each CSV file to provide detailed information
    file_info = []
    for i, file in enumerate(csv_files, 1):
        try:
            # Get file size
            file_size = os.path.getsize(file)
            size_mb = file_size / (1024 * 1024)
            
            # Load file to get record count and columns
            df = pd.read_csv(file)
            record_count = len(df)
            column_count = len(df.columns)
            
            # Get sample column names (first 5)
            sample_columns = list(df.columns)[:5]
            if len(df.columns) > 5:
                sample_columns.append(f"... and {len(df.columns) - 5} more")
            
            # Use LLM to detect dataset type instead of static keywords
            dataset_type = llm_detect_dataset_type(df, llm)
            
            # Calculate data quality indicators
            missing_data_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            duplicate_rows = len(df[df.duplicated()])
            
            # Quality rating
            if missing_data_pct < 5 and duplicate_rows == 0:
                quality_rating = "🟢 Excellent"
            elif missing_data_pct < 15 and duplicate_rows < 5:
                quality_rating = "🟡 Good"
            elif missing_data_pct < 30:
                quality_rating = "🟠 Fair"
            else:
                quality_rating = "🔴 Poor"
            
            file_info.append({
                'file': file,
                'size_mb': size_mb,
                'record_count': record_count,
                'column_count': column_count,
                'sample_columns': sample_columns,
                'dataset_type': dataset_type,
                'missing_data_pct': missing_data_pct,
                'duplicate_rows': duplicate_rows,
                'quality_rating': quality_rating
            })
            
            print(f"   {i}. 📊 {file}")
            print(f"      📈 Records: {record_count:,} | Columns: {column_count} | Quality: {quality_rating}")
            
        except Exception as e:
            print(f"   {i}. ⚠️ {file} (Error reading: {str(e)[:50]}...)")
            print()
    
    print(f"💡 Found {len(csv_files)} dataset(s)")
    
    # Provide recommendations
    if len(file_info) > 1:
        print(f"\n💡 **Recommendations:**")
        
        # Find best quality dataset
        best_quality = min(file_info, key=lambda x: x['missing_data_pct'])
        print(f"   🏆 Best Quality: {best_quality['file']}")
        
        # Find largest dataset
        largest_dataset = max(file_info, key=lambda x: x['record_count'])
        print(f"   📊 Largest Dataset: {largest_dataset['file']}")
        
        # Find most relevant dataset type (prioritize change control)
        change_control_files = [f for f in file_info if 'change control' in f['dataset_type'].lower()]
        if change_control_files:
            print(f"   🎯 Recommended for Change Control: {change_control_files[0]['file']}")
    
    # Let user select dataset
    while True:
        try:
            choice = quit_check(input(f"\n🔗 Select dataset (1-{len(csv_files)}) or 'auto' for automatic selection: ").strip())
            
            if choice.lower() == 'auto':
                # Auto-select the first CSV file
                selected_file = csv_files[0]
                print(f"🤖 Auto-selected: {selected_file}")
                break
            else:
                choice_num = int(choice)
                if 1 <= choice_num <= len(csv_files):
                    selected_file = csv_files[choice_num - 1]
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(csv_files)}")
        except ValueError:
            print("❌ Please enter a valid number or 'auto'")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return None, None
    
    # Load the selected dataset
    print(f"\n📊 Loading dataset: {selected_file}")
    try:
        df = pd.read_csv(selected_file)
        print(f"✅ Successfully loaded {len(df)} records with {len(df.columns)} columns")
        return df, selected_file
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None, None

def llm_detect_dataset_type(df, llm):
    """Enhanced LLM-based dataset type detection for pharmaceutical systems"""
    prompt = f"""
    You are an expert pharmaceutical data analyst with 15+ years of experience in pharma manufacturing, QMS, LIMS, MES, and other pharmaceutical systems.
    
    Analyze these column names from a dataset:
    {list(df.columns)}
    
    Based on the column names, determine what type of pharmaceutical system or dataset this is. Consider:
    1. Change Control/QMS: columns like Change_Control_ID, Change_Type, Approval_Status, etc.
    2. LIMS (Laboratory Information Management): columns like Lab_Sample_ID, Test_Name, Result_Value, etc.
    3. MES (Manufacturing Execution): columns like Work_Order_ID, Batch_ID, Parameter_Name, etc.
    4. Clinical Trials: columns like Patient_ID, Protocol_ID, Visit_Date, etc.
    5. Inventory/Warehouse: columns like Product_ID, Stock_Level, Location, etc.
    6. Quality Assurance: columns like Deviation_ID, CAPA_ID, Audit_Date, etc.
    7. Regulatory/Compliance: columns like Submission_ID, Approval_Date, Compliance_Status, etc.
    
    Respond with ONLY the most likely system type and a brief justification.
    Format: "SystemType: justification"
    """
    response = llm.invoke(prompt)
    # Expecting: "Change Control: because columns include ..."
    if ":" in response.content:
        dataset_type, _ = response.content.split(":", 1)
        return dataset_type.strip()
    return response.content.strip()

def llm_auto_map_columns(df, required_logical_columns, llm, dataset_type):
    prompt = f"""
    You are an expert pharma data analyst. Here are the columns: {list(df.columns)}
    For a {dataset_type} dataset, map each of these required logical columns to the best-matching actual column in the dataset. If a required column is missing, say so. Respond as JSON: {{logical_column: actual_column or null}}
    Required logical columns: {required_logical_columns}
    """
    response = llm.invoke(prompt)
    try:
        mapping = json.loads(response.content)
    except Exception:
        mapping = {}
    return mapping

# 3. Enhanced Column Mapping System with Session Persistence
class ColumnMappingManager:
    """🔧 Continuous Column Mapping System with session persistence"""
    
    def __init__(self, df, logical_columns):
        self.df = df
        self.logical_columns = logical_columns
        self.mapping = {}
        self.mapping_file = "column_mappings.json"
        self.load_mappings()
        
    def load_mappings(self):
        """Load saved mappings from file"""
        try:
            if os.path.exists(self.mapping_file):
                with open(self.mapping_file, 'r') as f:
                    self.mapping = json.load(f)
                print(f"📂 Loaded {len(self.mapping)} saved column mappings")
        except Exception as e:
            print(f"⚠️ Could not load mappings: {e}")
    
    def save_mappings(self):
        """Save mappings to file"""
        try:
            with open(self.mapping_file, 'w') as f:
                json.dump(self.mapping, f, indent=2)
            print(f"💾 Saved {len(self.mapping)} column mappings")
        except Exception as e:
            print(f"⚠️ Could not save mappings: {e}")
    
    def extract_missing_columns(self, text: str) -> List[str]:
        """🎯 LLM-Enhanced Column Extraction Logic"""
        try:
            # Use LLM to extract missing column names from text
            prompt = f"""
            You are an expert data analyst. Extract the names of missing columns from this text.
            Text: "{text}"
            
            Return ONLY a JSON array of column names that are missing, like: ["column1", "column2"]
            If no missing columns are mentioned, return an empty array: []
            """
            
            response = llm.invoke(prompt)
            response_content = response.content.strip()
            
            # Try to parse JSON response
            try:
                if response_content.startswith('```json'):
                    response_content = response_content.replace('```json', '').replace('```', '').strip()
                elif response_content.startswith('```'):
                    response_content = response_content.replace('```', '').strip()
                
                missing_columns = json.loads(response_content)
                if isinstance(missing_columns, list):
                    # Validate that extracted columns are in our logical columns
                    validated_columns = [col for col in missing_columns if col in self.logical_columns]
                    return validated_columns
            except json.JSONDecodeError:
                pass
            
            # Fallback to regex-based extraction if LLM fails
            missing_columns = []
            
            # Extract quoted column names
            quoted_pattern = r'"([^"]+)"'
            quoted_matches = re.findall(quoted_pattern, text)
            missing_columns.extend(quoted_matches)
            
            # Extract columns after "correspond to"
            correspond_pattern = r'correspond to\s+["\']?([^"\']+)["\']?'
            correspond_matches = re.findall(correspond_pattern, text, re.IGNORECASE)
            missing_columns.extend(correspond_matches)
            
            # Validate extracted columns
            validated_columns = [col for col in missing_columns if col in self.logical_columns]
            
            return list(set(validated_columns))  # Remove duplicates
            
        except Exception as e:
            print(f"⚠️ LLM column extraction failed: {e}, using fallback method")
            # Fallback to original regex method
            missing_columns = []
            
            # Extract quoted column names
            quoted_pattern = r'"([^"]+)"'
            quoted_matches = re.findall(quoted_pattern, text)
            missing_columns.extend(quoted_matches)
            
            # Extract columns after "correspond to"
            correspond_pattern = r'correspond to\s+["\']?([^"\']+)["\']?'
            correspond_matches = re.findall(correspond_pattern, text, re.IGNORECASE)
            missing_columns.extend(correspond_matches)
            
            # Validate extracted columns
            validated_columns = [col for col in missing_columns if col in self.logical_columns]
            
            return list(set(validated_columns))  # Remove duplicates
    
    def map_all_missing_columns(self, missing_columns: List[str]) -> Dict[str, str]:
        """Maps ALL missing columns at once with intelligent suggestions"""
        new_mappings = {}
        
        print(f"\n🔧 Mapping {len(missing_columns)} missing columns...")
        
        for missing_col in missing_columns:
            if missing_col in self.mapping:
                print(f"   ✅ '{missing_col}' already mapped to '{self.mapping[missing_col]}'")
                continue
                
            # Get intelligent suggestions
            suggestions = self.get_column_suggestions(missing_col)
            
            print(f"\n   📋 Column: '{missing_col}'")
            print(f"   💡 Suggestions: {', '.join(suggestions[:3])}")
            
            # Auto-select if there's a perfect match
            if suggestions and suggestions[0].lower() == missing_col.lower():
                selected = suggestions[0]
                print(f"   ✅ Auto-selected: '{selected}' (perfect match)")
            else:
                # Show available columns
                available_cols = list(self.df.columns)
                print(f"   📊 Available columns: {', '.join(available_cols[:5])}{'...' if len(available_cols) > 5 else ''}")
                
                # Auto-select first suggestion if available
                if suggestions:
                    selected = suggestions[0]
                    print(f"   🤖 Auto-selected: '{selected}' (best match)")
                else:
                    # If no suggestions, skip this column
                    print(f"   ⚠️ No suitable mapping found for '{missing_col}'")
                    continue
            
            new_mappings[missing_col] = selected
            print(f"   ✅ Mapped '{missing_col}' → '{selected}'")
        
        # Update the mapping
        self.mapping.update(new_mappings)
        self.save_mappings()
        
        return new_mappings
    
    def get_column_suggestions(self, target_col: str) -> List[str]:
        """🎯 LLM-Enhanced Intelligent Column Suggestion System"""
        try:
            # Use LLM to find semantically similar columns
            prompt = f"""
            You are an expert data analyst. Given a target column name "{target_col}" and these available columns: {list(self.df.columns)}
            
            Find the most semantically similar columns to "{target_col}". Consider:
            1. Exact matches (case-insensitive)
            2. Partial matches (contains the target word)
            3. Semantic similarity (similar meaning, synonyms)
            4. Common abbreviations or variations
            
            Return ONLY a JSON array of the top 5 most relevant column names, ordered by relevance: ["column1", "column2", "column3"]
            """
            
            response = llm.invoke(prompt)
            response_content = response.content.strip()
            
            # Try to parse JSON response
            try:
                if response_content.startswith('```json'):
                    response_content = response_content.replace('```json', '').replace('```', '').strip()
                elif response_content.startswith('```'):
                    response_content = response_content.replace('```', '').strip()
                
                suggestions = json.loads(response_content)
                if isinstance(suggestions, list):
                    # Validate that suggested columns exist in the dataset
                    valid_suggestions = [col for col in suggestions if col in self.df.columns]
                    return valid_suggestions[:5]  # Return top 5
            except json.JSONDecodeError:
                pass
            
            # Fallback to original fuzzy matching if LLM fails
            return self._fallback_column_suggestions(target_col)
            
        except Exception as e:
            print(f"⚠️ LLM column suggestions failed: {e}, using fallback method")
            return self._fallback_column_suggestions(target_col)
    
    def _fallback_column_suggestions(self, target_col: str) -> List[str]:
        """Fallback column suggestion method using fuzzy matching"""
        target_lower = target_col.lower()
        suggestions = []
        
        # Direct matches
        for col in self.df.columns:
            if col.lower() == target_lower:
                suggestions.append(col)
        
        # Partial matches
        for col in self.df.columns:
            col_lower = col.lower()
            if target_lower in col_lower or col_lower in target_lower:
                if col not in suggestions:
                    suggestions.append(col)
        
        # Fuzzy matches using difflib
        if not suggestions:
            col_names = list(self.df.columns)
            matches = difflib.get_close_matches(target_col, col_names, n=3, cutoff=0.6)
            suggestions.extend(matches)
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def reset_mappings(self):
        """Reset all column mappings"""
        self.mapping = {}
        self.save_mappings()
        print("🔄 All column mappings have been reset")
    
    def show_mappings(self):
        """Display current column mappings"""
        if not self.mapping:
            print("📋 No column mappings defined")
            return
        
        print("\n📋 **Current Column Mappings:**")
        for logical, actual in self.mapping.items():
            print(f"   • {logical} → {actual}")
    
    def remap_column(self, logical_name: str):
        """Remap a specific column"""
        if logical_name not in self.logical_columns:
            print(f"❌ '{logical_name}' is not a valid logical column")
            return
        
        suggestions = self.get_column_suggestions(logical_name)
        print(f"\n🔧 Remapping '{logical_name}'")
        print(f"💡 Suggestions: {', '.join(suggestions[:3])}")
        
        available_cols = list(self.df.columns)
        print(f"📊 Available columns: {', '.join(available_cols[:5])}{'...' if len(available_cols) > 5 else ''}")
        
        if suggestions:
            selected = suggestions[0]
            self.mapping[logical_name] = selected
            self.save_mappings()
            print(f"✅ Remapped '{logical_name}' → '{selected}'")

    def validate_column_mapping(self) -> list:
        """Validate that all required logical columns are mapped to real columns in the DataFrame."""
        missing = []
        for logical_col in self.logical_columns:
            actual_col = self.mapping.get(logical_col)
            if not actual_col or actual_col not in self.df.columns:
                missing.append(logical_col)
        return missing

    def get_mapped_column(self, logical_name):
        return self.mapping.get(logical_name)

def handle_special_commands(user_input: str) -> str:
    """🔧 Handle special commands and return status"""
    
    if user_input == 'mappings':
        column_mapping_manager.show_mappings()
        return "HANDLED"
    
    elif user_input == 'reset mappings':
        confirm = quit_check(input("⚠️ Are you sure you want to reset all column mappings? (y/n): ").strip().lower())
        if confirm in ['y', 'yes']:
            column_mapping_manager.reset_mappings()
        return "HANDLED"
    
    elif user_input.startswith('remap '):
        column_name = user_input[6:].strip()
        if column_name:
            column_mapping_manager.remap_column(column_name)
        else:
            print("❌ Please specify a column name: remap <column_name>")
        return "HANDLED"
    
    elif user_input in ['help', 'capabilities']:
        show_capabilities_overview()
        return "HANDLED"
    
    elif user_input.lower() in ['incompleteness', 'check incompleteness', 'data incompleteness', 'run incompleteness check']:
        if current_df is None or current_df.empty:
            print("❌ No dataset loaded. Please load a dataset first using 'load <filename>'")
            return "HANDLED"
        
        print("🔍 Running incompleteness check using IncompletenessPipeline...")

        # Debug: Print DataFrame info
        print(f"DEBUG: current_df shape: {current_df.shape}")
        print(f"DEBUG: current_df columns: {list(current_df.columns)}")
        
        # Auto-detect system type using LLM
        column_names = list(current_df.columns)
        prompt = f"""
        You are an expert pharmaceutical data analyst. Based on these column names:
        {column_names}
        
        What type of pharmaceutical system is this? Choose from: LIMS, MES, QMS, Change Control, or Other.
        Respond with ONLY the system type.
        """
        
        try:
            response = llm.invoke(prompt)
            system_name = response.content.strip()
            print(f"🤖 Auto-detected system type: {system_name}")
        except Exception as e:
            print(f"⚠️ Auto-detection failed: {e}, using 'Generic'")
            system_name = "Generic"
        
        # Create dynamic system configuration
        try:
            system_config = create_dynamic_system_config(system_name, column_names, llm)
            print(f"DEBUG: system_config keys: {list(system_config.keys())}")
        except Exception as e:
            print(f"❌ Error creating system config: {e}")
            return "HANDLED"
        
        # Configure the pipeline
        try:
            config = IncompletenessPipelineConfig(
                systems={system_name: system_config},
                max_workers=1
            )
            pipeline = IncompletenessPipeline(config)
        except Exception as e:
            print(f"❌ Error initializing pipeline: {e}")
            return "HANDLED"
        
        # Monkey-patch the data loading function to use our DataFrame
        def patched_load_system_data(s_name: str):
            if s_name == system_name:
                return {"dynamic_table": current_df}
            else:
                return {"dynamic_table": pd.DataFrame()}
        
        pipeline._load_system_data = patched_load_system_data
        pipeline._patched_data = {system_name: {"dynamic_table": current_df}}
        
        # Run the pipeline
        try:
            findings = pipeline.run_pipeline()
            print(f"DEBUG: findings: {findings}")
        except Exception as e:
            print(f"❌ Error running pipeline: {e}")
            return "HANDLED"
        
        if findings:
            print(f"✅ Incompleteness check completed. Found {len(findings)} issues.")
            print("📄 CSV files generated in 'data_quality_reports/incompleteness/' directory")
        else:
            print("✅ No incompleteness issues found. Data quality is excellent!")
        
        return "HANDLED"
    
    elif user_input.lower() in ['inconsistency', 'check inconsistency', 'data inconsistency', 'interactive inconsistency', 'multi dataset inconsistency']:
        print("🔍 Running multi-dataset inconsistency check (requires 2 or more datasets)...")
        print("💡 This will prompt you to select multiple datasets for comparison.")
        
        # Step 1: Select multiple datasets
        datasets = select_multiple_datasets()
        if not datasets:
            print("❌ No datasets selected for inconsistency analysis")
            return "HANDLED"
        
        print(f"✅ Selected {len(datasets)} datasets for inconsistency analysis")
        
        # Step 2: Create system configurations for each dataset using LLM
        systems_config = {}
        systems_data = {}
        
        for dataset_name, df in datasets.items():
            # Auto-detect system type using LLM
            column_names = list(df.columns)
            prompt = f"""
            You are an expert pharmaceutical data analyst. Based on these column names:
            {column_names}
            
            What type of pharmaceutical system is this? Choose from: LIMS, MES, QMS, Change Control, or Other.
            Respond with ONLY the system type.
            """
            
            try:
                response = llm.invoke(prompt)
                system_type = response.content.strip()
                print(f"🤖 Auto-detected system type for {dataset_name}: {system_type}")
            except Exception as e:
                print(f"⚠️ Auto-detection failed for {dataset_name}: {e}, using 'Generic'")
                system_type = "Generic"
            
            # Create system configuration using LLM
            system_config = create_inconsistency_system_config(system_type, column_names, llm)
            systems_config[dataset_name] = system_config
            systems_data[dataset_name] = {"dynamic_table": df}
        
        # Step 3: Generate field mappings between datasets using LLM
        print(f"\n🔗 **Step 3: LLM Field Mapping Generation**")
        print("-" * 40)
        
        # Clear existing mappings
        DYNAMIC_FIELD_MAPPINGS.clear()
        
        # Generate mappings between all pairs of datasets
        dataset_names = list(datasets.keys())
        for i in range(len(dataset_names)):
            for j in range(i + 1, len(dataset_names)):
                dataset1_name = dataset_names[i]
                dataset2_name = dataset_names[j]
                
                print(f"🔗 Generating mappings between {dataset1_name} ↔ {dataset2_name}")
                
                # Get column names for both datasets
                system1_columns = list(datasets[dataset1_name].columns)
                system2_columns = list(datasets[dataset2_name].columns)
                
                # Use LLM to generate field mappings
                field_mapping = create_dynamic_field_mappings(
                    dataset1_name, system1_columns,
                    dataset2_name, system2_columns,
                    llm
                )
                
                if field_mapping and field_mapping.get('field_map'):
                    DYNAMIC_FIELD_MAPPINGS.append(field_mapping)
                    mapped_columns = list(field_mapping['field_map'].keys())
                    print(f"   ✅ Mapped {len(mapped_columns)} columns: {', '.join(mapped_columns[:3])}{'...' if len(mapped_columns) > 3 else ''}")
                else:
                    print(f"   ⚠️ No mappings generated between {dataset1_name} and {dataset2_name}")
        
        print(f"\n📊 **Mapping Summary:**")
        print(f"   • Total dataset pairs: {len(DYNAMIC_FIELD_MAPPINGS)}")
        print(f"   • Total field mappings: {sum(len(mapping.get('field_map', {})) for mapping in DYNAMIC_FIELD_MAPPINGS)}")
        
        # Step 4: Configure and run the pipeline
        print(f"\n🔍 **Step 4: Running Cross-Dataset Inconsistency Checks**")
        print("-" * 40)
        
        config = InconsistencyPipelineConfig(
            systems=systems_config,
            max_workers=1
        )
        pipeline = InconsistencyPipeline(config)
        
        # Monkey-patch the data loading function to use our DataFrames
        def patched_load_system_data(s_name: str):
            if s_name in systems_data:
                return systems_data[s_name]
            else:
                return {"dynamic_table": pd.DataFrame()}
        
        pipeline._load_system_data = patched_load_system_data
        
        # Run the pipeline
        findings = pipeline.run_pipeline()
        
        if findings:
            print(f"✅ Inconsistency check completed. Found {len(findings)} issues.")
            print("📄 CSV files generated in 'data_quality_reports/inconsistency/' directory")
        else:
            print("✅ No inconsistencies found across datasets!")
        
        return "HANDLED"
    
    elif user_input in ['quit', 'exit', 'bye']:
        print("👋 Thank you for using the Pharma Change Control Agentic AI System!")
        try:
            if 'column_mapping_manager' in globals() and column_mapping_manager:
                column_mapping_manager.save_mappings()
        except Exception:
            pass
        print("💾 Your column mappings have been saved for future sessions.")
        return "QUIT"
    
    return "CONTINUE"

# Custom system prompt for better tool selection
CUSTOM_AGENT_PROMPT = """You are an expert pharmaceutical data analyst assistant. Your job is to help users analyze their change control data and generate insights.

IMPORTANT: The dataset has already been loaded and is available for analysis. You do NOT need to ask the user to upload any files. The dataset contains change control data with columns like Change_Control_ID, TypeOfChange, Creation_Date, Implementation_Date, Status, Authority_Approval_Status, Dept, etc.

CRITICAL CONTEXT AWARENESS RULES:
1. ALWAYS read the "Context:" section at the beginning of user input
2. If the user says "yes", "provide", "show me", "details", "results", or similar follow-up requests, refer to the previous analysis in the context
3. When the user asks for "details" or "results" after an analysis, provide the specific output from the last tool mentioned in context
4. If the context shows a recent analysis was performed, use that information to understand what the user is asking for
5. Use the "Last analysis performed" and "Last result" information from context to provide relevant follow-up

IMPORTANT TOOL SELECTION RULES:
1. When user asks for "chart", "visualize", "export", "PNG", "CSV", "file", or "generate files" → Use llm_generate_visualization_code
2. When user asks for "KPI summary", "KPI chart", "KPI CSV", or "export KPI" → Use llm_kpi_analysis (it generates files)
3. When user asks for "cycle time chart", "cycle time CSV", "export cycle time" → Use llm_generate_visualization_code
4. When user asks for analysis without file generation → Use analysis tools (llm_cycle_time_analysis, llm_root_cause_analysis, etc.)
5. When user asks for "inconsistency" or "multi-dataset" → Use InconsistencyPipeline directly

FILE GENERATION TOOLS:
- llm_generate_visualization_code: Generates PNG and CSV for any visualization request
- llm_kpi_analysis: Generates PNG and CSV for KPI analysis
- IncompletenessPipeline: Generates CSV for incompleteness analysis
- InconsistencyPipeline: Generates CSV for inconsistency analysis

ANALYSIS TOOLS (NO FILES):
- llm_cycle_time_analysis: Returns cycle time analysis summary only
- llm_root_cause_analysis: Returns root cause analysis summary only
- llm_forecast_analysis: Returns forecasting summary only
- llm_optimization_analysis: Returns optimization suggestions only

CONTEXT HANDLING EXAMPLES:
- If context shows "Last analysis performed: llm_cycle_time_analysis" and user says "yes provide", use llm_generate_visualization_code to create charts
- If context shows "Last analysis performed: llm_kpi_analysis" and user says "details", provide the full KPI results
- If context shows recent cycle time analysis and user says "export", use llm_generate_visualization_code
- If user asks for "more" or "continue", refer to the previous analysis and suggest next steps

Always prioritize file generation tools when the user asks for charts, exports, or files. The visualization tools will create actual PNG and CSV files in the data_quality_reports directory.

{input}

{agent_scratchpad}"""

def interactive_analysis(df, mapping_manager):
    """🤖 Main interactive analysis loop with enhanced user experience"""
    
    # Set tool data for access by tools
    set_tool_data(df, mapping_manager)
    
    # Show capabilities overview
    show_capabilities_overview()
    
    # Initialize agent with enhanced tools
    tools = [
    Tool(
        name="llm_generate_analysis",
        func=lambda x: llm_generate_analysis(current_df, x, llm),
        description="📈 **Custom Analysis**: Performs custom data analysis based on user requests using LLM. Use this for 'analyze data', 'custom analysis', or specific analytical requests."
    ),
    Tool(
        name="llm_generate_visualization_code",
        func=lambda x: llm_generate_visualization_code(current_df, x, llm),
        description="💻 **Visualization Code**: Generates Python code for creating data visualizations using LLM. Use this for 'generate chart code', 'create plot script', or 'visualization code'."
    ),
    Tool(
        name="llm_cycle_time_analysis",
        func=lambda x: llm_cycle_time_analysis(current_df, llm),
        description="🔍 **LLM-Driven Cycle Time Analysis**: ADAPTIVE cycle time analysis that automatically identifies date columns, calculates cycle times, and analyzes performance by change type. Works with ANY dataset structure - no hardcoded column names. Use this for 'cycle time analysis', 'performance analysis', or 'time tracking'."
    ),
    Tool(
        name="llm_root_cause_analysis",
        func=lambda x: llm_root_cause_analysis(current_df, llm),
        description="🔍 **LLM-Driven Root Cause Analysis**: ADAPTIVE root cause analysis that automatically identifies department, status, and priority columns to find bottlenecks and delays. Works with ANY dataset structure - no hardcoded column names. Use this for 'root cause analysis', 'bottleneck identification', or 'delay analysis'."
    ),
    Tool(
        name="llm_kpi_analysis",
        func=lambda x: llm_kpi_analysis(current_df, llm),
        description="📊 **LLM-Driven KPI Analysis**: ADAPTIVE KPI analysis that automatically identifies relevant columns and GENERATES PNG AND CSV FILES with comprehensive performance metrics. Works with ANY dataset structure - no hardcoded column names. Use this for 'KPI analysis', 'performance metrics', or 'export KPI data'."
    ),
    Tool(
        name="llm_forecast_analysis",
        func=lambda x: llm_forecast_analysis(current_df, llm),
        description="🔮 **LLM-Driven Forecast Analysis**: ADAPTIVE forecasting that automatically identifies time-series columns and predicts future performance trends. Works with ANY dataset structure - no hardcoded column names. Use this for 'forecast analysis', 'trend prediction', or 'future planning'."
    ),
    Tool(
        name="llm_optimization_analysis",
        func=lambda x: llm_optimization_analysis(current_df, llm),
        description="🔧 **Optimization Analysis**: Uses LLM to analyze data and suggest process optimizations, efficiency improvements, and best practices. Use this for 'optimization suggestions', 'process improvement', or 'efficiency analysis'."
    ),
    Tool(
        name="llm_visualize_data",
        func=lambda x: llm_visualize_data(current_df, x, llm),
        description="📊 **Data Visualization**: Creates charts, graphs, and visual representations of data using LLM. Use this for 'create chart', 'visualize data', 'plot graph', or 'show trends'."
    ),
    Tool(
        name="llm_track_pending_approvals",
        func=lambda x: llm_track_pending_approvals(current_df, llm),
        description="🔍 **Track Pending Approvals**: Dynamically tracks changes waiting for authority approval and provides breakdown by type/department. Use this for 'pending approvals', 'approval delays', or 'authority approval analysis'."
    ),
    ]
    
    # Initialize memory for conversation context with persistence
    memory_file = "conversation_memory.json"
    
    # Create a custom memory system that stores both conversation and tool outputs
    class EnhancedMemory:
        def __init__(self):
            self.conversation_history = []
            self.tool_outputs = []
            self.last_analysis = None
            self.last_tool_used = None
        
        def add_user_message(self, content):
            self.conversation_history.append({"type": "human", "content": content})
        
        def add_ai_message(self, content):
            self.conversation_history.append({"type": "ai", "content": content})
        
        def add_tool_output(self, tool_name, output):
            self.tool_outputs.append({"tool": tool_name, "output": output})
            self.last_analysis = output
            self.last_tool_used = tool_name
        
        def get_context_summary(self):
            if not self.conversation_history:
                return ""
            
            # Get the last few exchanges for context
            recent_history = self.conversation_history[-6:]  # Last 3 exchanges
            context = "Recent conversation:\n"
            for msg in recent_history:
                if msg["type"] == "human":
                    context += f"User: {msg['content']}\n"
                else:
                    context += f"Assistant: {msg['content']}\n"
            
            if self.last_tool_used and self.last_analysis:
                context += f"\nLast analysis performed: {self.last_tool_used}\n"
                if isinstance(self.last_analysis, dict):
                    context += f"Last result: {str(self.last_analysis)[:200]}...\n"
            
            return context
        
        def save(self, filename):
            data = {
                "conversation_history": self.conversation_history,
                "tool_outputs": self.tool_outputs,
                "last_analysis": self.last_analysis,
                "last_tool_used": self.last_tool_used
            }
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
        
        def load(self, filename):
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    data = json.load(f)
                    self.conversation_history = data.get("conversation_history", [])
                    self.tool_outputs = data.get("tool_outputs", [])
                    self.last_analysis = data.get("last_analysis")
                    self.last_tool_used = data.get("last_tool_used")
                    print(f"📂 Loaded conversation memory ({len(self.conversation_history)} messages)")
        
        def clear_memory(self, filename):
            """Clear conversation memory and delete the file"""
            self.conversation_history = []
            self.tool_outputs = []
            self.last_analysis = None
            self.last_tool_used = None
            if os.path.exists(filename):
                os.remove(filename)
                print("🗑️ Conversation memory cleared for new session")
    
    # Initialize enhanced memory
    enhanced_memory = EnhancedMemory()
    # Don't load previous conversation - start fresh each session
    # enhanced_memory.load(memory_file)
    
        # Also keep the LangChain memory for compatibility
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="input"
    )
    
    # Load existing conversation into LangChain memory (empty for fresh session)
    for msg in enhanced_memory.conversation_history:
        if msg['type'] == 'human':
            memory.chat_memory.add_user_message(msg['content'])
        elif msg['type'] == 'ai':
            memory.chat_memory.add_ai_message(msg['content'])
    
    # Initialize agent with enhanced tools and memory
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        memory=memory,
        agent_kwargs={
            "system_message": CUSTOM_AGENT_PROMPT
        }
    )
    
    print("🚀 **Ready to analyze your data!**")
    print("💬 Ask me anything about your data, or use 'help' for commands.")
    
    # Main interaction loop
    while True:
        try:
            user_input = quit_check(input("\n💬 You: ").strip())
            
            if not user_input:
                continue
            
            # Handle special commands
            result = handle_special_commands(user_input)
            
            if result == "QUIT":
                # Clear conversation memory for next session
                enhanced_memory.clear_memory(memory_file)
                break
            elif result == "HANDLED":
                continue
            elif result == "CONTINUE":
                mapped_df = apply_column_mapping(current_df, column_mapping_manager.mapping)
                
                # Run the analysis and print full tool output
                # Add context to the user input
                context_summary = enhanced_memory.get_context_summary()
                enhanced_input = user_input
                if context_summary:
                    enhanced_input = f"Context: {context_summary}\n\nCurrent request: {user_input}"
                
                agent_result = agent.invoke({"input": enhanced_input})
                # If there are intermediate steps, show the last tool output in complete format
                if isinstance(agent_result, dict) and 'intermediate_steps' in agent_result and agent_result['intermediate_steps']:
                    last_action, last_observation = agent_result['intermediate_steps'][-1]
                    print("\n🔎 **Complete Tool Output:**")
                    print("=" * 60)
                    print(f"**Tool Used:** {last_action.tool}")
                    print(f"**Tool Input:** {last_action.tool_input}")
                    print("\n**Complete Results:**")
                    print(json.dumps(last_observation, indent=2, default=str))
                    print("=" * 60)
                    # Store the tool output in enhanced memory
                    enhanced_memory.add_tool_output(last_action.tool, last_observation)
                else:
                    print(f"\n🤖 Assistant: {agent_result.get('output', agent_result)}")
                
                # Provide context-aware suggestions
                suggestions = get_context_suggestions("generic", str(agent_result))
                if suggestions:
                    print(f"\n💡 **Suggested Next Steps:**")
                    for suggestion in suggestions[:3]:
                        print(f"   • {suggestion}")
            else:
                continue
            # Save conversation memory after each interaction
            try:
                enhanced_memory.add_user_message(user_input)
                enhanced_memory.add_ai_message(agent_result.get('output', str(agent_result)))
                enhanced_memory.save(memory_file)
            except Exception as e:
                print(f"⚠️ Could not save conversation memory: {e}")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Your session has been saved.")
            # Clear conversation memory for next session
            enhanced_memory.clear_memory(memory_file)
            break
        except Exception as e:
            print(f"\n❌ **Error**: {str(e)}")
            print("💡 Try rephrasing your question or use 'help' for guidance.")

def apply_column_mapping(df, mapping):
    """Apply column mapping to DataFrame"""
    if not mapping:
        return df
    
    mapped_df = df.copy()
    # Create reverse mapping for tool compatibility
    reverse_mapping = {v: k for k, v in mapping.items()}
    mapped_df.rename(columns=reverse_mapping, inplace=True)
    return mapped_df

def show_capabilities_overview():
    """Display system capabilities overview"""
    print("\n🚀 **Pharma Change Control Agentic AI System**")
    print("=" * 60)
    print("📊 **Core Capabilities:**")
    print("   • 📈 Cycle Time Analysis & Optimization (LLM-Driven)")
    print("   • 🔍 Root Cause Investigation (LLM-Driven)")
    print("   • 📊 KPI Monitoring & Compliance (with CSV/Charts)")
    print("   • 🔮 Predictive Analytics & Forecasting (LLM-Driven)")
    print("   • 🚀 Process Optimization Recommendations (LLM-Driven)")
    print("   • 🔍 Data Incompleteness Analysis (LLM-Driven)")
    print("   • 🔄 Multi-Dataset Inconsistency Analysis (LLM-Driven)")
    print("   • 📊 LLM-Driven Visualization (Any Data)")
    print("   • 📧 Professional Report Generation")
    print("\n💬 **Special Commands:**")
    print("   • 'help' or 'capabilities' - Show this overview")
    print("   • 'mappings' - View current column mappings")
    print("   • 'reset mappings' - Reset all column mappings")
    print("   • 'remap <column>' - Remap a specific column")
    print("   • 'incompleteness' - Run data incompleteness check")
    print("   • 'inconsistency' - Run multi-dataset inconsistency check")
    print("   • 'quit', 'exit', 'bye' - Exit the system")
    print("\n🎯 **Ask me anything about your change control data!**")
    print("=" * 60)

def get_context_suggestions(context_type: str, response: str) -> List[str]:
    """Generate context-aware suggestions based on response"""
    try:
        # Use LLM to generate contextual suggestions
        prompt = f"""
        You are an expert pharmaceutical data analyst. Based on this analysis response:
        "{response}"
        
        Generate 3-5 relevant follow-up questions or next steps that would be valuable for the user.
        Focus on actionable insights, deeper analysis, or related areas to explore.
        
        Return ONLY a JSON array of suggestions: ["suggestion1", "suggestion2", "suggestion3"]
        """
        
        llm_response = llm.invoke(prompt)
        response_content = llm_response.content.strip()

        # Try to parse JSON response
        try:
            if response_content.startswith('```json'):
                response_content = response_content.replace('```json', '').replace('```', '').strip()
            elif response_content.startswith('```'):
                response_content = response_content.replace('```', '').strip()
            
            suggestions = json.loads(response_content)
            if isinstance(suggestions, list):
                return suggestions[:5]  # Return top 5
        except json.JSONDecodeError:
            pass
        
        # Fallback suggestions
        return [
            "Analyze cycle times by department to identify bottlenecks",
            "Generate a forecast for the next quarter",
            "Investigate root causes of delays",
            "Create a KPI summary report"
        ]
        
    except Exception as e:
        # Return generic suggestions if LLM fails
        return [
            "Analyze cycle times by department to identify bottlenecks",
            "Generate a forecast for the next quarter",
            "Investigate root causes of delays"
        ]

# Main execution
if __name__ == "__main__":
    print("🔍 Starting Pharma Change Control Agentic AI System...")

    try:
        print("🔄 Loading dataset...")
        current_df, dataset_file = select_dataset()

        if current_df is None or current_df.empty:
            raise ValueError("Dataset could not be loaded or is empty.")

    except Exception as e:
        print(f"❌ **Critical Error**: {e}")
        print("💡 **Troubleshooting:**")
        print("   • Ensure CSV files are in the current directory")
        print("   • Check file permissions and format")
        print("   • Verify the CSV file contains valid data")
        exit(1)

    print(f"✅ **Data Loaded Successfully**: {len(current_df)} records, {len(current_df.columns)} columns")
    

# Step 1: Define required logical columns
required_logical_columns = [
    "Change_Control_ID", "Change_Type", "Creation_Date", "Implementation_Date",
    "Status", "Authority_Approval_Status", "Initiating_Department"
]

# Step 2: Use LLM to detect dataset type and auto-map
dataset_type = llm_detect_dataset_type(current_df, llm)
print(f"🤖 **Auto-detected System Type**: {dataset_type}")

auto_mapping = llm_auto_map_columns(current_df, required_logical_columns, llm, dataset_type)

# Step 3: Initialize column mapping manager
column_mapping_manager = ColumnMappingManager(current_df, required_logical_columns)

# Step 4: Apply auto-mapped columns
if auto_mapping:
    for logical_col, actual_col in auto_mapping.items():
        if actual_col and actual_col in current_df.columns:
            column_mapping_manager.mapping[logical_col] = actual_col
    print(f"🤖 Auto-mapped {len([v for v in auto_mapping.values() if v])} columns")

# Step 5: Validate column mapping before analysis
missing_mapped = column_mapping_manager.validate_column_mapping()
if missing_mapped:
    print(f"⚠️ **Missing or Unmapped Required Columns**: {missing_mapped}")
    print("🔧 Attempting to map these columns...")
    column_mapping_manager.map_all_missing_columns(missing_mapped)

    # Re-validate after mapping
    missing_mapped = column_mapping_manager.validate_column_mapping()
    if missing_mapped:
        print(f"❌ **Critical Error**: Still missing required columns after mapping: {missing_mapped}")
        exit(1)

# Step 6: Show current mappings
print(f"\n🔍 **Initial Column Mapping Validation**")
print("=" * 50)
for logical_col in required_logical_columns:
    actual_col = column_mapping_manager.mapping.get(logical_col, "❌ NOT MAPPED")
    print(f"   • {logical_col} → {actual_col}")

# Step 7: Ask if user wants to validate manually
skip_validation = False  # Fix: Declare early to prevent UnboundLocalError

while True:
    validate_choice = input("\n❓ Do you want to make changes to the current column mappings? (y/n): ").strip().lower()
    if validate_choice in ['y', 'yes']:
        validation_passed = True
        break
    elif validate_choice in ['n', 'no']:
        print("✅ Proceeding with current column mappings...")
        skip_validation = True
        break
    else:
        print("❌ Please enter 'y' or 'n'.")

# Step 8: If skipped, proceed directly
if skip_validation:
    print(f"\n✅ **Final Column Mappings:**")
    print("=" * 50)
    for logical_col in required_logical_columns:
        actual_col = column_mapping_manager.mapping.get(logical_col, "❌ NOT MAPPED")
        print(f"   • {logical_col} → {actual_col}")
    print(f"\n🚀 **Starting Analysis...**")
    interactive_analysis(current_df, column_mapping_manager)
    exit(0)

# Step 9: Detailed confirmation
print(f"\n🔍 **Detailed Column Mapping Validation**")
print("=" * 50)
print("💡 **Instructions:**")
print("   • Review each mapping carefully")
print("   • Enter 'y' to confirm, 'n' to reject, 'r' to remap")
print("   • Enter 'skip' to skip validation (not recommended)")
print("   • Enter 'skip all' to skip all remaining validations")
print("   • Enter 'quit' to exit the system")

validation_passed = True
for logical_col in required_logical_columns:
    actual_col = column_mapping_manager.mapping.get(logical_col, "❌ NOT MAPPED")
    
    if actual_col == "❌ NOT MAPPED" or actual_col not in current_df.columns:
        print(f"\n❌ **CRITICAL**: {logical_col} is not mapped!")
        available_cols = list(current_df.columns)
        for i, col in enumerate(available_cols, 1):
            print(f"   {i:2d}. {col}")
        while True:
            choice = input(f"\n🔗 Select column number for '{logical_col}' (1-{len(available_cols)}) or 'r' to remap: ").strip()
            if choice.lower() == 'r':
                column_mapping_manager.remap_column(logical_col)
                break
            elif choice.lower() == 'quit':
                print("👋 Goodbye!")
                exit(0)
            else:
                try:
                    col_idx = int(choice) - 1
                    if 0 <= col_idx < len(available_cols):
                        column_mapping_manager.mapping[logical_col] = available_cols[col_idx]
                        break
                    else:
                        print("❌ Invalid column number.")
                except ValueError:
                    print("❌ Please enter a valid number or 'r'")
    else:
        sample_values = current_df[actual_col].dropna().head(3).tolist()
        sample_str = ", ".join([str(v)[:20] + "..." if len(str(v)) > 20 else str(v) for v in sample_values])
        print(f"\n📋 **Mapping Review**: {logical_col} → {actual_col}")
        print(f"📊 Sample values: {sample_str}")

        while True:
            confirm = input("✅ Is this mapping correct? (y/n/r/skip/skip all/quit): ").strip().lower()
            if confirm in ['y', 'yes']:
                print(f"✅ Confirmed: {logical_col} → {actual_col}")
                break
            elif confirm in ['n', 'no']:
                validation_passed = False
                print(f"❌ Rejected mapping for {logical_col}")
            elif confirm == 'r':
                column_mapping_manager.remap_column(logical_col)
                break
            elif confirm == 'skip':
                print(f"⚠️ Skipping validation for {logical_col} (not recommended)")
                break
            elif confirm == 'skip all':
                print(f"⚠️ Skipping all remaining validations (not recommended)")
                validation_passed = True
                skip_validation = True
                break
            elif confirm == 'quit':
                print("👋 Goodbye!")
                exit(0)
            else:
                print("❌ Invalid input. Please enter y/n/r/skip/skip all/quit")
        if confirm == 'skip all':
            break

# Step 10: Final validation
if not validation_passed:
    print(f"\n❌ **Validation Failed**")
    print("Some column mappings were rejected. Please restart and correct them.")
    exit(1)

print(f"\n✅ **Final Column Mappings:**")
print("=" * 50)
for logical_col in required_logical_columns:
    actual_col = column_mapping_manager.mapping.get(logical_col, "❌ NOT MAPPED")
    print(f"   • {logical_col} → {actual_col}")

print(f"\n🚀 **Starting Analysis...**")
interactive_analysis(current_df, column_mapping_manager)
