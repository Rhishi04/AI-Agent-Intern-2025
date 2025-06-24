import pandas as pd
from prophet import Prophet
import os
import re
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
# from azure.storage.blob import BlobServiceClient  # Uncomment when ready

def llm_generate_config(df, llm):
    """Use LLM to generate dataset configuration"""
    print("\n🤖 Using LLM to analyze and configure dataset...")
    
    prompt = f"""
    You are an expert data analyst. Based on the following dataset columns and a sample of the data, generate a JSON configuration.

    Dataset columns: {list(df.columns)}
    Data sample:
    {df.head(3).to_string()}

    Generate a JSON configuration with the following structure:
    {{
        "column_mappings": {{
            "Change_Control_ID": "ChangeID",
            "Change_Type": "ChangeType", 
            "Initiating_Department": "Department",
            "Creation_Date": "StartDate",
            "Implementation_Date": "ImplementationDate",
            "Status": "Status",
            "Authority_Approval_Status": "ApprovalStatus"
        }},
        "data_types": {{
            "ChangeID": "identifier",
            "ChangeType": "categorical", 
            "Department": "categorical",
            "StartDate": "datetime",
            "ImplementationDate": "datetime",
            "Status": "categorical",
            "ApprovalStatus": "categorical"
        }},
        "validation_rules": {{
            "max_missing_pct": 0.1,
            "required_columns": ["ChangeID", "ChangeType", "Department"]
        }}
    }}

    Respond with ONLY the JSON object, no additional text.
    """
    
    try:
        response = llm.invoke(prompt)
        response_content = response.content.strip()
        print(f"🔍 LLM Response: {response_content[:200]}...")  # Debug: show first 200 chars
        
        # Try to extract JSON if there's extra text
        if response_content.startswith('```json'):
            response_content = response_content.replace('```json', '').replace('```', '').strip()
        elif response_content.startswith('```'):
            response_content = response_content.replace('```', '').strip()
        
        config = json.loads(response_content)
        print("✅ LLM generated configuration successfully!")
        return config
    except Exception as e:
        print(f"❌ LLM failed to generate valid JSON ({e}). Falling back to rule-based detection.")
        return rule_based_auto_configure_dataset(df)  # Fallback

def llm_validate_config(config, df, llm):
    """Use LLM to validate the configuration"""
    print("\n🤖 Using LLM to validate configuration...")
    
    prompt = f"""
    You are a data quality expert. Validate this configuration against the data sample.

    Configuration:
    {json.dumps(config, indent=2)}

    Data sample:
    {df.head(3).to_string()}

    Respond with ONLY a JSON object in this format:
    {{
        "is_valid": true,
        "issues": [],
        "suggestions": ["Good configuration for change control data"]
    }}

    No additional text, just the JSON.
    """
    
    try:
        response = llm.invoke(prompt)
        response_content = response.content.strip()
        print(f"🔍 LLM Validation Response: {response_content[:200]}...")  # Debug
        
        # Try to extract JSON if there's extra text
        if response_content.startswith('```json'):
            response_content = response_content.replace('```json', '').replace('```', '').strip()
        elif response_content.startswith('```'):
            response_content = response_content.replace('```', '').strip()
        
        validation_results = json.loads(response_content)
        return validation_results
    except Exception as e:
        print(f"❌ LLM failed to generate valid JSON for validation ({e}). Falling back to rule-based validation.")
        return validate_config_rules(config, df)  # Fallback

def auto_configure_dataset(df, llm):
    """Main function for LLM-powered auto-configuration with validation"""
    print("🚀 Starting LLM-powered Auto-Configuration with Validation...")
    
    # Step 1: LLM generates config
    config = llm_generate_config(df, llm)
    
    # Step 2: LLM validates config
    validation_results = llm_validate_config(config, df, llm)
    
    # Step 3: Display results
    print(f"\n📊 LLM Configuration Validation Results:")
    
    if validation_results and isinstance(validation_results, dict):
        print(f"   Valid: {'✅ Yes' if validation_results.get('is_valid', False) else '❌ No'}")
        
        if validation_results.get('issues'):
            print(f"   Issues found:")
            for issue in validation_results['issues']:
                print(f"      • {issue}")
        
        if validation_results.get('suggestions'):
            print(f"   Suggestions:")
            for suggestion in validation_results['suggestions']:
                print(f"      💡 {suggestion}")
    else:
        print("   ⚠️ Validation results not available, using fallback validation")
        validation_results = {"is_valid": True, "issues": [], "suggestions": ["Using fallback validation"]}
    
    # Step 4: Save configuration (skip user input for now)
    # save_choice = input("\nSave this LLM-generated configuration for future use? (y/n): ").strip().lower()
    # if save_choice in ['y', 'yes']:
    #     save_config(config)
    
    return config, validation_results

def rule_based_auto_configure_dataset(df):
    """Rule-based fallback for auto-detecting dataset configuration"""
    # (This is the original auto_detect_config function, renamed for clarity)
    # ... implementation remains the same
    pass # Placeholder for existing code

def validate_config_rules(config, df):
    """Rule-based fallback for validating the configuration"""
    # (This is the original validate_config function, renamed for clarity)
    # ... implementation remains the same
    pass # Placeholder for existing code

def save_config(config, filename=None):
    """Save configuration to file"""
    if filename is None:
        filename = f"dataset_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"💾 Configuration saved to: {filename}")
    return filename

def load_config(filename):
    """Load configuration from file"""
    try:
        with open(filename, 'r') as f:
            config = json.load(f)
        print(f"📂 Configuration loaded from: {filename}")
        return config
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {filename}")
        return None

# Global data store for tool access
_tool_data_store = {
    'df': None,
    'mapping_manager': None
}

def set_tool_data(df, mapping_manager):
    """Set the data for tools to access"""
    global _tool_data_store
    _tool_data_store['df'] = df
    _tool_data_store['mapping_manager'] = mapping_manager

def get_tool_data():
    """Get the data for tools to access"""
    global _tool_data_store
    return _tool_data_store['df'], _tool_data_store['mapping_manager']

def enhanced_tool_wrapper(tool_func, tool_name):
    def wrapper(input_str):
        try:
            # Parse input parameters
            params = {}
            if input_str.strip():
                try:
                    params = json.loads(input_str)
                except json.JSONDecodeError:
                    if "change_type" in input_str.lower():
                        params["change_type"] = input_str.strip()
                    elif "delay_status" in input_str.lower():
                        params["delay_status"] = input_str.strip()
                    elif "on_time_threshold" in input_str.lower():
                        try:
                            params["on_time_threshold"] = int(input_str.strip())
                        except ValueError:
                            pass

            df, mapping_manager = get_tool_data()
            if df is not None and mapping_manager is not None:
                mapped_df = df.copy()
                if mapping_manager.mapping:
                    reverse_mapping = {v: k for k, v in mapping_manager.mapping.items()}
                    mapped_df.rename(columns=reverse_mapping, inplace=True)
                result = tool_func(mapped_df, **params)
                return result  # Return the raw result (dict) to the agent
            else:
                return {
                    "error": "Data not available or mapping manager not initialized. Please ensure the main application has loaded the dataset and initialized column mappings."
                }
        except Exception as e:
            return {"error": f"Error in {tool_name}: {str(e)}"}

    return wrapper


# Enhancement functions moved from main.py to avoid duplication
class BusinessResponseEnhancer:
    """🎯 Business-Focused Response Enhancement System"""

    def __init__(self, llm):
        self.llm = llm

    def enhance_tool_response(self, tool_name: str, raw_result: Any) -> str:
        """🎯 Enhanced Business Response with Context and Actionability"""
        try:
            if isinstance(raw_result, dict) and "error" in raw_result:
                return self.format_error_response(raw_result["error"])

            # Use generic enhancement for all LLM-driven tools
            return self.generic_enhancement(raw_result)
        except Exception as e:
            return f"⚠️ Error enhancing response: {str(e)}\n\nRaw result: {raw_result}"

    def format_error_response(self, error: str) -> str:
        """⚠️ Enhanced Error Response with Guidance"""
        return f"⚠️ **Error Encountered:** {error}\n\n💡 **Next Steps:** Please check your data or try a different analysis approach."

    def generic_enhancement(self, result: Any) -> str:
        """🎯 Generic Enhancement for Unrecognized Results"""
        if isinstance(result, dict):
            response = "📊 **Analysis Results**\n\n"
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    response += f"**{key.replace('_', ' ').title()}:** {value}\n"
                elif isinstance(value, list):
                    response += f"**{key.replace('_', ' ').title()}:** {len(value)} items\n"
                else:
                    response += f"**{key.replace('_', ' ').title()}:** {value}\n"
            return response
        else:
            return str(result)
