import pandas as pd
from typing import List, Dict
import ast
import os  
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from incompleteness.checker import IncompletenessChecker
from incompleteness.config import create_dynamic_system_config, DYNAMIC_SYSTEMS_CONFIG
from incompleteness.pipeline import IncompletenessPipeline

from inconsistency.pipeline import InconsistencyPipeline
from inconsistency.config import create_dynamic_system_config as create_inconsistency_system_config, create_dynamic_field_mappings, PipelineConfig, DYNAMIC_FIELD_MAPPINGS

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
            if input_str:
                # Simple parameter parsing - can be enhanced
                if '=' in input_str:
                    for param in input_str.split(','):
                        if '=' in param:
                            key, value = param.split('=', 1)
                            params[key.strip()] = value.strip()
            
            # Get current data
            df, mapping_manager = get_tool_data()
            
            if df is None or df.empty:
                return {"error": f"No data loaded for {tool_name}"}
            
            # Call the tool function
            result = tool_func(df, **params)
            
            # Enhanced response formatting
            if isinstance(result, dict):
                if 'error' in result:
                    return result
                else:
                    return {
                        "success": True,
                        "tool": tool_name,
                        "data": result,
                        "summary": f"Successfully executed {tool_name} on {len(df)} records"
                    }
            else:
                return {
                    "success": True,
                    "tool": tool_name,
                    "data": result,
                    "summary": f"Successfully executed {tool_name} on {len(df)} records"
                }
                
        except Exception as e:
            return {
                "error": f"Error in {tool_name}: {str(e)}",
                "tool": tool_name
            }
    
    return wrapper

def get_available_systems() -> List[str]:
    """Get list of available systems for analysis"""
    return ["LIMS", "MES", "QMS", "Change Control", "Generic"]

def str_to_bool(s: str) -> bool:
    """Convert string to boolean"""
    return s.lower() in ('true', '1', 'yes', 'on', 'y')

def list_system_columns(system_name: str, config_type: str = 'incompleteness') -> list:
    """List available columns for a system"""
    # This would typically load from a configuration file
    # For now, return a generic list
    return ["ID", "Name", "Status", "Date", "Description"]

def select_multiple_datasets() -> Dict[str, pd.DataFrame]:
    """
    Interactive function to select multiple datasets for inconsistency checking.
    
    Returns:
        Dictionary of {dataset_name: DataFrame} pairs
    """
    import os
    
    print("🔍 **Multi-Dataset Selection for Inconsistency Analysis**")
    print("=" * 60)
    
    # Look for CSV files in current directory
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if len(csv_files) < 2:
        print("❌ At least 2 CSV files required for multi-dataset inconsistency checking.")
        print(f"💡 Found only {len(csv_files)} CSV file(s) in current directory.")
        return {}
    
    print("📁 Available datasets:")
    print("-" * 50)
    
    # Analyze each CSV file
    file_info = []
    for i, file in enumerate(csv_files, 1):
        try:
            df = pd.read_csv(file)
            file_info.append({
                'file': file,
                'df': df,
                'records': len(df),
                'columns': len(df.columns),
                'sample_columns': list(df.columns)[:5]
            })
            
            print(f"   {i}. 📊 {file}")
            print(f"      📈 Records: {len(df):,}")
            print(f"      📋 Columns: {len(df.columns)}")
            print(f"      🔍 Sample columns: {', '.join(list(df.columns)[:5])}")
            print()
            
        except Exception as e:
            print(f"   {i}. ⚠️ {file} (Error reading: {str(e)[:50]}...)")
            print()
    
    print(f"💡 Select 2 or more datasets for inconsistency analysis")
    
    selected_datasets = {}
    while len(selected_datasets) < 2:
        try:
            choice = input(f"Enter dataset number (1-{len(csv_files)}) or 'done' if you have selected at least 2: ").strip()
            
            if choice.lower() == 'done':
                if len(selected_datasets) >= 2:
                    break
                else:
                    print(f"❌ Please select at least 2 datasets. Currently selected: {len(selected_datasets)}")
                    continue
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(csv_files):
                selected_file = csv_files[choice_num - 1]
                if selected_file not in selected_datasets:
                    selected_datasets[selected_file] = file_info[choice_num - 1]['df']
                    print(f"✅ Added: {selected_file}")
                else:
                    print(f"⚠️ {selected_file} already selected")
            else:
                print(f"❌ Please enter a number between 1 and {len(csv_files)}")
        except ValueError:
            print("❌ Please enter a valid number or 'done'")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return {}
    
    print(f"\n✅ Selected {len(selected_datasets)} datasets for inconsistency analysis")
    return selected_datasets

def llm_visualize_data(df: pd.DataFrame, user_request: str, llm) -> dict:
    """
    Universal visualization tool that uses LLM to interpret ANY user request and generate the appropriate chart and CSV.
    Can handle any chart type, any data column, and any visualization request.

    Args:
        df: The pandas DataFrame to visualize
        user_request: The user's natural language visualization request
        llm: The LLM instance
    Returns:
        dict with chart_file, csv_file, summary, and details
    """
    if df is None or df.empty:
        return {"error": "No data provided for visualization"}
    if not user_request or not llm:
        return {"error": "User request and LLM instance required"}

    print(f"🧠 **LLM-Driven Dynamic Analysis**")
    print("=" * 60)
    print(f"📋 Request: {user_request}")
    
    # Generate timestamp first
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Step 1: Use LLM to parse the request and understand what to visualize
    prompt = f"""
    You are a data visualization expert. Given the following DataFrame columns:
    {list(df.columns)}
    
    And the user request: '{user_request}'
    
    Analyze the request and determine:
    1. What type of chart should be created? (bar, line, box, pie, histogram, scatter, heatmap, etc.)
    2. Which column(s) should be used for x-axis, y-axis, grouping, or aggregation?
    3. Should any aggregation (mean, sum, count, median, etc.) be applied?
    4. Should any filters be applied?
    5. What should the title be?
    
    Consider the data types and choose the most appropriate visualization. For example:
    - Categorical data → bar chart, pie chart
    - Numerical data → histogram, box plot, line chart
    - Two numerical variables → scatter plot
    - Time series → line chart
    - Multiple categories → grouped bar chart
    
    Respond ONLY in this JSON format:
    {{
      "chart_type": "bar/line/box/pie/histogram/scatter/heatmap",
      "x": "column_name or null",
      "y": "column_name or null",
      "group_by": "column_name or null",
      "aggregation": "mean/sum/count/median/none",
      "filter": "filter_expression or null",
      "title": "Chart Title"
    }}
    """
    
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()
        viz_config = json.loads(content)
    except Exception as e:
        return {"error": f"LLM parsing failed: {e}"}

    # Step 2: Prepare data according to config
    dfx = df.copy()
    
    # Apply filter if specified
    if viz_config.get('filter') and viz_config['filter'] != 'null':
        try:
            dfx = dfx.query(viz_config['filter'])
        except Exception:
            pass
    
    # Get column names
    x = viz_config.get('x') if viz_config.get('x') != 'null' else None
    y = viz_config.get('y') if viz_config.get('y') != 'null' else None
    group_by = viz_config.get('group_by') if viz_config.get('group_by') != 'null' else None
    aggregation = viz_config.get('aggregation', 'none')
    chart_type = viz_config.get('chart_type', 'bar')
    title = viz_config.get('title', user_request)

    # Step 3: Handle different chart types and data preparation
    if chart_type == 'histogram':
        # For histograms, we need a single numerical column
        if y and y in dfx.columns:
            data_for_viz = dfx[y].dropna()
        elif x and x in dfx.columns:
            data_for_viz = dfx[x].dropna()
        else:
            # Find first numerical column
            numeric_cols = dfx.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                data_for_viz = dfx[numeric_cols[0]].dropna()
                y = numeric_cols[0]
            else:
                return {"error": "No numerical columns found for histogram"}
    elif chart_type == 'pie':
        # For pie charts, we need categorical data
        if x and x in dfx.columns:
            data_for_viz = dfx[x].value_counts()
        elif group_by and group_by in dfx.columns:
            data_for_viz = dfx[group_by].value_counts()
        else:
            # Find first categorical column
            categorical_cols = dfx.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                data_for_viz = dfx[categorical_cols[0]].value_counts()
                x = categorical_cols[0]
            else:
                return {"error": "No categorical columns found for pie chart"}
    else:
        # For other charts, aggregate if needed
        if aggregation and aggregation != 'none' and y and (group_by or x):
            group_col = group_by or x
            try:
                grouped = dfx.groupby(group_col)[y]
                if aggregation == 'mean':
                    data_for_viz = grouped.mean().reset_index()
                elif aggregation == 'sum':
                    data_for_viz = grouped.sum().reset_index()
                elif aggregation == 'count':
                    data_for_viz = grouped.count().reset_index()
                elif aggregation == 'median':
                    data_for_viz = grouped.median().reset_index()
                else:
                    data_for_viz = grouped.apply(list).reset_index()
            except Exception as e:
                return {"error": f"Aggregation failed: {e}"}
        else:
            data_for_viz = dfx

    # Step 4: Generate chart using matplotlib/seaborn
    import matplotlib.pyplot as plt
    import seaborn as sns
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    chart_file = f"data_quality_reports/llm_chart_{timestamp}.png"
    csv_file = f"data_quality_reports/llm_chart_data_{timestamp}.csv"
    os.makedirs(os.path.dirname(chart_file), exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    try:
        if chart_type == 'bar':
            if isinstance(data_for_viz, pd.Series):
                data_for_viz.plot(kind='bar')
            else:
                plt.bar(data_for_viz[x], data_for_viz[y])
            plt.xticks(rotation=45)
            
        elif chart_type == 'line':
            if isinstance(data_for_viz, pd.Series):
                data_for_viz.plot(kind='line', marker='o')
            else:
                plt.plot(data_for_viz[x], data_for_viz[y], marker='o')
            plt.xticks(rotation=45)
            
        elif chart_type == 'pie':
            plt.pie(data_for_viz.values, labels=data_for_viz.index, autopct='%1.1f%%')
            
        elif chart_type == 'histogram':
            plt.hist(data_for_viz, bins=20, edgecolor='black')
            
        elif chart_type == 'scatter':
            plt.scatter(data_for_viz[x], data_for_viz[y])
            
        elif chart_type == 'box':
            if group_by and group_by in dfx.columns:
                dfx.boxplot(column=y, by=group_by)
            else:
                dfx[y].plot(kind='box')
                
        elif chart_type == 'heatmap':
            # Create correlation matrix for numerical columns
            numeric_df = dfx.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
            else:
                return {"error": "Need at least 2 numerical columns for heatmap"}
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save data to CSV
        if isinstance(data_for_viz, pd.Series):
            data_for_viz.to_csv(csv_file)
        else:
            data_for_viz.to_csv(csv_file, index=False)
            
        return {
            "success": True,
            "chart_file": chart_file,
            "csv_file": csv_file,
            "config": viz_config,
            "summary": f"Visualization complete! Chart: {chart_file}, Data: {csv_file}"
        }
        
    except Exception as e:
        return {"error": f"Chart generation failed: {e}"}

def llm_generate_analysis(df: pd.DataFrame, user_request: str, llm) -> dict:
    """
    LLM-driven dynamic analysis and code generation tool that can create any type of analysis,
    visualization, or report based on natural language requests.

    Args:
        df: The pandas DataFrame to analyze
        user_request: The user's natural language analysis request
        llm: The LLM instance
    Returns:
        dict with analysis results, generated files, and code
    """
    if df is None or df.empty:
        return {"error": "No data provided for analysis"}
    if not user_request or not llm:
        return {"error": "User request and LLM instance required"}

    print(f"🧠 **LLM-Driven Dynamic Analysis**")
    print("=" * 60)
    print(f"📋 Request: {user_request}")
    
    # Generate timestamp first
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Step 1: Use LLM to analyze the request and generate analysis plan
    analysis_prompt = f"""
    You are an expert data analyst and Python programmer. Given the following DataFrame columns:
    {list(df.columns)}
    
    And the user request: '{user_request}'
    
    Analyze the request and generate a complete Python analysis plan that includes:
    1. What type of analysis is needed (summary, visualization, statistical analysis, etc.)
    2. What data preprocessing is required
    3. What calculations or aggregations to perform
    4. What visualizations to create (if any)
    5. What insights to extract
    
    Respond with ONLY a Python dictionary in this format:
    {{
        "analysis_type": "summary/visualization/statistical/correlation/trend/temporal",
        "data_preprocessing": ["step1", "step2", ...],
        "calculations": ["calc1", "calc2", ...],
        "visualizations": ["viz1", "viz2", ...],
        "insights_to_extract": ["insight1", "insight2", ...],
        "output_format": "text/chart/csv/both"
    }}
    """
    
    try:
        response = llm.invoke(analysis_prompt)
        content = response.content.strip()
        print("DEBUG: Raw LLM response for analysis plan:", repr(content))  # <-- Add this line
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()
        analysis_plan = json.loads(content)
    except Exception as e:
        print("DEBUG: LLM analysis planning failed, raw content:", repr(content))  # <-- Add this line
        return {
            "error": f"LLM analysis planning failed: {e}",
            "raw_llm_response": content
        }

    # Step 2: Use LLM to generate the actual Python code
    code_generation_prompt = f"""
    You are an expert Python programmer. Generate complete, executable Python code for the following analysis plan:
    
    Analysis Plan: {analysis_plan}
    
    DataFrame columns: {list(df.columns)}
    DataFrame name: df
    
    Requirements:
    1. Generate complete, runnable Python code
    2. Include all necessary imports (pandas, matplotlib, seaborn, numpy)
    3. Handle missing data and errors gracefully
    4. Create meaningful visualizations if requested
    5. Save results to files (CSV and PNG if applicable)
    6. Return a dictionary with results, file paths, and insights
    
    The code should:
    - Use 'df' as the DataFrame variable
    - Save charts to 'data_quality_reports/llm_analysis_{timestamp}.png'
    - Save data to 'data_quality_reports/llm_analysis_{timestamp}.csv'
    - Return a dictionary with 'success', 'results', 'files', 'insights'
    
    Generate ONLY the Python code, no explanations.
    """
    
    try:
        response = llm.invoke(code_generation_prompt)
        generated_code = response.content.strip()
        
        # Clean up the code if it's in a code block
        if generated_code.startswith('```python'):
            generated_code = generated_code.replace('```python', '').replace('```', '').strip()
        elif generated_code.startswith('```'):
            generated_code = generated_code.replace('```', '').strip()
            
    except Exception as e:
        return {"error": f"LLM code generation failed: {e}"}

    # Step 3: Execute the generated code
    try:
        # Create a safe execution environment
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import os
        
        # Create output directory
        os.makedirs("data_quality_reports", exist_ok=True)
        
        # Execute the generated code
        exec_globals = {
            'df': df,
            'pd': pd,
            'plt': plt,
            'sns': sns,
            'np': np,
            'os': os,
            'timestamp': timestamp,
            'json': json
        }
        
        exec(generated_code, exec_globals)
        
        # Get the result from the executed code
        if 'result' in exec_globals:
            result = exec_globals['result']
        else:
            result = {"success": True, "message": "Analysis completed successfully"}
            
        print(f"✅ LLM-generated analysis completed successfully")
        return result
        
    except Exception as e:
        return {"error": f"Code execution failed: {e}", "generated_code": generated_code}

def fix_series_boolean_errors(code: str) -> str:
    """
    Fix common 'truth value of a Series is ambiguous' issues in generated code.
    Replaces 'if df[...]==...' with '(df[...]==...).any()' in if statements.
    """
    import re
    pattern = r"if\s+(df\[[^\]]+\]\s*==\s*[^:\n]+)"
    fixed_code = re.sub(pattern, r"if (\1).any()", code)
    return fixed_code

def llm_generate_visualization_code(df: pd.DataFrame, user_request: str, llm) -> dict:
    """
    LLM-driven dynamic visualization code generator that creates any chart based on user request.

    Args:
        df: The pandas DataFrame to visualize
        user_request: The user's natural language visualization request
        llm: The LLM instance
    Returns:
        dict with visualization results and file paths
    """
    if df is None or df.empty:
        return {"error": "No data provided for visualization"}
    if not user_request or not llm:
        return {"error": "User request and LLM instance required"}

    print(f"🎨 **LLM-Driven Dynamic Visualization**")
    print("=" * 60)
    print(f"📋 Request: {user_request}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prompt for LLM
    viz_prompt = f"""
You are an expert data visualization programmer. Generate Python code for the following visualization task:

🧾 Request: {user_request}
📊 DataFrame columns: {list(df.columns)}
📌 DataFrame variable name: df

Your code must:
1. Preprocess the data as needed.
2. Create the visualization using matplotlib/seaborn.
3. Save the chart as PNG to: 'data_quality_reports/llm_viz_{timestamp}.png'
4. Save the data used for visualization as CSV to: 'data_quality_reports/llm_viz_{timestamp}.csv'
5. Return a dictionary named `result` containing:
    {{
        'chart_file': 'file_path',
        'csv_file': 'file_path',
        'summary': 'summary string'
    }}

⚠️ IMPORTANT:
- NEVER write: `if df['col'] == 'value'`
- INSTEAD use: `if (df['col'] == 'value').any()` to avoid Series ambiguity
- Do not include explanations, comments, or markdown. Only return valid Python code.
"""

    # Step 1: Call LLM to generate code
    try:
        response = llm.invoke(viz_prompt)
        generated_code = response.content.strip()
        if generated_code.startswith("```python"):
            generated_code = generated_code.replace("```python", "").replace("```", "").strip()
        elif generated_code.startswith("```"):
            generated_code = generated_code.replace("```", "").strip()
    except Exception as e:
        return {"error": f"LLM visualization code generation failed: {e}"}

    # Step 2: Fix known LLM issues (e.g., Series truth value bug)
    try:
        cleaned_code = fix_series_boolean_errors(generated_code)
    except Exception as e:
        return {"error": f"Post-processing of generated code failed: {e}", "generated_code": generated_code}

    # Step 3: Execute the generated code
    try:
        os.makedirs("data_quality_reports", exist_ok=True)
        exec_globals = {
            'df': df.copy(),
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'os': os,
            'datetime': datetime,
            'timestamp': timestamp
        }

        exec(cleaned_code, exec_globals)
        result = exec_globals.get("result", {
            "chart_file": f"data_quality_reports/llm_viz_{timestamp}.png",
            "csv_file": f"data_quality_reports/llm_viz_{timestamp}.csv",
            "summary": "Visualization completed, but no structured result returned from code."
        })

        print("✅ LLM-generated visualization completed successfully.")
        return result

    except Exception as e:
        return {
            "error": f"Visualization code execution failed: {e}",
            "generated_code": cleaned_code
        }
def llm_cycle_time_analysis(df: pd.DataFrame, llm) -> dict:
    """
    LLM-driven cycle time analysis that automatically adapts to any dataset and column names.

    Args:
        df: The pandas DataFrame to analyze
        llm: The LLM instance
    Returns:
        dict with cycle time analysis results
    """
    if df is None or df.empty:
        return {"error": "No data provided for cycle time analysis"}
    if not llm:
        return {"error": "LLM instance required"}

    print(f"📈 **LLM-Driven Cycle Time Analysis**")
    print("=" * 60)
    
    # Use LLM to identify cycle time related columns
    column_analysis_prompt = f"""
    You are an expert data analyst. Given these DataFrame columns:
    {list(df.columns)}
    
    Identify which columns are related to cycle time analysis:
    1. Date columns (creation, start, implementation, completion dates)
    2. Time-related columns (duration, cycle time, processing time)
    3. Change type or category columns
    4. Status or phase columns
    5."Implementation date" ( DO NOT confuse with 'Approval date')
    
    Return ONLY a JSON dictionary with these keys:
    {{
        "creation_date_col": "column_name_or_null",
        "implementation_date_col": "column_name_or_null", 
        "cycle_time_col": "column_name_or_null",
        "change_type_col": "column_name_or_null",
        "status_col": "column_name_or_null"
    }}
    
    If a column type is not found, use null. Be smart about column name variations.
    """
    
    try:
        response = llm.invoke(column_analysis_prompt)
        content = response.content.strip()
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()
        column_mapping = json.loads(content)
    except Exception as e:
        return {"error": f"LLM column analysis failed: {e}"}

    # Generate analysis code based on identified columns
    analysis_prompt = f"""
    You are an expert Python data analyst. Generate code to analyze cycle times with these columns:
    {column_mapping}
    
    DataFrame name: df
    
    Generate Python code that:
    1. Calculates cycle times if not already present
    2. Analyzes cycle times by change type (if available)
    3. Identifies changes above average
    4. Provides summary statistics
    5. Returns a dictionary with results
    
    Handle missing columns gracefully. Return ONLY the Python code.
    """
    
    try:
        response = llm.invoke(analysis_prompt)
        generated_code = response.content.strip()
        
        if generated_code.startswith('```python'):
            generated_code = generated_code.replace('```python', '').replace('```', '').strip()
        elif generated_code.startswith('```'):
            generated_code = generated_code.replace('```', '').strip()
            
    except Exception as e:
        return {"error": f"LLM analysis code generation failed: {e}"}

    # Execute the generated analysis
    try:
        exec_globals = {
            'df': df,
            'pd': pd,
            'np': np,
            'json': json
        }
        
        exec(generated_code, exec_globals)
        
        if 'result' in exec_globals:
            result = exec_globals['result']
        else:
            result = {"success": True, "message": "Cycle time analysis completed"}
            
        print(f"✅ LLM-driven cycle time analysis completed successfully")
        return result
        
    except Exception as e:
        return {"error": f"Cycle time analysis execution failed: {e}", "generated_code": generated_code}

def llm_root_cause_analysis(df: pd.DataFrame, llm) -> dict:
    """
    LLM-driven root cause analysis that automatically adapts to any dataset and column names.

    Args:
        df: The pandas DataFrame to analyze
        llm: The LLM instance
    Returns:
        dict with root cause analysis results
    """
    if df is None or df.empty:
        return {"error": "No data provided for root cause analysis"}
    if not llm:
        return {"error": "LLM instance required"}

    print(f"🔍 **LLM-Driven Root Cause Analysis**")
    print("=" * 60)
    
    # Use LLM to identify relevant columns for root cause analysis
    column_analysis_prompt = f"""
    You are an expert data analyst. Given these DataFrame columns:
    {list(df.columns)}
    
    Identify which columns are relevant for root cause analysis:
    1. Department or team columns
    2. Status or phase columns (open, pending, delayed, etc.)
    3. Date columns (creation, implementation, etc.)
    4. Priority or severity columns
    5. Category or type columns
    
    Return ONLY a JSON dictionary with these keys:
    {{
        "department_col": "column_name_or_null",
        "status_col": "column_name_or_null",
        "creation_date_col": "column_name_or_null",
        "priority_col": "column_name_or_null",
        "category_col": "column_name_or_null"
    }}
    
    If a column type is not found, use null. Be smart about column name variations.
    """
    
    try:
        response = llm.invoke(column_analysis_prompt)
        content = response.content.strip()
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()
        column_mapping = json.loads(content)
    except Exception as e:
        return {"error": f"LLM column analysis failed: {e}"}

    # Generate root cause analysis code
    analysis_prompt = f"""
    You are an expert Python data analyst. Generate code for root cause analysis with these columns:
    {column_mapping}
    
    DataFrame name: df
    
    Generate Python code that:
    1. Identifies departments/teams with most delays/issues
    2. Analyzes delay patterns by status
    3. Finds bottlenecks and problem areas
    4. Provides actionable insights
    5. Returns a dictionary with results
    
    Handle missing columns gracefully. Return ONLY the Python code.
    """
    
    try:
        response = llm.invoke(analysis_prompt)
        generated_code = response.content.strip()
        
        if generated_code.startswith('```python'):
            generated_code = generated_code.replace('```python', '').replace('```', '').strip()
        elif generated_code.startswith('```'):
            generated_code = generated_code.replace('```', '').strip()
            
    except Exception as e:
        return {"error": f"LLM analysis code generation failed: {e}"}

    # Execute the generated analysis
    try:
        exec_globals = {
            'df': df,
            'pd': pd,
            'np': np,
            'json': json
        }
        
        exec(generated_code, exec_globals)
        
        if 'result' in exec_globals:
            result = exec_globals['result']
        else:
            result = {"success": True, "message": "Root cause analysis completed"}
            
        print(f"✅ LLM-driven root cause analysis completed successfully")
        return result
        
    except Exception as e:
        return {"error": f"Root cause analysis execution failed: {e}", "generated_code": generated_code}


def llm_kpi_analysis(df: pd.DataFrame, llm) -> dict:
    """
    LLM-driven KPI analysis that automatically adapts to any dataset and column names.
    
    Args:
        df: The pandas DataFrame to analyze
        llm: The LLM instance
    Returns:
        dict with KPI analysis results and files
    """
    if df is None or df.empty:
        return {"error": "No data provided for KPI analysis"}
    if not llm:
        return {"error": "LLM instance required"}

    print(f"📊 **LLM-Driven KPI Analysis**")
    print("=" * 60)
    
    # Generate timestamp for files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "data_quality_reports"
    os.makedirs(output_dir, exist_ok=True)
    csv_file = f"{output_dir}/kpi_analysis_{timestamp}.csv"
    chart_file = f"{output_dir}/kpi_analysis_{timestamp}.png"
    print(f"📂 Output files will be saved to: {output_dir}")   
     
    # Use LLM to identify KPI-related columns
    column_analysis_prompt = f"""
    You are an expert data analyst. Given these DataFrame columns:
    {list(df.columns)}
    
    Identify which columns are relevant for KPI analysis:
    1. Date columns (creation, implementation, completion)
    2. Status columns (open, closed, pending, approved)
    3. Department or team columns
    4. Priority or severity columns
    5. Time-related columns (duration, cycle time)
    6. Category or type columns
    
    Return ONLY a JSON dictionary with these keys:
    {{
        "creation_date_col": "column_name_or_null",
        "implementation_date_col": "column_name_or_null",
        "status_col": "column_name_or_null",
        "department_col": "column_name_or_null",
        "priority_col": "column_name_or_null",
        "cycle_time_col": "column_name_or_null",
        "category_col": "column_name_or_null"
    }}
    
    If a column type is not found, use null. Be smart about column name variations.
    """
    
    try:
        response = llm.invoke(column_analysis_prompt)
        content = response.content.strip()
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()
        column_mapping = json.loads(content)
    except Exception as e:
        return {"error": f"LLM column analysis failed: {e}"}

    # Generate KPI analysis and visualization code
    analysis_prompt = f"""
    You are an expert Python data analyst. Generate code for comprehensive KPI analysis with these columns:
    {column_mapping}
    
    DataFrame name: df
    Timestamp: {timestamp}
    
    Generate Python code that:
    1. Calculates key performance indicators
    2. Analyzes on-time performance
    3. Tracks pending approvals
    4. Creates visualizations (charts and CSV files)
    5. Saves results to 'data_quality_reports/kpi_analysis_{timestamp}.png' and 'data_quality_reports/kpi_analysis_{timestamp}.csv'
    6. Returns a dictionary with results and file paths
    
    Include matplotlib/seaborn for visualizations. Handle missing columns gracefully.
    Return ONLY the Python code.
    """
    import os
    try:
        response = llm.invoke(analysis_prompt)
        generated_code = response.content.strip()
        
        if generated_code.startswith('```python'):
            generated_code = generated_code.replace('```python', '').replace('```', '').strip()
        elif generated_code.startswith('```'):
            generated_code = generated_code.replace('```', '').strip()
            
    except Exception as e:
        return {"error": f"LLM analysis code generation failed: {e}"}

    # Execute the generated analysis
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        
        # Create output directory
        os.makedirs("data_quality_reports", exist_ok=True)
        
        exec_globals = {
            'df': df,
            'pd': pd,
            'plt': plt,
            'sns': sns,
            'np': np,
            'os': os,
            'timestamp': timestamp,
            'json': json
        }
        
        exec(generated_code, exec_globals)
        
        if 'result' in exec_globals:
            result = exec_globals['result']
        else:
            result = {"success": True, "message": "KPI analysis completed"}
            
        print(f"✅ LLM-driven KPI analysis completed successfully")
        return result
        
    except Exception as e:
        return {"error": f"KPI analysis execution failed: {e}", "generated_code": generated_code}

def llm_track_pending_approvals(df: pd.DataFrame, llm) -> dict:
    """
    Dynamically tracks changes waiting for authority approval using LLM to:
    - Identify correct column names
    - Understand status value meanings
    - Plan filtering and summary strategy
    """
    if df.empty:
        return {"error": "Dataset is empty"}
    if not llm:
        return {"error": "LLM instance required"}

    # 🧠 Step 1: Ask LLM to identify relevant columns and status values
    prompt = f"""
You are an expert AI analyst. You are analyzing a DataFrame with columns:
{list(df.columns)}

Your job is to:
1. Identify the column that contains Authority Approval Status
2. Determine the values in that column which represent "waiting for approval"
3.From that column, extract values that mean the **approval is still pending or not granted yet**  
Examples: "Pending", "Pending Submission", "Waiting", "Under Review", etc.
3. Identify the columns for Change Type and Department (if available)
4. Suggest how to compute number of pending approvals and breakdown by type/department

Return a valid JSON like:
{{
  "approval_status_col": "column_name",
  "waiting_values": ["value1", "value2", ...],
  "change_type_col": "column_name_or_null",
  "department_col": "column_name_or_null"
}}
"""
    try:
        plan_resp = llm.invoke(prompt)
        raw_content = plan_resp.content.strip().replace("```json", "").replace("```", "")
        plan = json.loads(raw_content)
    except Exception as e:
        return {"error": f"LLM planning failed: {e}"}

    # 🔧 Step 2: Run analysis based on LLM plan
    try:
        status_col = plan["approval_status_col"]
        waiting_values = [v.lower() for v in plan.get("waiting_values", [])]
        change_type_col = plan.get("change_type_col")
        department_col = plan.get("department_col")

        if status_col not in df.columns:
            return {"error": f"Approval status column '{status_col}' not found in dataset"}

        status_series = df[status_col].astype(str).str.lower()
        waiting_df = df[status_series.isin(waiting_values)]

        result = {
            "total_pending_approvals": len(waiting_df),
            "approval_status_column": status_col,
            "waiting_status_values": waiting_values
        }

        if change_type_col and change_type_col in waiting_df.columns:
            result["breakdown_by_type"] = (
                waiting_df[change_type_col].value_counts().to_dict()
            )

        if department_col and department_col in waiting_df.columns:
            result["breakdown_by_department"] = (
                waiting_df[department_col].value_counts().to_dict()
            )

        return result
    except Exception as e:
        return {"error": f"Execution failed: {str(e)}"}


def llm_forecast_analysis(df: pd.DataFrame, llm) -> dict:
    """
    LLM-driven forecast analysis that automatically adapts to any dataset and column names.
    
    Args:
        df: The pandas DataFrame to analyze
        llm: The LLM instance
    Returns:
        dict with forecast analysis results
    """
    if df is None or df.empty:
        return {"error": "No data provided for forecast analysis"}
    if not llm:
        return {"error": "LLM instance required"}

    print(f"🔮 **LLM-Driven Forecast Analysis**")
    print("=" * 60)
    
    # Use LLM to identify time-series related columns
    column_analysis_prompt = f"""
    You are an expert data analyst. Given these DataFrame columns:
    {list(df.columns)}
    
    Identify which columns are relevant for forecasting:
    1. Date columns (creation, implementation, completion)
    2. Time-related columns (duration, cycle time, processing time)
    3. Status or phase columns
    4. Category or type columns
    
    Return ONLY a JSON dictionary with these keys:
    {{
        "date_col": "column_name_or_null",
        "cycle_time_col": "column_name_or_null",
        "status_col": "column_name_or_null",
        "category_col": "column_name_or_null"
    }}
    
    If a column type is not found, use null. Be smart about column name variations.
    """
    
    try:
        response = llm.invoke(column_analysis_prompt)
        content = response.content.strip()
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()
        column_mapping = json.loads(content)
    except Exception as e:
        return {"error": f"LLM column analysis failed: {e}"}

    # Generate forecast analysis code
    analysis_prompt = f"""
    You are an expert Python data analyst. Generate code for forecasting with these columns:
    {column_mapping}
    
    DataFrame name: df
    
    Generate Python code that:
    1. Analyzes time trends in the data
    2. Identifies patterns and seasonality
    3. Makes predictions for future performance
    4. Provides confidence intervals
    5. Returns a dictionary with forecast results
    
    Handle missing columns gracefully. Use statistical methods for forecasting.
    Return ONLY the Python code.
    """
    
    try:
        response = llm.invoke(analysis_prompt)
        generated_code = response.content.strip()
        
        if generated_code.startswith('```python'):
            generated_code = generated_code.replace('```python', '').replace('```', '').strip()
        elif generated_code.startswith('```'):
            generated_code = generated_code.replace('```', '').strip()
            
    except Exception as e:
        return {"error": f"LLM analysis code generation failed: {e}"}

    # Execute the generated analysis
    try:
        exec_globals = {
            'df': df,
            'pd': pd,
            'np': np,
            'json': json
        }
        
        exec(generated_code, exec_globals)
        
        if 'result' in exec_globals:
            result = exec_globals['result']
        else:
            result = {"success": True, "message": "Forecast analysis completed"}
            
        print(f"✅ LLM-driven forecast analysis completed successfully")
        return result
        
    except Exception as e:
        return {"error": f"Forecast analysis execution failed: {e}", "generated_code": generated_code}

def llm_optimization_analysis(df: pd.DataFrame, llm) -> dict:
    """
    LLM-driven optimization analysis that automatically adapts to any dataset and column names.
    
    Args:
        df: The pandas DataFrame to analyze
        llm: The LLM instance
    Returns:
        dict with optimization recommendations
    """
    if df is None or df.empty:
        return {"error": "No data provided for optimization analysis"}
    if not llm:
        return {"error": "LLM instance required"}

    print(f"💡 **LLM-Driven Optimization Analysis**")
    print("=" * 60)
    
    # Use LLM to identify optimization-related columns
    column_analysis_prompt = f"""
    You are an expert data analyst. Given these DataFrame columns:
    {list(df.columns)}
    
    Identify which columns are relevant for optimization analysis:
    1. Performance metrics (cycle time, duration, efficiency)
    2. Resource columns (department, team, person)
    3. Status or phase columns
    4. Priority or severity columns
    5. Category or type columns
    6. Date columns for trend analysis
    
    Return ONLY a JSON dictionary with these keys:
    {{
        "performance_col": "column_name_or_null",
        "resource_col": "column_name_or_null",
        "status_col": "column_name_or_null",
        "priority_col": "column_name_or_null",
        "category_col": "column_name_or_null",
        "date_col": "column_name_or_null"
    }}
    
    If a column type is not found, use null. Be smart about column name variations.
    """
    
    try:
        response = llm.invoke(column_analysis_prompt)
        content = response.content.strip()
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()
        column_mapping = json.loads(content)
    except Exception as e:
        return {"error": f"LLM column analysis failed: {e}"}

    # Generate optimization analysis code
    analysis_prompt = f"""
    You are an expert Python data analyst. Generate code for optimization analysis with these columns:
    {column_mapping}
    
    DataFrame name: df
    
    Generate Python code that:
    1. Identifies bottlenecks and inefficiencies
    2. Analyzes performance by resource/category
    3. Finds optimization opportunities
    4. Provides actionable recommendations
    5. Returns a dictionary with optimization insights
    
    Handle missing columns gracefully. Focus on practical improvements.
    Return ONLY the Python code.
    """
    
    try:
        response = llm.invoke(analysis_prompt)
        generated_code = response.content.strip()
        
        if generated_code.startswith('```python'):
            generated_code = generated_code.replace('```python', '').replace('```', '').strip()
        elif generated_code.startswith('```'):
            generated_code = generated_code.replace('```', '').strip()
            
    except Exception as e:
        return {"error": f"LLM analysis code generation failed: {e}"}

    # Execute the generated analysis
    try:
        exec_globals = {
            'df': df,
            'pd': pd,
            'np': np,
            'json': json
        }
        
        exec(generated_code, exec_globals)
        
        if 'result' in exec_globals:
            result = exec_globals['result']
        else:
            result = {"success": True, "message": "Optimization analysis completed"}
            
        print(f"✅ LLM-driven optimization analysis completed successfully")
        return result
        
    except Exception as e:
        return {"error": f"Optimization analysis execution failed: {e}", "generated_code": generated_code}