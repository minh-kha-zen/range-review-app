import os
import pandas as pd
from openai import OpenAI

def identify_optimization_levels_dspy(
    hierarchy_df: pd.DataFrame, 
    sous_famille: str, 
    api_key: str, 
    example_models: list,
    model: str
) -> tuple:
    client = OpenAI(api_key=api_key)

    logged_text = ""
    results = []

    # Read existing feedback to include in the prompt and track rejected levels
    feedback_file = "feedback_hierarchy.csv"
    combined_feedback = ""
    rejected_levels = {}  # Dictionary to track rejected levels per Sous-Famille

    if os.path.exists(feedback_file):
        feedback_df = pd.read_csv(feedback_file)
        feedback_logs = feedback_df['Logged Text'].dropna().tolist()
        combined_feedback = "\n".join(feedback_logs)

        # Collect rejected optimization levels per Sous-Famille
        rejected_df = feedback_df[feedback_df['Feedback'] == 'Rejected']
        for _, row in rejected_df.iterrows():
            sf = row['Sous-Famille']
            level_name = row['Optimization Level Name']
            if sf not in rejected_levels:
                rejected_levels[sf] = set()
            rejected_levels[sf].add(level_name)

    # Convert example models to lowercase
    example_models = [model.lower() for model in example_models]

    logged_text += f"\n=== Processing Sous-Famille: {sous_famille} ===\n"

    # Filter hierarchy for the current Sous-Famille
    current_level = hierarchy_df[hierarchy_df['name'] == sous_famille]

    if current_level.empty:
        logged_text += f"Sous-Famille '{sous_famille}' not found in hierarchy. Skipping.\n"
        results.append({
            'Sous-Famille': sous_famille,
            'Optimization Level Name': 'Not Found',
            'Optimization Level ID': 'N/A'
        })
        return pd.DataFrame(results), logged_text  # Return early if not found

    current_material_ids = current_level['material_id'].tolist()
    level_id = current_level.iloc[0]['level']
    optimization_level_name = 'SKU Level'
    optimization_level_id = None

    logged_text += f"Starting at level: {current_level.iloc[0]['level_name']} (ID: {level_id})\n"
    
    


    return pd.DataFrame(results), logged_text