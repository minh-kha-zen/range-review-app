import os
import pandas as pd
from openai import OpenAI

def extract_models_for_sub_family(hierarchy_df: pd.DataFrame, sous_famille: str, api_key: str, example_models: list) -> tuple:
    """
    Extracts all models for a given sous_famille from the hierarchy using ChatGPT API.

    Args:
        hierarchy_df (pd.DataFrame): The hierarchy dataframe.
        sous_famille (str): The sub-family to process.
        api_key (str): OpenAI API key.
        example_models (list): List of example model names.

    Returns:
        tuple: A tuple containing the results dataframe and the logged text.
    """
    client = OpenAI(api_key=api_key)

    logged_text = ""
    results = []

    # Read existing feedback to include in the prompt and track rejected levels
    feedback_file = "feedback_model.csv"  # Updated to match Streamlit app
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

    logged_text += f"\n=== Extracting Models for Sous-Famille: {sous_famille} ===\n"

    # Get specific feedback for the current sous_famille
    if os.path.exists(feedback_file):
        your_feedback = feedback_df[feedback_df['Sous-Famille'] == sous_famille]['Your Feedback'].dropna().tolist()
        specific_feedback = "\n".join(your_feedback) if your_feedback else ""
    else:
        specific_feedback = ""

    # Retrieve rejected levels for the current Sous-Famille
    rejected_for_sf = rejected_levels.get(sous_famille, set())

    # Filter hierarchy for the current Sous-Famille
    current_level = hierarchy_df[hierarchy_df['name'].str.lower() == sous_famille.lower()]

    if current_level.empty:
        logged_text += f"Sous-Famille '{sous_famille}' not found in hierarchy. Skipping.\n"
        results.append({
            'Sous-Famille': sous_famille,
            'model_name': 'Not Found',
            'model_level': 'N/A',
            'model_level_name': 'N/A'
        })
        return pd.DataFrame(results), logged_text  # Return early if not found

    current_material_ids = current_level['material_id'].tolist()
    level_id = current_level.iloc[0]['level']
    level_name = current_level.iloc[0]['level_name']
    logged_text += f"Starting at level: {level_name} (ID: {level_id})\n"

    # Track material_ids to exclude variants of identified models in lower levels
    excluded_material_ids = set()

    while True:
        next_level = level_id + 1
        child_entities = hierarchy_df[
            (hierarchy_df['material_id'].isin(current_material_ids)) &
            (hierarchy_df['level'] == next_level)
        ]

        if child_entities.empty:
            logged_text += f"No further child entities found at level {next_level}. Extraction complete.\n"
            break

        # Exclude materials that are variants of already identified models
        child_entities = child_entities[~child_entities['material_id'].isin(excluded_material_ids)]

        if child_entities.empty:
            logged_text += f"All child entities at level {next_level} are excluded. Extraction complete.\n"
            break

        # Process entity names to keep only the part after the hyphen (if any) and convert to lowercase
        entity_names = child_entities['name'].unique().tolist()
        processed_entity_names = [name.split('-')[1].strip().lower() if '-' in name else name.strip().lower() for name in entity_names]

        strings_to_remove = [
            's/m', 
            'std', 
            'obso', 
            'expo',
            '(acc+pcd)'
        ]
        processed_entity_names = [
            name for name in processed_entity_names 
            if not any(rem in name for rem in strings_to_remove)
        ]
        processed_entity_names = list(set(processed_entity_names))  # Remove duplicates
        level_name = child_entities.iloc[0]['level_name']

        logged_text += f"Entities at level {next_level} ({level_name}): {processed_entity_names}\n"

        # Convert example models list to a lowercase string for the prompt
        example_models_str = ', '.join(example_models)

        prompt = f"""
        You are an expert in product portfolio management analyzing a product hierarchy. 
        Your task is to identify all models in a list of entities.
        Models are specific product variants (e.g., amandie, cancale, molene).
        I am providing you with a list of all models that you can use to identify the models from the list of entities.
        Typically, models are named after a person, a place, or a fantasy name. 

        **Decision Process:**
        1. **Model Indicators:**
            - Names are unique and often reflect people, places, or unique identifiers.
            - Multiple distinct names at the same level indicate models.
        2. **Type Indicators:**
            - Descriptive of product categories or styles.
            - Common nouns representing a kind of product.
        3. **Product Line Indicators:**
            - Broad categories encompassing multiple models or types.
            - Often represent a series or collection.
        4. **Specifications Indicators:**
            - Specifications are additional information that describe a product (e.g. color, size, number of doors, etc.).
            - Specifications are in French and need to be translated to English.
            - French specification examples are: foyers, coleur, finition.

        **Decision Process:**
        - **If the entities at the current level include any names from the Models list, identify those entities as models.**
        - **Otherwise, they are not models.**

        **Previous Feedback for '{sous_famille}':**
        {specific_feedback}

        **Previously Rejected Optimization Levels for '{sous_famille}':**
        {', '.join(rejected_for_sf) if rejected_for_sf else 'None'}

        **Entities at Current Level:**
        {processed_entity_names}

        **Question:**
        Which of these entity names correspond to specific product models from the Models list? Please list them.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in product portfolio management analyzing a product hierarchy."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=150,
                temperature=0
            )
            gpt_reply = response.choices[0].message.content.strip()
            logged_text += f"GPT response: {gpt_reply}\n"

            # Parse GPT response to extract identified models
            identified_models = []
            for line in gpt_reply.split('\n'):
                line = line.strip().lstrip('-').strip()
                if line:
                    identified_models.append(line.lower())

            logged_text += f"Identified Models: {identified_models}\n"

            for model in identified_models:
                # Check if the model exists in the processed_entity_names
                exact_model = next((name for name in processed_entity_names if model.lower() == name.lower()), None)
                if exact_model:
                    # Retrieve the original entity name (case-sensitive)
                    original_model = next((name for name in entity_names if name.lower().strip().split('-')[-1] == exact_model), None)
                    if original_model:
                        model_material_id = child_entities[child_entities['name'] == original_model]['material_id'].iloc[0]
                        results.append({
                            'Sous-Famille': sous_famille,
                            'model_name': original_model,  # Use the exact model name with original casing
                            'model_level': next_level,
                            'model_level_name': level_name
                        })
                        logged_text += f"Model '{original_model}' detected at level {next_level} ({level_name}).\n"
                        # Exclude all material_ids under this model in lower levels
                        descendants = get_descendants(hierarchy_df, model_material_id)
                        excluded_material_ids.update(descendants)

        except Exception as e:
            gpt_reply = "No response from GPT"
            logged_text += f"Error during GPT API call: {e}\n"
            break

        # Move to the next level
        level_id = next_level
        current_material_ids = child_entities['material_id'].tolist()

    # Log the total number of identified models
    logged_text += f"Total models identified: {len(results)}\n"

    return pd.DataFrame(results), logged_text

def get_descendants(hierarchy_df: pd.DataFrame, parent_material_id: int) -> set:
    """
    Recursively finds all descendant material_ids for a given parent_material_id.

    Args:
        hierarchy_df (pd.DataFrame): The hierarchy dataframe.
        parent_material_id (int): The parent material_id.

    Returns:
        set: A set of descendant material_ids.
    """
    descendants = set()
    direct_children = hierarchy_df[hierarchy_df['parent_id'] == parent_material_id]['material_id'].tolist()
    for child_id in direct_children:
        descendants.add(child_id)
        descendants.update(get_descendants(hierarchy_df, child_id))
    return descendants