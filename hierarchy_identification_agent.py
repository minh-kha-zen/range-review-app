import os
import pandas as pd
from openai import OpenAI

def identify_optimization_levels(
    hierarchy_df: pd.DataFrame, 
    sous_famille: str, 
    api_key: str, 
    example_models: list
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

    # Get specific feedback for the current sous_famille
    if os.path.exists(feedback_file):
        your_feedback = feedback_df[feedback_df['Sous-Famille'] == sous_famille]['Your Feedback'].dropna().tolist()
        specific_feedback = "\n".join(your_feedback) if your_feedback else ""
    else:
        specific_feedback = ""

    # Retrieve rejected levels for the current Sous-Famille
    rejected_for_sf = rejected_levels.get(sous_famille, set())

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
    
    while True:
        next_level = level_id + 1
        child_entities = hierarchy_df[
            (hierarchy_df['material_id'].isin(current_material_ids)) &
            (hierarchy_df['level'] == next_level)
        ]

        if child_entities.empty:
            logged_text += f"No further child entities found at level {next_level}. Defaulting to SKU level.\n"
            optimization_level_name = 'SKU Level'
            optimization_level_id = 'SKU'
            break

        # Process entity names to keep only the part after the hyphen and convert to lowercase
        entity_names = child_entities['name'].unique().tolist()
        strings_to_remove = [
            's/m', 
            'std', 
            'obso', 
            'expo',
            '(acc+pcd)',
            'h77',
            '(su)',
        ]
        
        # Process entity names to remove unwanted substrings
        processed_entity_names = []
        for name in entity_names:
            if '-' in name:
                # Split the name and take the part after the hyphen
                name = name.split('-')[1].lower()
            else:
                name = name.lower()  # Ensure the name is in lowercase
            
            # Remove unwanted substrings
            for rem in strings_to_remove:
                name = name.replace(rem, '')
            
            processed_entity_names.append(name.strip())  # Strip any leading/trailing whitespace

        processed_entity_names = list(set(processed_entity_names))  # Remove duplicates
        level_name = child_entities.iloc[0]['level_name']

        # Skip rejected levels
        if level_name in rejected_for_sf:
            logged_text += f"Level '{level_name}' has been previously rejected for '{sous_famille}'. Skipping this level.\n"
            level_id = next_level
            current_material_ids = child_entities['material_id'].tolist()
            continue  # Continue to the next level

        logged_text += f"Entities at level {next_level} ({level_name}): {processed_entity_names}\n"

        # Convert example models list to a string for the prompt
        example_models_str = ', '.join(example_models)

        prompt = f"""
        You are an expert in product portfolio management analyzing a product hierarchy.
        Models are specific product variants (e.g., amandie, cancale, molene).
        Typically, models are named after a person, a place, or a fantasy name.
        Types refer to general categories or styles of products (e.g., mononbloc, panneau).
        Product lines are broader categories or series (e.g., etoile).

        **Identification Guidelines:**
        1. **Model Indicators:**
            - Model names are unique and often reflect people, places, or unique identifiers.
            - Multiple distinct model names at the same level indicate models.
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

        Examples:
        - Models: {example_models_str}.
        - Types: 'mononbloc', 'panneau'.
        - Product Lines: 'etoile'.

        **Important:** When you find 'Types' or 'Product Lines' at the current level, move to the next level.

        **Decision Process:**
        - **If the entities at the current level include multiple distinct names from the Models list, identify this level as containing models.**
        - **If the entities primarily consist of types or product lines, move to the next hierarchy level.**
        - **If the entities specification a function of a product (e.g. color, size, number of doors, etc.), move to the next hierarchy level. Be aware that the specifications are in French.**

        **Previous Feedback for '{sous_famille}':**
            {specific_feedback}

        **Previously Rejected Optimization Levels for '{sous_famille}':**
            {', '.join(rejected_for_sf) if rejected_for_sf else 'None'}

        **Entities at Current Level:**
        {processed_entity_names}

        **Question:**
        Do these entity names correspond to specific product models? 
        - If **yes**, reply with exactly: `Yes, these are models.` 
        - If **no**, reply with exactly: `No, these are not models.`.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
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
                max_tokens=100,
                temperature=0
            )
            gpt_reply = response.choices[0].message.content.strip()
            logged_text += f"GPT response: {gpt_reply}\n"
        except Exception as e:
            gpt_reply = "No response from GPT"
            logged_text += f"Error during GPT API call: {e}\n"
            break

        if "Yes, these are models." in gpt_reply:
            logged_text += f"Model names detected at level {next_level}. Stopping search.\n"
            optimization_level_name = level_name
            optimization_level_id = next_level
            break
        else:
            logged_text += f"No models detected. Moving to next level.\n"
            level_id = next_level
            current_material_ids = child_entities['material_id'].tolist()

    results.append({
        'Sous-Famille': sous_famille,
        'Optimization Level Name': optimization_level_name,
        'Optimization Level ID': optimization_level_id
    })
    logged_text += f"Optimization level for '{sous_famille}': {optimization_level_name} (ID: {optimization_level_id})\n"

    return pd.DataFrame(results), logged_text