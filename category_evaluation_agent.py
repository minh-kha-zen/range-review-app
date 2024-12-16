from openai import OpenAI

import pandas as pd

def evaluate_sub_families(insights_df: pd.DataFrame, api_key: str, no_of_categories_to_review: int) -> pd.DataFrame:
    """
    Evaluates sub-families in the insights_table to determine if they should be included in the portfolio review.

    Args:
        insights_df (pd.DataFrame): The insights table containing sub-families and their metrics.
        api_key (str): OpenAI API key.
        no_of_categories_to_review (int): The number of sub-families to review.

    Returns:
        pd.DataFrame: A DataFrame with sub-families, assessment (Yes/No), and explanations.
    """
    
    client = OpenAI(api_key=api_key)

    # Create a single prompt with all sub-families
    prompt = create_evaluation_prompt(insights_df, no_of_categories_to_review)

    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an expert in product portfolio management."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=1500,  # Adjust max_tokens as needed
        temperature=0)

        gpt_reply = response.choices[0].message.content.strip()
        results = parse_gpt_response(gpt_reply)

    except Exception as e:
        results = [{
            'Sub-Family': 'Error',
            'Assess for Evaluation': 'Error',
            'Explanation': f"An error occurred: {e}"
        }]

    results = [result for result in results if result['Sub-Family'] and result['Assess for Evaluation'] == 'Yes']

    return pd.DataFrame(results)

def create_evaluation_prompt(insights_df: pd.DataFrame, no_of_categories_to_review: int) -> str:
    """
    Creates a prompt for the LLM to evaluate all sub-families based on provided metrics.

    Args:
        insights_df (pd.DataFrame): The insights table containing sub-families and their metrics.
        no_of_categories_to_review (int): The number of sub-families to review.

    Returns:
        str: The formatted prompt.
    """
    metrics_list = []
    for _, row in insights_df.iterrows():
        sub_family = row['Model Name']
        metrics = "\n".join([f"- **{key}**: {row.get(key, 0)}" for key in row.index if key != 'Model Name'])
        metrics_list.append(f"Sub-Family: {sub_family}\n{metrics}")

    metrics_str = "\n\n".join(metrics_list)

    metrics_context = """
    - **Total Net Revenue**: The higher this value, the more important the sub-family is. Low revenue may indicate poor performance compared to others.
    - **Total Margin**: The higher this value, the more important the sub-family is. A low margin suggests a need for improvement.
    - **Relative Margin**: A lower relative margin is more critical, indicating that the sub-family may not be performing well compared to others.
    - **Avg Discount**: A higher average discount indicates more room for improvements in pricing strategy.
    - **Latest Introduction**: This indicates the last time this sub-family was evaluated. The more this date is in the past, the more urgent the sub-family should be re-evaluated.
    - **Total SKUs**: A higher number of SKUs indicates more complexity in the sub-family, which can lead to inefficiencies.
    - **Margin Spread**: A higher margin spread indicates opportunities to shift sales from low-margin products to higher-margin products.
    - **YoY Margin Change**: A lower year-over-year margin change is critical, suggesting declining performance.
    - **Number of Models**: A higher number of models indicates more complexity, which can complicate management and performance.
    - **Margin per Model**: A lower margin per model is worse, indicating inefficiencies in the product line.
    """

    prompt = f"""
    You are an expert in product portfolio management analyzing product sub-families. Your task is to rank the sub-families based on their performance and identify the {no_of_categories_to_review} worst-performing sub-families that should be evaluated in the portfolio review process. 

    Consider the following metrics for each sub-family:

    {metrics_str}

    Here is the context for each metric:

    {metrics_context}

    Provide your assessment as follows:
    - Sub-Family: [Sub-Family Name]
    - Assessment (Yes/No): [Your assessment here]
    - Explanation: [Your reasoning here, including comparisons to other sub-families]
    """

    print(prompt)

    return prompt

def parse_gpt_response(response: str) -> list:
    """
    Parses the GPT response to extract the assessment and explanation.

    Args:
        response (str): The raw response from GPT.

    Returns:
        list: A list of dictionaries containing the sub-family, assessment, and explanation.
    """
    results = []
    sub_families = response.split('\n\n')
    for sub_family in sub_families:
        lines = sub_family.split('\n')
        sub_family_name = ''
        assessment = 'No'
        explanation = ''

        for line in lines:
            if 'Sub-Family' in line:
                sub_family_name = line.split(':')[-1].strip()
            elif 'Assessment' in line:
                assessment = line.split(':')[-1].strip()
            elif 'Explanation' in line:
                explanation = line.split(':')[-1].strip()

        results.append({
            'Sub-Family': sub_family_name,
            'Assess for Evaluation': assessment,
            'Explanation': explanation
        })

    return results