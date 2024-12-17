from openai import OpenAI

import pandas as pd
import re

def evaluate_sub_families(
    insights_df: pd.DataFrame, 
    api_key: str, 
    no_of_categories_to_review: int, 
    model: str
) -> pd.DataFrame:
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
        if "o1" not in model:
            response = client.chat.completions.create(
                model=model,
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
                max_tokens=1500, 
                temperature=0
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

        print("============ RESPONSE =============\n")
        print(response)
        print("\n==================================\n")

        gpt_reply = response.choices[0].message.content.strip()
        results = parse_gpt_response(gpt_reply)

    except Exception as e:
        results = [{
            'Sub-Family': 'Error',
            'Assess for Evaluation': 'Error',
            'Explanation': f"An error occurred: {e}"
        }]

        print("============ ERROR =============\n")
        print(e)
        print("\n================================\n")

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

    Assign a "Yes" to the "Assessment" only for sub-families that are performing poorly and should be prioritized for evaluation. Sub-families with strong performance metrics should receive a "No".

    Please compile a brief explanation for each sub-family that you have assessed as "Yes", highlighting why you have made this assessment.

    Provide your assessment as follows:
    - Sub-Family: [Sub-Family Name]
    - Assessment (Yes/No): [Your assessment here]
    - Explanation: [Your reasoning here, including comparisons to other sub-families]
    """

    print("============ PROMPT =============\n")
    print(prompt)
    print("\n================================\n")

    return prompt

def parse_gpt_response(response: str) -> list:
    """
    Parses the GPT response to extract the assessment and explanation.
    Handles responses from o1-mini, o1-preview, and gpt-4o models.

    Args:
        response (str): The raw response from GPT.

    Returns:
        list: A list of dictionaries containing the sub-family, assessment, and explanation.
    """

    results = []

    # Remove markdown formatting
    response = response.replace('**', '').replace('---', '').strip()

    # Split the response into sections based on markers
    # Using regex to split on patterns that separate entries
    sections = re.split(r'\n\s*\n+', response)  # Split on multiple newlines

    for section in sections:
        section = section.strip()
        if not section:
            continue

        sub_family_name = ''
        assessment = ''
        explanation = ''

        # Use regex to find Sub-Family, Assessment, and Explanation
        # The patterns may vary, so we check multiple possibilities

        # Sub-Family
        sub_family_match = re.search(r'(?:Sub-Family[:]?|Sub-Family)\s*(?:\s*[:\-]\s*)?\s*(.*)', section)
        if sub_family_match:
            sub_family_line = sub_family_match.group(1).strip()
            # Remove any extra prefixes or suffixes
            sub_family_name = sub_family_line.strip('-').strip()

        # Assessment
        assessment_match = re.search(r'Assessment\s*\(Yes/No\)\s*[:\-]?\s*(\w+)', section)
        if not assessment_match:
            assessment_match = re.search(r'Assessment\s*[:\-]?\s*(\w+)', section)
        if assessment_match:
            assessment = assessment_match.group(1).strip()

        # Explanation
        explanation_match = re.search(r'Explanation\s*[:\-]\s*(.*)', section, re.DOTALL)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
        else:
            # If 'Explanation' keyword is missing, try to capture content after 'Assessment'
            assessment_end = section.find('Assessment')
            if assessment_end != -1:
                # Find the end of the 'Assessment' line
                assessment_line_end = section.find('\n', assessment_end)
                if assessment_line_end != -1:
                    explanation = section[assessment_line_end+1:].strip()

        # Only add results if the sub-family name is not empty
        if sub_family_name and assessment:
            result = {
                'Sub-Family': sub_family_name,
                'Assess for Evaluation': assessment,
                'Explanation': explanation
            }
            results.append(result)

    return results