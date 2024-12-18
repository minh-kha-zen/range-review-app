import pandas as pd
import dspy

def evaluate_sub_families_dspy(
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
        model (str): Model name to use for the OpenAI API.

    Returns:
        pd.DataFrame: A DataFrame with sub-families, assessment (Yes/No), and explanations.
    """
    
    dspy.configure(lm=dspy.LM(
        'openai/' + model, 
        api_key=api_key,
        max_tokens=5000,
        temperature=1.0,
        )
    )

    # For the given sub-family, extract the performance metrics of other sub-families in the same category and return them as a list of dictionaries.
    sub_families = insights_df.to_dict(orient='records')

    metrics_context = """
    - **Total Net Revenue**: Higher values indicate more important sub-families. Low revenue may suggest poor performance.
    - **Total Margin**: Higher values indicate more important sub-families. Low margin suggests a need for improvement.
    - **Relative Margin**: Lower relative margin is more critical, indicating poor performance compared to others.
    - **Avg Discount**: Higher average discount indicates room for pricing strategy improvements.
    - **Latest Introduction**: Indicates the last evaluation time. Older dates suggest urgency for re-evaluation.
    - **Total SKUs**: Higher SKU count indicates more complexity, leading to inefficiencies.
    - **Margin Spread**: Higher margin spread indicates opportunities to shift sales to higher-margin products. Thus, the higher the margin spread, the poorer the performance.
    - **YoY Abs. Margin Change**: Lower year-over-year change of the absolute margin in percent suggests declining performance.
    - **YoY Rel. Margin Change**: Lower year-over-year change of the relative margin in percentage points suggests declining performance.
    - **Number of Models**: Higher model count indicates more complexity, complicating management and performance.
    - **Margin per Model**: Lower margin per model indicates inefficiencies in the product line.
    """

    class AssessCategoryForReview(dspy.Signature):
        task: str = dspy.InputField()
        context: dict = dspy.InputField(desc="The dictionary tells you how to interpret each of the input metrics to assess if a sub-family should be evaluated in the portfolio review process due to low performance.")
        no_of_sub_families_to_evaluate: int = dspy.InputField(desc="The number of sub-families to select for the portfolio review process.")
        sub_families: list[dict] = dspy.InputField(desc="The list of dictionaries contains the performance metrics the sub-families to be assessed.")
        assessments: list[bool] = dspy.OutputField(desc="The list of of assessments (True/False) for each sub-family.")
        reasonings: list[str] = dspy.OutputField(desc="The list of of reasonings behind the assessments for each sub-family. Explicitly mention the metrics values in the reasoning output. Create a list of up to three bullet points. Each bullet point should be a sentence, with a reasoning for each of the most critical metrics.")
        summaries: list[str] = dspy.OutputField(desc="The list of of reasoning summaries behind the assessments for each sub-family. Generate a one sentence summary of the most critical reason for each sub-family.")

    # Define a module (ChainOfThought) and assign it a signature (return an answer, given a question).
    assessor = dspy.ChainOfThought(AssessCategoryForReview)
    response = assessor(
        task="Select sub-families with poor performance, that should be evaluated in the portfolio review process, and provide reasons for your selection. Only select as many sub-families as specified in 'no_of_sub_families_to_evaluate'. Please generate output lists with assessments and reasonings for each sub-family.", 
        context=metrics_context, 
        no_of_sub_families_to_evaluate=no_of_categories_to_review,
        sub_families = sub_families,
    )
    return parse_dspy_response(response, insights_df)

def parse_dspy_response(response, insights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses the dspy response into a DataFrame.

    Args:
        response: The response object from dspy.
        insights_df (pd.DataFrame): The original insights DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with sub-families, assessment (Yes/No), and explanations.
    """
    sub_families = insights_df['Category Name'].tolist()
    assessments = response.assessments
    reasonings = response.reasonings
    summaries = response.summaries
    results = []

    for sub_family, assessment, reasoning, summary in zip(sub_families, assessments, reasonings, summaries):
        results.append({
            'name': sub_family,
            'assessment': 'Yes' if assessment else 'No',
            'reasoning': reasoning,
            'summary': summary
        })

    return pd.DataFrame(results)