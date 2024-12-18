import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from category_evaluation_agent_dspy import evaluate_sub_families_dspy
from hierarchy_identification_agent import identify_optimization_levels
from model_extraction_agent import extract_models_for_sub_family 
from category_evaluation_agent import evaluate_sub_families 
from utils import prepare_table_data, prepare_data_for_sub_family, create_or_update_list_item, format_insights_table

st.set_page_config(layout="wide")

FEEDBACK_FILE_HIERARCHY = "feedback_hierarchy.csv"
FEEDBACK_FILE_MODEL = "feedback_model.csv"
api_key = os.getenv("OPENAI_API_KEY")

# Helper function to save feedback
def save_feedback(feedback_data, feedback_file):
    """
    Save feedback to a CSV file.

    Args:
        feedback_data (dict): A dictionary containing feedback information.
        feedback_file (str): Path to the feedback CSV file.
    """
    # Format logged text to replace line breaks with a space or a placeholder
    feedback_data['Logged Text'] = feedback_data['Logged Text'].replace('\n', ' ')  # Replace line breaks with a space

    if not os.path.exists(feedback_file):
        pd.DataFrame([feedback_data]).to_csv(feedback_file, index=False)
    else:
        feedback_df = pd.read_csv(feedback_file)
        feedback_df = pd.concat([feedback_df, pd.DataFrame([feedback_data])], ignore_index=True)
        feedback_df.to_csv(feedback_file, index=False)

# Initialize session state variables
if 'optimization_df' not in st.session_state:
    st.session_state.optimization_df = None
if 'logged_text' not in st.session_state:
    st.session_state.logged_text = ""
if 'feedback' not in st.session_state:
    st.session_state.feedback = ""
if 'model_df' not in st.session_state:
    st.session_state.model_df = None
if 'model_logged_text' not in st.session_state:
    st.session_state.model_logged_text = ""

st.title("Range Review Demo")
st.write("Demo workflow for LLM-based range review")

load_dotenv()  # Load environment variables from .env file
BASE_PATH = os.path.join(os.getenv("BASE_PATH"), "03_Dashboard_data", "dev")

# Read parquet files
master = pd.read_parquet(os.path.join(BASE_PATH, "master.parquet"))
hierarchy = pd.read_parquet(os.path.join(BASE_PATH, "hierarchy.parquet"))
supplier = pd.read_parquet(os.path.join(BASE_PATH, "suppliers.parquet"))
sales = pd.read_parquet(os.path.join(BASE_PATH, "sales", "sales_by_month.parquet"))
inventory = pd.read_parquet(os.path.join(BASE_PATH, "inventory.parquet"))
attributes = pd.read_parquet(os.path.join(BASE_PATH, "attributes.parquet"))

st.header("1. Data Preview")
st.write("Master Data")
st.write(master.head())
st.write("Sales Data")
st.write(sales.head())
st.write("Hierarchy Data")
st.write(hierarchy.head(10))

st.markdown("---")
st.header("2. Data Preparation")
# Create columns for filters
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    # Date filter for start date
    start_date = st.date_input("Select start date", value=pd.to_datetime("2024-01-01"))

with col2:
    # Date filter for end date
    end_date = st.date_input("Select end date", value=pd.to_datetime("2024-11-30"))

with col3:
    # Dropdown for bundle column
    bundle_option = st.selectbox("Select Bundle", options=["APD", "AVD"], index=0)

with col4:
    # Dropdown for product_status in master data
    product_status = st.multiselect("Select Product Status", options=["0", "1", "2", "3", "4"], default=["0"])

with col5:
    # Dropdown for Type de complément in attributes data
    unique_type_de_complement = attributes[attributes['key'] == 'Type de complément']['value'].unique().tolist()
    type_de_complement = st.multiselect("Select Type de complément", options=unique_type_de_complement, default=["Principal"])

# Convert start_date and end_date to datetime
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filter sales data based on the selected date range and bundle
filtered_sales = sales[
    (sales['date'] >= start_date) & 
    (sales['date'] <= end_date) & 
    (sales['bundle'] == bundle_option)
]

# Filter master data based on product status
filtered_master = master[master['product_status'].astype(str).isin(product_status)]

# Filter master data based on type_de_complement
print("number of principal ids", len(attributes[attributes['value'].isin(type_de_complement)]['material_id'].unique().tolist()))
type_de_complement_material_ids = attributes[
    (attributes['value'].isin(type_de_complement)) & 
    (attributes['key'] == 'Type de complément')
]['material_id'].unique().tolist()
filtered_master = filtered_master[filtered_master['material_id'].isin(type_de_complement_material_ids)]

# Inner join master and sales data on material_id
merged_data = pd.merge(filtered_master, filtered_sales, on='material_id', how='inner')

# Group by material_id and calculate sum of quantity, net_revenue, margin, and gross_revenue
grouped_sales = merged_data.groupby('material_id').agg(
    quantity=('quantity', 'sum'),
    net_revenue=('net_revenue', 'sum'),
    margin=('margin', 'sum'),
    gross_revenue=('gross_revenue', 'sum')  # Include gross_revenue in the aggregation
).reset_index()

# Calculate total number of SKUs
total_skus = grouped_sales['material_id'].nunique()

# Format the values
total_quantity = f"{grouped_sales['quantity'].sum() / 1_000_000:.1f}M"
total_net_revenue = f"€{grouped_sales['net_revenue'].sum() / 1_000_000:.1f}M"
total_margin = f"€{grouped_sales['margin'].sum() / 1_000_000:.1f}M"
relative_margin = f"{(grouped_sales['margin'].sum() / grouped_sales['net_revenue'].sum() * 100):.1f}%" if grouped_sales['net_revenue'].sum() != 0 else "N/A"
total_skus_display = f"{total_skus / 1_000:.1f}K"

# Create columns for KPIs
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(label="Total Quantity", value=total_quantity)

with col2:
    st.metric(label="Total Net Revenue", value=total_net_revenue)

with col3:
    st.metric(label="Total Margin", value=total_margin)

with col4:
    st.metric(label="Relative Margin", value=relative_margin)

with col5:
    st.metric(label="Total SKUs", value=total_skus_display)

st.markdown("---")  # This adds a horizontal divider line

st.header("3. Data Analysis")

# First, join all hierarchy levels to grouped_sales
hierarchy_pivoted = hierarchy.pivot(
    index='material_id',
    columns='level_name',
    values='name'
).reset_index()

# Join the pivoted hierarchy with grouped_sales
analysis_data = pd.merge(
    grouped_sales,
    hierarchy_pivoted,
    on='material_id',
    how='left'
)

# Create columns for the 7 filters
filter_cols = st.columns(7)

# Dictionary to store filter selections
selected_filters = {}

# Create a filter for each level in the same row
for idx in range(1, 8):  # Levels 1 to 7
    level_name = hierarchy[hierarchy['level'] == idx]['level_name'].values[0]  # Get level_name for the current level
    with filter_cols[idx - 1]:  # Adjust index for 0-based
        options = ['All'] + sorted(analysis_data[level_name].astype(str).unique().tolist())
        selected_filters[level_name] = st.selectbox(
            f"Filter {level_name}",
            options=options,
            key=f"filter_{level_name}"
        )

# Apply filters to the data
filtered_data = analysis_data.copy()
for level_name, selected_value in selected_filters.items():
    if selected_value != 'All':
        filtered_data = filtered_data[filtered_data[level_name] == selected_value]

# Create a mapping from level_name to level_id
level_mapping = hierarchy.set_index('level_name')['level'].to_dict()

# Dropdown for selecting which level to show in the bar chart
level_options = hierarchy['level_name'].unique().tolist()
selected_level_name = st.selectbox("Select Level for Chart", options=level_options, key="selected_level_chart")

# Map selected_level_name to its corresponding level_id
selected_level_id = level_mapping.get(selected_level_name)

# Group by selected level and calculate total margin
margin_by_level = filtered_data.groupby(selected_level_name).agg(
    total_margin=('margin', 'sum')
).reset_index()

# Sort by total margin in descending order
margin_by_level = margin_by_level.sort_values(by='total_margin', ascending=False)

# Plotting the bar chart
st.header("Total Margin by Hierarchy Level")
st.bar_chart(margin_by_level.set_index(selected_level_name)['total_margin'], horizontal=True, height=400)

# Prepare previous year sales data if needed (for YoY calculations)
previous_year_sales = sales[
    (sales['date'] >= start_date - pd.DateOffset(years=1)) & 
    (sales['date'] <= end_date - pd.DateOffset(years=1)) & 
    (sales['bundle'] == bundle_option)
]

# Get list of material ids from filtered data
material_ids = filtered_data['material_id'].unique().tolist()

# Create table data using the utility function
table_data = prepare_table_data(
    material_ids=material_ids,
    hierarchy=hierarchy,
    sales=filtered_sales,
    master=filtered_master,
    optimization_level_id=selected_level_id,
    previous_year_sales=previous_year_sales
)

# Display the table
st.header("Detailed Data Table")
st.dataframe(format_insights_table(table_data))

st.markdown("---")  # Divider

# ------------------ Chapter 4: Model Hierarchy Identification Agent ------------------

st.header("4. Model Hierarchy Identification Agent")
st.write("Select a Sous-Famille to identify the optimization level.")
sous_famille_list = sorted(hierarchy[hierarchy['level_name'] == 'Sous-Famille']['name'].unique().tolist())
selected_sous_famille_ch4 = st.selectbox("Select a Sous-Famille", options=sous_famille_list, key="selected_sous_famille_ch4")

example_models = attributes[attributes['key'] == 'Modèle']['value'].unique().tolist()


if 'feedback' not in st.session_state:
    st.session_state.feedback = ""

if 'validation_status' not in st.session_state:
    st.session_state.validation_status = ""

if st.button("Identify Optimization Levels"):
    # Initialize sous_famille to None
    sous_famille = None
    
    # Check if user provided input
    if not selected_sous_famille_ch4.strip():
        st.error("Please select a Sous-Famille.")
    else:
        # Assign the input to sous_famille
        sous_famille = selected_sous_famille_ch4.strip()
        
        if not api_key:
            st.error("OPENAI_API_KEY not found in environment variables.")
        else:
            with st.spinner("Identifying optimization levels..."):
                try:
                    # Pass the single Sous-Famille to the function
                    optimization_df, logged_text = identify_optimization_levels(hierarchy, sous_famille, api_key, example_models)
                    
                    # Update session state
                    st.session_state.optimization_df = optimization_df
                    st.session_state.logged_text = logged_text
                    st.session_state.feedback = ""
                    st.session_state.validation_status = ""
                    
                    # Display success and output
                    st.success("Optimization levels identified successfully.")
                    st.dataframe(optimization_df)

                except Exception as e:
                    st.error(f"An error occurred: {e}")

# Display Approve and Reject buttons only if optimization_df is available
if st.session_state.optimization_df is not None:
    optimization_df = st.session_state.optimization_df
    logged_text = st.session_state.logged_text

    # Assume there's only one Sous-Famille
    row = optimization_df.iloc[0]

    # Display the logged text in grey with a border
    if st.session_state.logged_text:  # Only display if logged_text exists in session state
        st.markdown("**Logged Text:**")
        st.markdown(
            f"""<div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; border: 1px solid #e0e0e0; color: black;'>
            {st.session_state.logged_text.replace('\n', '<br>')}
            </div>""", 
            unsafe_allow_html=True
        )

    # Feedback input
    st.markdown("---")
    st.header("Provide Your Feedback")
    feedback = st.text_area("Your Feedback", value=st.session_state.feedback, height=150, key="feedback_input")

    # Approval Buttons in separate rows with different colors
    approve_clicked = st.button(
        "✓ Approve",
        key="approve_button",
        type="primary",  # This will give it a colorful appearance
    )

    reject_clicked = st.button(
        "✗ Reject",
        key="reject_button",
        type="secondary",  # This gives it a more neutral appearance
    )

    if approve_clicked or reject_clicked:
        decision = "Approved" if approve_clicked else "Rejected"
        if feedback.strip() == "":
            st.error("Please provide your feedback before submitting.")
        else:
            feedback_data = {
                "Sous-Famille": row["Sous-Famille"],
                "Optimization Level Name": row["Optimization Level Name"],
                "Optimization Level ID": row["Optimization Level ID"],
                "Feedback": decision,
                "Logged Text": logged_text,
                "Your Feedback": feedback.strip()
            }
            save_feedback(feedback_data, FEEDBACK_FILE_HIERARCHY)
            st.session_state.validation_status = "success"
            st.session_state.feedback = ""
            st.session_state.optimization_df = None
            st.session_state.logged_text = ""
            st.success(f"Feedback for '{row['Sous-Famille']}' has been saved successfully.")

# ------------------ Chapter 5: Model Extraction Agent ------------------

st.markdown("---")  # Divider

st.header("5. Model Extraction Agent")
st.write("Select a Sous-Famille to extract all models.")
sous_famille_list_model = sorted(hierarchy[hierarchy['level_name'] == 'Sous-Famille']['name'].unique().tolist())
selected_sous_famille_ch5 = st.selectbox("Select a Sous-Famille", options=sous_famille_list_model, key="selected_sous_famille_ch5")

extract_button = st.button("Extract Models")

if extract_button:
    # Initialize sous_famille_model to None
    sous_famille_model = None
    
    # Check if user provided input
    if selected_sous_famille_ch5.strip() == "":
        st.error("Please select a Sous-Famille.")
    else:
        # Assign the input to sous_famille_model
        sous_famille_model = selected_sous_famille_ch5.strip()
        
        if not api_key:
            st.error("OPENAI_API_KEY not found in environment variables.")
        else:
            with st.spinner("Extracting models..."):
                try:
                    # Pass the single Sous-Famille to the extraction function
                    model_df, model_logged_text = extract_models_for_sub_family(hierarchy, sous_famille_model, api_key, example_models)
                    
                    # Update session state
                    st.session_state.model_df = model_df
                    st.session_state.model_logged_text = model_logged_text
                    
                    # Display success and output
                    st.success("Models extracted successfully.")
                    st.dataframe(model_df)

                except Exception as e:
                    st.error(f"An error occurred: {e}")

# Display Logged Text and Feedback for Model Extraction only if model_df is available
if st.session_state.model_df is not None and not st.session_state.model_df.empty:
    model_df = st.session_state.model_df
    model_logged_text = st.session_state.model_logged_text

    # Assume there's only one Sous-Famille
    model_row = model_df.iloc[0]

    # Display the logged text in grey with a border
    if st.session_state.model_logged_text:  # Only display if logged_text exists in session state
        st.markdown("**Logged Text:**")
        st.markdown(
            f"""<div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; border: 1px solid #e0e0e0; color: black;'>
            {st.session_state.model_logged_text.replace('\n', '<br>')}
            </div>""", 
            unsafe_allow_html=True
        )

    # Feedback input
    st.markdown("---")
    st.header("Provide Your Feedback for Model Extraction")
    model_feedback = st.text_area("Your Feedback", value=st.session_state.feedback, height=150, key="model_feedback_input")

    # Approval Buttons in separate rows with different colors
    model_approve_clicked = st.button(
        "✓ Approve",
        key="model_approve_button",
        type="primary",  # This will give it a colorful appearance
    )

    model_reject_clicked = st.button(
        "✗ Reject",
        key="model_reject_button",
        type="secondary",  # This gives it a more neutral appearance
    )

    if model_approve_clicked or model_reject_clicked:
        model_decision = "Approved" if model_approve_clicked else "Rejected"
        if model_feedback.strip() == "":
            st.error("Please provide your feedback before submitting.")
        else:
            feedback_data = {
                "Sous-Famille": model_row["Sous-Famille"],
                "model_name": model_row["model_name"],
                "model_level": model_row["model_level"],
                "model_level_name": model_row["model_level_name"],
                "Feedback": model_decision,
                "Logged Text": model_logged_text,
                "Your Feedback": model_feedback.strip()
            }
            save_feedback(feedback_data, FEEDBACK_FILE_MODEL)
            st.session_state.validation_status = "success"
            st.session_state.feedback = ""
            st.session_state.model_df = None
            st.session_state.model_logged_text = ""
            st.success(f"Feedback for model '{model_row['model_name']}' in '{model_row['Sous-Famille']}' has been saved successfully.")
else:
    # If no models were extracted, display the logged text
    st.warning("No models were extracted. Please check the input or the hierarchy data.")
    if st.session_state.model_logged_text:  # Display logged text if it exists
        st.markdown("**Logged Text:**")
        st.markdown(
            f"""<div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; border: 1px solid #e0e0e0; color: black;'>
            {st.session_state.model_logged_text.replace('\n', '<br>')}
            </div>""", 
            unsafe_allow_html=True
        )

# ------------------ Chapter 6: Run Hierarchy Identification Agent ------------------

st.markdown("---")  # Divider

st.header("6. Collect Insight Metrics for One Sub-Family")

# Dropdown to select a sous-famille
sous_famille_list = sorted(hierarchy[hierarchy['level_name'] == 'Sous-Famille']['name'].unique().tolist())
selected_sous_famille_ch6 = st.selectbox("Select a Sous-Famille", options=sous_famille_list, key="selected_sous_famille_ch6")

# Button to insight data
collect_button = st.button("Collect Insight Data", key="collect_data_button")

if collect_button:
    if not api_key:
        st.error("OPENAI_API_KEY not found in environment variables.")
    else:
        with st.spinner("Running hierarchy identification and collecting model data..."):
            try:
                # Call the new function to prepare data for the selected sous-famille
                table_data_ch6 = prepare_data_for_sub_family(
                    selected_sous_famille_ch6,
                    hierarchy,
                    sales,
                    start_date,
                    end_date,
                    bundle_option,
                    master,
                    api_key,
                    example_models
                )
                
                # Display the table
                st.success("Insight Data collected successfully.")
                st.dataframe(format_insights_table(table_data_ch6))
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

st.markdown("---")  # Divider

st.header("7. Identify Sub-Families for Range Review")
st.write("Select a Famille to identify all Sub-Families for Range Review.")

# Dropdown to select a sous-famille
famille_list = sorted(hierarchy[hierarchy['level_name'] == 'Famille']['name'].unique().tolist())
selected_famille_ch7 = st.selectbox("Select a Famille", options=famille_list, key="selected_famille_ch7")

# Text input for users to enter sub-families to exclude
exclude_input = st.text_input(
    "Enter Sub-Families to Exclude (comma-separated)",
    value="ACCESSOIRES, PIECES DETACHEE, PROMOS, EXPO, LIMITEES, DIVERS, AUTRES, GARANTIE",  # Default value for guidance
    key="exclude_sub_families_input"
)

# Split the input into a list
exclude_sub_families = [sub_family.strip() for sub_family in exclude_input.split(',')]
no_of_categories_to_review = st.number_input("Set the number of sub-families to review:", min_value=1, max_value=100, value=2, step=1)

# Button to insight data
compile_button = st.button("Compile Data", key="compile_data_button")

if compile_button:
    # Run identify_sub_families in selected family
    selected_hierarchy = hierarchy[hierarchy['name'] == selected_famille_ch7]
    material_ids = selected_hierarchy['material_id'].unique()

    sous_famille_hierarchy = hierarchy[
        (hierarchy['level_name'] == 'Sous-Famille') & 
        (hierarchy['material_id'].isin(material_ids))
    ]
    
    # Adjust the filtering logic to check for substrings
    filtered_sous_famille_hierarchy = sous_famille_hierarchy[
        ~sous_famille_hierarchy['name'].str.contains('|'.join(exclude_sub_families), case=False, na=False)
    ]

    unique_sous_famille_names = filtered_sous_famille_hierarchy['name'].unique()

    # Initialize an empty list to store DataFrames
    all_dataframes = []

    # Create a progress bar
    progress_text = "Preparing data for sub-families. Please wait..."
    progress_bar = st.progress(0, text=progress_text)

    # Loop through each unique sous-famille name
    for i, sous_famille in enumerate(unique_sous_famille_names):
        # Call the function to prepare data for each sous-famille
        table_data_ch7 = prepare_data_for_sub_family(
            sous_famille,
            hierarchy,
            sales,
            start_date,
            end_date,
            bundle_option,
            filtered_master,
            api_key,
            example_models,
        )
        
        # Append the resulting DataFrame to the list
        all_dataframes.append(table_data_ch7)

        # Update the progress bar
        progress_percentage = int((i + 1) / len(unique_sous_famille_names) * 100)  # Convert to integer
        progress_bar.progress(progress_percentage, text=progress_text)

    # Concatenate all DataFrames into one
    insights_table = pd.concat(all_dataframes, ignore_index=True)

    # Store the final_table in session state
    st.session_state.insights_table = insights_table

# Check if final_table exists in session state and display it
if 'insights_table' in st.session_state:
    st.write(
        "Combined Data Table for Selected Sous-Familles:", format_insights_table(st.session_state.insights_table)
    )

    model_options = [
        "gpt-4o", 
        "gpt-4o-mini", 
        "gpt-3.5-turbo", 
        "o1-mini",
        "o1-preview",
        "o1",
    ]
    selected_model = st.selectbox("Select GPT Model for Evaluation", options=model_options, index=model_options.index("o1-mini"))

    evaluate_button = st.button("Evaluate Sub-Families")

    if evaluate_button:
        if 'insights_table' in st.session_state:
            # Input for maximum number of evaluations
            insights_df = format_insights_table(st.session_state.insights_table)
            with st.spinner("Evaluating sub-families..."):
                evaluation_results = evaluate_sub_families_dspy(
                    insights_df, 
                    api_key, 
                    no_of_categories_to_review, 
                    selected_model
                )
                st.success("Sub-families evaluated successfully.")
                st.session_state.evaluation_results = evaluation_results
        else:
            st.warning("No data available to evaluate. Please prepare the data first.")

    if 'evaluation_results' in st.session_state:
        st.dataframe(st.session_state.evaluation_results, use_container_width=True)

# ------------------ Chapter 8: Upload Results to Sharepoint List ------------------

st.markdown("---")  # Divider

st.header("8. Upload Results to Sharepoint List")

access_token = st.text_input("Enter Access Token")

start_upload_button = st.button("Start Upload")

if start_upload_button:
    if 'evaluation_results' in st.session_state:
        # Input for maximum number of evaluations
        results_df = st.session_state.evaluation_results

        # Filter results to keep only rows where 'Assess for Evaluation' is 'Yes'
        filtered_results_df = results_df[results_df['assessment'] == 'Yes'].copy()

        # Left join with insights_table from session state
        if 'insights_table' in st.session_state:
            insights_df = st.session_state.insights_table
            filtered_results_df = filtered_results_df.merge(insights_df, on='name', how='left')

        # Drop the 'Assess for Evaluation' column
        columns_to_drop = [
            'assessment', 
            'Entity Level Name',
            'Entity Level ID',
        ]
        filtered_results_df.drop(columns=columns_to_drop, inplace=True)

        # Rename columns as specified
        filtered_results_df.rename(columns={
            'name': 'Title',
        }, inplace=True)


        categories_to_review_list_id = '9148877c-ce53-421e-aebf-e6dcd60266a3'
        project_list_id = '58cc53df-4e47-413a-9088-ef828aaeb7f8'
        site_id = '89d6ee16-e20e-438e-a93b-81f067e2843b'

        # Initialize the progress bar
        progress_bar_ch8 = st.progress(0)

        # Iterate through the results DataFrame
        for index, row in filtered_results_df.iterrows():
            item = row.to_dict()
            id_column = 'Title'
            print("Uploading item: ", item[id_column]) # Line break added
            create_or_update_list_item(site_id, categories_to_review_list_id, item, id_column, access_token)
            print("Uploaded item: ", item)
            print()

            # Update the progress bar
            progress_percentage = (index + 1) / len(filtered_results_df)
            progress_bar_ch8.progress(progress_percentage)

        st.success("Uploaded results to Sharepoint list successfully.")