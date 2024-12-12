import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from hierarchy_identification_agent import identify_optimization_levels_with_ai  # Importing the agent

st.set_page_config(layout="wide")

FEEDBACK_FILE = "feedback.csv"

# Helper function to save feedback
def save_feedback(feedback_data):
    """
    Save feedback to a CSV file.

    Args:
        feedback_data (dict): A dictionary containing feedback information.
    """
    # Format logged text to replace line breaks with a space or a placeholder
    feedback_data['Logged Text'] = feedback_data['Logged Text'].replace('\n', ' ')  # Replace line breaks with a space

    if not os.path.exists(FEEDBACK_FILE):
        pd.DataFrame([feedback_data]).to_csv(FEEDBACK_FILE, index=False)
    else:
        feedback_df = pd.read_csv(FEEDBACK_FILE)
        feedback_df = pd.concat([feedback_df, pd.DataFrame([feedback_data])], ignore_index=True)
        feedback_df.to_csv(FEEDBACK_FILE, index=False)

# Initialize session state variables
if 'optimization_df' not in st.session_state:
    st.session_state.optimization_df = None
if 'logged_text' not in st.session_state:
    st.session_state.logged_text = ""
if 'feedback' not in st.session_state:
    st.session_state.feedback = ""

st.title("Range Review Demo")
st.write("Demo workflow for LLM-based range review")

load_dotenv()  # Load environment variables from .env file
BASE_PATH = os.path.join(os.getenv("BASE_PATH"), "03_Dashboard_data", "dev")

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

st.markdown("---")  # This adds a horizontal divider line

st.header("2. Data Preparation")

# Create columns for filters
col1, col2, col3, col4 = st.columns(4)

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
    product_status = st.multiselect("Select Product Status", options=["0", "1", "2", "3", "4"], default=["0", "1", "2", "3", "4"])

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
total_skus = f"{total_skus / 1_000:.1f}K"

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
    st.metric(label="Total SKUs", value=total_skus)

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

# Dropdown for selecting which level to show in the bar chart
level_options = hierarchy['level_name'].unique().tolist()
selected_level_name = st.selectbox("Select Level for Chart", options=level_options)

# Group by selected level and calculate total margin
margin_by_level = filtered_data.groupby(selected_level_name).agg(
    total_margin=('margin', 'sum')
).reset_index()

# Sort by total margin in descending order
margin_by_level = margin_by_level.sort_values(by='total_margin', ascending=False)

# Plotting the bar chart
st.header("Total Margin by Hierarchy Level")
st.bar_chart(margin_by_level.set_index(selected_level_name)['total_margin'], horizontal=True, height=400)

# Join the master data to get the date_of_introduction
filtered_data = pd.merge(filtered_data, master[['material_id', 'date_of_introduction']], on='material_id', how='left')

# Create a table with one row per bar in the bar chart, including gross_revenue
table_data = filtered_data.groupby(selected_level_name).agg(
    margin=('margin', 'sum'),
    net_revenue=('net_revenue', 'sum'),
    number_of_skus=('material_id', 'nunique'),  # Count unique material_id for SKU count
    latest_intro=('date_of_introduction', 'max'),  # Get the latest introduction date
    gross_revenue=('gross_revenue', 'sum')  # Include gross_revenue in the aggregation
).reset_index()

# Calculate relative margin as margin/net_revenue
table_data['relative_margin'] = table_data.apply(
    lambda row: (row['margin'] / row['net_revenue'] * 100) if row['net_revenue'] != 0 else 0,
    axis=1
)

# Calculate avg_discount
table_data['avg_discount'] = ((table_data['gross_revenue'] - table_data['net_revenue']) / table_data['gross_revenue'] * 100).fillna(0)

# Format avg_discount as a percentage
table_data['avg_discount'] = table_data['avg_discount'].apply(lambda x: f"{x:.2f}%" if x != 0 else "0.00%")

# Round the margin to a full integer
table_data['margin'] = table_data['margin'].round().astype(int)

# Format the relative margin as a percentage
table_data['relative_margin'] = table_data['relative_margin'].apply(lambda x: f"{x:.2f}%" if x != 0 else "0.00%")

# Sort the table by descending margin
table_data = table_data.sort_values(by='margin', ascending=False)

# Calculate margin share
total_margin = table_data['margin'].sum()
table_data['margin_share'] = table_data['margin'] / total_margin * 100

# Calculate cumulative margin share using cumulative sum
table_data['cum_margin_share'] = table_data['margin_share'].cumsum()

# Determine Pareto flag
table_data['pareto'] = table_data['cum_margin_share'].apply(lambda x: "Top 80%" if x < 80 else "Bottom 20%")

# Calculate margin spread for the selected level
def calculate_margin_spread(current_value):
    # Get current level number from hierarchy
    current_level = hierarchy[hierarchy['level_name'] == selected_level_name]['level'].iloc[0]
    
    # Get next level name from hierarchy
    next_level = current_level + 1
    next_level_name = hierarchy[hierarchy['level'] == next_level]['level_name'].iloc[0]
    
    # Filter data for the current selection
    next_level_data = filtered_data[filtered_data[selected_level_name] == current_value]
    
    if not next_level_data.empty:
        # Group by next level and calculate aggregates
        next_level_margins = next_level_data.groupby(next_level_name).agg(
            total_margin=('margin', 'sum'),
            total_revenue=('net_revenue', 'sum')
        ).reset_index()
        
        # Calculate relative margin for each group
        next_level_margins['relative_margin'] = next_level_margins.apply(
            lambda row: (row['total_margin'] / row['total_revenue'] * 100) if row['total_revenue'] != 0 else 0,
            axis=1
        )
        
        # Calculate spread
        max_margin = next_level_margins['relative_margin'].max()
        min_margin = next_level_margins['relative_margin'].min()
        return max_margin - min_margin
    return 0

# Add margin_spread column
table_data['margin_spread'] = table_data[selected_level_name].apply(calculate_margin_spread)

# Create filtered sales data for previous year (2023)
filtered_sales_prev_year = sales[
    (sales['date'] >= pd.to_datetime("2023-01-01")) & 
    (sales['date'] <= pd.to_datetime("2023-11-30")) & 
    (sales['bundle'] == bundle_option)
]

# Create merged data for previous year (same filters as current year)
merged_data_prev_year = pd.merge(filtered_master, filtered_sales_prev_year, on='material_id', how='inner')

# Group previous year data by material_id (same as current year grouping)
grouped_sales_prev_year = merged_data_prev_year.groupby('material_id').agg(
    quantity=('quantity', 'sum'),
    net_revenue=('net_revenue', 'sum'),
    margin=('margin', 'sum')
).reset_index()

# Create analysis data for previous year (same process as current year)
analysis_data_prev_year = pd.merge(
    grouped_sales_prev_year,
    hierarchy_pivoted,
    on='material_id',
    how='left'
)

# Apply the same filters as current year
filtered_data_prev_year = analysis_data_prev_year.copy()
for level_name, selected_value in selected_filters.items():
    if selected_value != 'All':
        filtered_data_prev_year = filtered_data_prev_year[filtered_data_prev_year[level_name] == selected_value]

# Create table data for previous year (same grouping as current year)
table_data_prev_year = filtered_data_prev_year.groupby(selected_level_name).agg(
    margin_prev_year=('margin', 'sum')
).reset_index()

# Join current and previous year data
table_data = pd.merge(
    table_data,
    table_data_prev_year,
    on=selected_level_name,
    how='left'
)

# Calculate YoY change
table_data['yoy_margin_change'] = table_data.apply(
    lambda row: ((row['margin'] - row['margin_prev_year']) / row['margin_prev_year'] * 100) 
    if row['margin_prev_year'] != 0 
    else float('inf'),
    axis=1
)

# Format YoY change as percentage
table_data['yoy_margin_change'] = table_data['yoy_margin_change'].apply(
    lambda x: f"{x:.1f}%" if x != float('inf') else "N/A"
)
# Drop the margin_prev_year, cum_margin_share, and margin_share columns from the final table_data
table_data = table_data.drop(columns=['margin_prev_year', 'cum_margin_share', 'margin_share'])

# Display the table
st.header("Detailed Data Table")
st.dataframe(table_data)

st.markdown("---")  # Divider

# Input for Sous-Familles
st.header("4. Model Hierarchy Identification Agent")
st.write("Enter a Sous-Famille to identify the optimization level.")
sous_famille_input = st.text_input("Enter a Sous-Famille")

example_models = attributes[attributes['key'] == 'Modèle']['value'].unique().tolist()


if 'feedback' not in st.session_state:
    st.session_state.feedback = ""

if 'validation_status' not in st.session_state:
    st.session_state.validation_status = ""

if st.button("Identify Optimization Levels"):
    # Initialize sous_famille to None
    sous_famille = None
    
    # Check if user provided input
    if sous_famille_input.strip() == "":
        st.error("Please enter a Sous-Famille.")
    else:
        # Assign the input to sous_famille
        sous_famille = sous_famille_input.strip()
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            st.error("OPENAI_API_KEY not found in environment variables.")
        else:
            with st.spinner("Identifying optimization levels..."):
                try:
                    # Pass the single Sous-Famille to the function
                    optimization_df, logged_text = identify_optimization_levels_with_ai(hierarchy, [sous_famille], api_key, example_models)
                    
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
            save_feedback(feedback_data)
            st.session_state.validation_status = "success"
            st.session_state.feedback = ""
            st.session_state.optimization_df = None
            st.session_state.logged_text = ""
            st.success(f"Feedback for '{row['Sous-Famille']}' has been saved successfully.")
        