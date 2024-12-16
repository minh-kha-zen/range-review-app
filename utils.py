import pandas as pd

def prepare_table_data(
    material_ids, 
    hierarchy, 
    sales, 
    master, 
    optimization_level_id,
    previous_year_sales,
    model_level_id=None
):
    """
    Prepare the table data for the given sous-famille and optimization levels.

    Args:
        material_ids (list): List of material IDs to filter by.
        hierarchy (pd.DataFrame): The hierarchy DataFrame.
        sales (pd.DataFrame): The sales DataFrame.
        master (pd.DataFrame): The master DataFrame.
        optimization_level_id (int): The hierarchy level ID for the optimization.
        previous_year_sales (pd.DataFrame, optional): Sales data for the previous year.

    Returns:
        pd.DataFrame: The prepared table data with additional metrics.
    """
    
    # Filter sales data based on the retrieved material_ids
    filtered_sales = sales[sales['material_id'].isin(material_ids)]
    
    # Filter hierarchy by the identified optimization level ID
    filtered_hierarchy = hierarchy[hierarchy['level'] == optimization_level_id]
    
    # Join the model names from the 'name' column
    merged_data = pd.merge(filtered_sales, filtered_hierarchy, on='material_id', how='inner')
    
    # Join the master data to get the date_of_introduction
    merged_data = pd.merge(
        merged_data, 
        master[['material_id', 'date_of_introduction']], 
        on='material_id', 
        how='left'
    )
    
    # Group by model name and compile necessary columns
    table_data = merged_data.groupby('name').agg(
        quantity=('quantity', 'sum'),
        net_revenue=('net_revenue', 'sum'),
        margin=('margin', 'sum'),
        gross_revenue=('gross_revenue', 'sum'),
        latest_intro=('date_of_introduction', 'max')  # Latest introduction date
    ).reset_index()

    # Calculate additional metrics
    table_data['relative_margin'] = table_data['margin'] / table_data['net_revenue']

    table_data['avg_discount'] = (
        (table_data['gross_revenue'] - table_data['net_revenue']) 
        / table_data['gross_revenue']
    )
    
    # Calculate total SKUs
    table_data['total_skus'] = merged_data.groupby('name')['material_id'].nunique().reset_index(drop=True)

    # Rearrange columns for better readability
    table_data = table_data[[
        'name', 
        'quantity', 
        'net_revenue',
        'margin', 
        'relative_margin', 
        'avg_discount',
        'latest_intro',
        'total_skus'
    ]]

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
    # Note: This requires hierarchy DataFrame to have 'level_name' and 'level' columns
    def calculate_margin_spread(current_value):
        # Pivot the hierarchy to get all levels as columns
        hierarchy_pivoted = hierarchy.pivot(
            index='material_id',
            columns='level_name',
            values='name'
        ).reset_index()

        # Join the pivoted hierarchy with merged_data
        merged_data_with_hierarchy = pd.merge(
            merged_data,
            hierarchy_pivoted,
            on='material_id',
            how='left'
        )

        if model_level_id is None:
            lower_level = optimization_level_id + 1
        else:
            lower_level = model_level_id

        if lower_level in hierarchy['level'].values:
            current_level_ids = hierarchy[hierarchy['name'] == current_value]['material_id'].unique().tolist()
            lower_level_data = merged_data_with_hierarchy[merged_data_with_hierarchy['material_id'].isin(current_level_ids)]

            # Get the name of the next level
            lower_level_name = hierarchy[hierarchy['level'] == lower_level]['level_name'].iloc[0]

            if lower_level_name in lower_level_data.columns and not lower_level_data.empty:
                lower_level_margins = lower_level_data.groupby(lower_level_name).agg(
                    total_margin=('margin', 'sum'),
                    total_revenue=('net_revenue', 'sum')
                ).reset_index()

                lower_level_margins['relative_margin'] = lower_level_margins.apply(
                    lambda row: (row['total_margin'] / row['total_revenue'] * 100) if row['total_revenue'] != 0 else 0,
                    axis=1
                )

                max_margin = lower_level_margins['relative_margin'].max()
                min_margin = lower_level_margins['relative_margin'].min()
                return max_margin - min_margin
        return 0
    
    table_data['margin_spread'] = table_data['name'].apply(calculate_margin_spread)
    
    # Calculate YoY Change if previous_year_sales is provided
    if previous_year_sales is not None:
        # Join the model names from the 'name' column
        merged_data_prev_year = pd.merge(previous_year_sales, filtered_hierarchy, on='material_id', how='inner')

        # Merge with previous year data
        table_data_prev_year = merged_data_prev_year.groupby('name').agg(
            margin_prev_year=('margin', 'sum')
        )
        
        table_data = pd.merge(
            table_data, 
            table_data_prev_year, 
            on='name', 
            how='left'
        )
        
        # Calculate YoY change
        table_data['yoy_margin_change'] = table_data.apply(
            lambda row: f"{((row['margin'] - row['margin_prev_year']) / row['margin_prev_year'] * 100):.1f}%" 
            if row['margin_prev_year'] > 0 else "N/A",
            axis=1
        )
        
        # Fill NaN for models without previous year data
        table_data['yoy_margin_change'] = table_data['yoy_margin_change'].fillna("N/A")

    # 
    table_data['Total Margin'] = '€' + (table_data['Total Margin'] / 1_000_000).round(1).astype(str) + 'M'
    
    # Rename columns at the end
    table_data.rename(columns={
        'name': 'Model Name',
        'quantity': 'Total Quantity',
        'net_revenue': 'Total Net Revenue',
        'margin': 'Total Margin',
        'relative_margin': 'Relative Margin',
        'avg_discount': 'Average Discount',
        'latest_intro': 'Latest Introduction',
        'total_skus': 'Total SKUs',
        'margin_spread': 'Margin Spread',
        'yoy_margin_change': 'YoY Margin Change',
        'pareto': 'Pareto'
    }, inplace=True)
    
    # Format the columns for display
    table_data['Total Margin'] = '€' + (table_data['Total Margin'] / 1_000_000).round(1).astype(str) + 'M'
    table_data['Total Quantity'] = (table_data['Total Quantity'] / 1_000).round(1).astype(str) + 'K'
    table_data['Total Net Revenue'] = '€' + (table_data['Total Net Revenue'] / 1_000_000).round(1).astype(str) + 'M'
    table_data['Total SKUs'] = table_data['Total SKUs'].astype(str)  # Assuming this is already in the correct format
    table_data['Relative Margin'] = (table_data['Relative Margin'] * 100).round(1).astype(str) + '%'

    # Drop columns that are not needed
    columns_to_drop = [
        'margin_share',
        'cum_margin_share',
        'margin_prev_year',
        'Pareto',
        'Total Quantity'
    ]
    table_data.drop(columns=columns_to_drop, inplace=True)

    # Return the final table_data
    return table_data