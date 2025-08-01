from .financial_engine import get_info

def model_passing_for_prediction(extracted_metrics):
    """Convert extracted metrics to ML-ready feature vector"""
    
    # Define the years to extract
    target_years = ["2021-03-31", "2022-03-31", "2023-03-31", "2024-03-31", "2025-03-31"]
    
    # This will hold the final flattened features
    features = {}
    
    # Function to extract multiple years for each metric
    def extract_multi_year(metric_name):
        df = extracted_metrics.get(metric_name)
        if df is not None:
            for year in target_years:
                col_name = f"{metric_name}_{year[:4]}"
                try:
                    value = float(df[year].values[0])
                except:
                    value = 0.0
                features[col_name] = value
    
    # Define which metrics to extract across all years
    metrics_to_use = ["revenue", "profit", "total_debt", "total_assets", "net_debt", "book_value"]
    
    # Run extraction for all metrics
    for metric in metrics_to_use:
        extract_multi_year(metric)
    
    # Convert to list for model input
    input_vector = [v for v in features.values()]
    return input_vector