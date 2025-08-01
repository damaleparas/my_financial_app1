import pandas as pd

def get_info(path):
    """Extract financial metrics from balance sheet CSV"""
    
    # Load the balance sheet
    df = pd.read_csv(path)
    df.rename(columns={df.columns[0]: "Metric"}, inplace=True)
    df['Metric'] = df['Metric'].str.lower().str.strip()

    # Define a dictionary of synonyms for each key metric
    metric_map = {
        "revenue": ["revenue", "sales", "turnover", "total income", "operating revenue"],
        "net_income": ["net income", "net profit", "profit after tax", "pat"],
        "total_assets": ["total assets", "assets"],
        "total_debt": ["total debt", "borrowings", "loans"],
        "net_debt": ["net debt"],
        "book_value": ["book value", "tangible book value"],
         "profit": ["net income", "net profit", "profit after tax", "pat", 
               "profit", "operating profit", "gross profit", "ebitda"]
    }

    # Create an empty dict to store extracted results
    extracted_metrics = {}

    # Loop through each target metric group
    for key, synonyms in metric_map.items():
        for synonym in synonyms:
            match = df[df['Metric'].str.contains(synonym, case=False, na=False)]
            if not match.empty:
                extracted_metrics[key] = match
                break  # Stop at first match to avoid duplicates

    return extracted_metrics