#%%
"""
Code to load Alpaca-formatted car forum data into Databricks SQL
"""
import json
import pandas as pd
from rich.progress import track
from rich.console import Console
from databricks.connect import DatabricksSession

#%%
spark = (
    DatabricksSession.builder.profile("DEFAULT").remote(serverless=True).getOrCreate()
)

console = Console()

def load_jsonl_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load a JSONL file into a pandas DataFrame
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        DataFrame containing the JSONL data
    """
    console.print(f"[bold green] Loading data from {file_path}...[/bold green]")
    
    # Read the JSONL file line by line
    data = []
    with open(file_path, 'r') as f:
        for line in track(f, description=f"Reading {file_path}"):
            # Parse each line as JSON
            try:
                record = json.loads(line)
                data.append(record)
            except json.JSONDecodeError as e:
                console.print(f"[bold red]Error parsing line: {e}[/bold red]")
                continue
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    console.print(f"[bold green]Successfully loaded {len(df)} records from {file_path}[/bold green]")
    
    return df





#%%   
    # Load training data
train_df = load_jsonl_to_dataframe("alpaca_car_forum_train.jsonl")

# count of rows with empty output
train_df[train_df['output'].str.strip() == ''].shape[0]

#remove rows with empty output
train_df = train_df[train_df['output'].str.strip() != '']

# Load validation data
val_df = load_jsonl_to_dataframe("alpaca_car_forum_val.jsonl")

# count of rows with empty output
val_df[val_df['output'].str.strip() == ''].shape[0]

#remove rows with empty output
val_df = val_df[val_df['output'].str.strip() != '']



#spdf_train
spdf_train = spark.createDataFrame(train_df)
spdf_train.show()

#spdf_val
spdf_val = spark.createDataFrame(val_df)
spdf_val.show()



# Upload to Databricks

spdf_train.write.format("delta").mode("overwrite").saveAsTable("main.sgfs.car_forum_train")
spdf_val.write.format("delta").mode("overwrite").saveAsTable("main.sgfs.car_forum_val")


console.print("[bold green]Upload complete![/bold green]") 

# %%
