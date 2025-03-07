#%%
from datasets import load_dataset
import pandas as pd  # noqa: F401
import json
import re
from rich.progress import track
from sklearn.model_selection import train_test_split

# Load the JSONL dataset
dataset = load_dataset("json", data_files="llm_sample.jsonl", split="train")

#%%
# View basic information about the dataset
print("Dataset Info:")
print(f"Number of examples: {len(dataset)}")
print(f"\nFeatures: {dataset.features}")

# Display first few examples
print("\nFirst 3 examples:")
for i, example in enumerate(dataset[:3]):
    print(f"\nExample {i+1}:")
    print(example)

#%%
# Convert dataset to pandas DataFrame for easier manipulation
df = dataset.to_pandas()

#%%
# Helper function to clean text from Unicode characters and other issues
def clean_text(text):
    if not isinstance(text, str):
        return text
        
    # Replace \u00a0 (non-breaking space) with regular space
    cleaned = text.replace('\u00a0', ' ')
    
    # Replace multiple consecutive spaces with a single space
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()
    
    return cleaned

# Function to convert a forum question to Alpaca format
def convert_to_alpaca_format(row):
    # Extract the question for the input field and clean it
    input_text = clean_text(row['question'])
    
    # Combine all replies into a single answer for the output field and clean them
    output = "\n\n".join([clean_text(reply) for reply in row['replies_text'] if reply])
    
    # Start building the instruction with the required prefix
    instruction = "Answer the following question after understanding the specifics of the vehicle in question."
    
    # Build a more natural description of the vehicle
    vehicle_details = []
    
    # Basic vehicle info in sentence form
    basic_info = []
    
    # Title
    if 'title' in row and row['title']:
        title = clean_text(row['title'])
        if title:
            basic_info.append(f"It is a {title}")
    
    # VIN and Mileage
    vin_mileage = []
    if 'VIN' in row and row['VIN']:
        vin = clean_text(row['VIN'])
        if vin:
            vin_mileage.append(f"with the VIN {vin}")
    
    if 'Mileage' in row and row['Mileage']:
        mileage = clean_text(row['Mileage'])
        if mileage:
            vin_mileage.append(f"and a mileage of {mileage}")
    
    if vin_mileage:
        basic_info.append(" ".join(vin_mileage))
    
    # Model, Engine, Trans
    engine_info = []
    if 'model' in row and row['model']:
        model = clean_text(row['model'])
        if model:
            engine_info.append(f"This {model} is")
    else:
        engine_info.append("This vehicle is")
    
    if 'Engine' in row and row['Engine']:
        engine = clean_text(row['Engine'])
        if engine:
            engine_info.append(f"a {engine} engine")
    
    if 'Trans' in row and row['Trans']:
        trans = clean_text(row['Trans'])
        if trans:
            engine_info.append(f"with {trans} transmission")
    
    if len(engine_info) > 1:  # Only add if we have more than just the "This vehicle is" part
        basic_info.append(" ".join(engine_info))
    
    # Combine basic information into complete sentences
    if basic_info:
        vehicle_details.append(". ".join(basic_info) + ".")
    
    # Additional details in a more structured format
    additional_details = []
    
    # Delivery system
    if 'Delivery' in row and row['Delivery']:
        delivery = clean_text(row['Delivery'])
        if delivery:
            additional_details.append(f"Delivery system: {delivery}")
    
    # Emissions
    if 'Emissions' in row and row['Emissions']:
        emissions = clean_text(row['Emissions'])
        if emissions:
            additional_details.append(f"Emissions: {emissions}")
    
    # Symptoms
    if 'Symptoms' in row and row['Symptoms']:
        symptoms = clean_text(row['Symptoms'])
        if symptoms:
            additional_details.append(f"Symptoms: {symptoms}")
    
    # When issue occurs
    if 'Occurs' in row and row['Occurs']:
        occurs = clean_text(row['Occurs'])
        if occurs:
            additional_details.append(f"Issue occurs: {occurs}")
    
    # Affected component
    if 'Affected' in row and row['Affected']:
        affected = clean_text(row['Affected'])
        if affected:
            additional_details.append(f"Affected component: {affected}")
    
    # Conditions
    if 'Conditions' in row and row['Conditions']:
        conditions = clean_text(row['Conditions'])
        if conditions:
            additional_details.append(f"Conditions: {conditions}")
    
    # Add additional details if available
    if additional_details:
        vehicle_details.append(" ".join(additional_details))
    
    # Combine all vehicle details into the instruction
    if vehicle_details:
        instruction += "\n\n" + "\n".join(vehicle_details)
    
    # Create Alpaca format entry
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output
    }

#%%
# Clean the DataFrame before processing
print("Cleaning data...")
for column in df.columns:
    if df[column].dtype == 'object':  # Only process string/object columns
        df[column] = df[column].apply(lambda x: clean_text(x) if isinstance(x, str) else x)

# Convert all examples to Alpaca format
alpaca_data = []
for _, row in track(df.iterrows(), total=len(df), description="Converting to Alpaca format"):
    try:
        alpaca_entry = convert_to_alpaca_format(row)
        alpaca_data.append(alpaca_entry)
    except Exception as e:
        print(f"Error processing a row: {e}")

#%%
# Save the Alpaca-formatted data to a new JSONL file
with open("alpaca_car_forum_data.jsonl", "w") as f:
    for item in alpaca_data:
        f.write(json.dumps(item) + "\n")

print(f"Conversion complete! {len(alpaca_data)} examples converted to Alpaca format.")

#%%
# Display a few examples of the converted data
print("\nSample of converted data:")
for i, example in enumerate(alpaca_data[:3]):
    print(f"\nExample {i+1}:")
    print(json.dumps(example, indent=2))

#%%
# Optional: Create a validation set
# If you want to split the data into training and validation sets
train_data, val_data = train_test_split(alpaca_data, test_size=0.1, random_state=42)

with open("alpaca_car_forum_train.jsonl", "w") as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")

with open("alpaca_car_forum_val.jsonl", "w") as f:
    for item in val_data:
        f.write(json.dumps(item) + "\n")

print(f"Split into {len(train_data)} training examples and {len(val_data)} validation examples.")
