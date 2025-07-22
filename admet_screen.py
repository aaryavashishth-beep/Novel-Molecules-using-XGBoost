import requests
import pandas as pd
import multiprocessing as mp
import torch  # For GPU detection (optional)
from tqdm import tqdm

# Check GPU availability
use_gpu = torch.cuda.is_available()
print(f"GPU Available: {use_gpu}")

# Load SMILES
with open("smiles.txt", "r") as f:
    smiles_list = [line.strip() for line in f.readlines()]

# Function to Check ADMET Properties
def check_admet(smiles):
    url = f"https://biosig.lab.uq.edu.au/pkcsm/prediction/admet/{smiles}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Apply Filtering Conditions
            if data.get("Hepatotoxicity") == "No" and data.get("hERG_inhibition") == "No":
                return smiles
    except requests.exceptions.RequestException:
        return None
    return None

# Use Multiprocessing to Speed Up Requests
if __name__ == "__main__":
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(check_admet, smiles_list), total=len(smiles_list), desc="Screening SMILES"))

    # Filter Out None Values
    filtered_smiles = [smi for smi in results if smi]

    # Save Filtered SMILES
    with open("filtered_smiles.txt", "w") as f:
        f.writelines("\n".join(filtered_smiles))

    print(f"Screening complete! {len(filtered_smiles)} compounds passed.")
