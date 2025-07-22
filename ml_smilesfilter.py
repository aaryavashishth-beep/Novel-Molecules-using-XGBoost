# Script 1: Machine Learning (XGBoost) for TDP-43 Inhibitor Prediction
# Install necessary dependencies
import subprocess
import multiprocessing
from tqdm import tqdm

# Install dependencies
dependencies = [
    "pandas", "numpy", "rdkit", "xgboost", "scikit-learn",
    "requests", "torch", "torchvision", "torchaudio"
]

for package in dependencies:
    subprocess.run(["pip", "install", package], check=True)

print("All dependencies installed successfully.")

import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import requests
import sqlite3
import os

# Define TDP-43 active site coordinates and residues
TDP_X, TDP_Y, TDP_Z = 49.4174, 20.0641, -46.7476  # Active site coordinates
TDP_ACTIVE_SITES = {"CYS", "HIS", "ASP", "GLU", "ARG", "LYS"}  # Residues
IC50_THRESHOLD = 10  # Example IC50 threshold

# Step 1: Download and Extract ChEMBL Dataset
chembl_url = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_35/chembl_35_sqlite.tar.gz"
chembl_tar_gz = "chembl_35_sqlite.tar.gz"
chembl_db = "chembl_35.db"

if not os.path.exists(chembl_db):
    print("Downloading ChEMBL dataset...")
    response = requests.get(chembl_url, stream=True)
    with open(chembl_tar_gz, "wb") as file:
        file.write(response.content)
    print("Download complete. Extracting...")
    subprocess.run(["tar", "-xzf", chembl_tar_gz])

# Step 2: Process ChEMBL Dataset and Filter Based on Active Site
conn = sqlite3.connect(chembl_db)
cursor = conn.cursor()

query = "SELECT molregno, canonical_smiles, standard_value FROM activities WHERE standard_type='IC50'"
chembl_data = pd.read_sql(query, conn)
conn.close()
chembl_data.columns = ['MolID', 'SMILES', 'IC50']

# Function to generate molecules based on active site properties
def generate_molecule(smiles, active_sites, ic50):
    mol = Chem.MolFromSmiles(smiles)
    if mol and ic50 <= IC50_THRESHOLD:
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in active_sites:
                return smiles
    return None

with multiprocessing.Pool() as pool:
    generated_molecules = list(tqdm(
        pool.imap(lambda x: generate_molecule(x[0], TDP_ACTIVE_SITES, x[1]), zip(chembl_data["SMILES"], chembl_data["IC50"])),
        total=len(chembl_data)
    ))

generated_molecules = [mol for mol in generated_molecules if mol]
generated_df = pd.DataFrame(generated_molecules, columns=["SMILES"])
generated_df.to_csv("generated_molecules.csv", index=False)
print("Generated molecules saved.")

# Step 3: Compute Molecular Descriptors
def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol)
        ]
    return [np.nan] * 6

with multiprocessing.Pool() as pool:
    descriptors_list = list(tqdm(
        pool.imap(compute_descriptors, generated_df['SMILES']),
        total=len(generated_df)
    ))

descriptors_df = pd.DataFrame(descriptors_list, columns=["MolWt", "LogP", "TPSA", "NumRotBonds", "HBD", "HBA"])
generated_df = pd.concat([generated_df, descriptors_df], axis=1).dropna()

if len(generated_df) < 10:
    print("XXXXWarning: Less than 10 molecules generated. Consider refining criteria.")

# Train XGBoost Model
X = generated_df[["MolWt", "LogP", "TPSA", "NumRotBonds", "HBD", "HBA"]]
y = np.random.randint(0, 2, size=len(X))  # Placeholder labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("ROC AUC Score:", roc_auc_score(y_test, predictions))

# Step 4: Screening
with multiprocessing.Pool() as pool:
    screening_descriptors_list = list(tqdm(
        pool.imap(compute_descriptors, generated_df['SMILES']),
        total=len(generated_df)
    ))

screening_descriptors = pd.DataFrame(screening_descriptors_list, columns=["MolWt", "LogP", "TPSA", "NumRotBonds", "HBD", "HBA"])
generated_df = pd.concat([generated_df, screening_descriptors], axis=1).dropna()

predicted_activity = model.predict(generated_df[["MolWt", "LogP", "TPSA", "NumRotBonds", "HBD", "HBA"]])
generated_df['Predicted_Activity'] = predicted_activity
generated_df.to_csv("predicted_inhibitors_xgboost.csv", index=False)
print("Screening completed. Results saved.")

