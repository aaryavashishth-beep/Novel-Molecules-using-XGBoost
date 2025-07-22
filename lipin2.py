print(f" Filtered molecules saved to '{tdp43_filtered_path}'")
#  Step 5: Apply TDP-43 Active Site Filtering
print("=9 Step 5: Filtering molecules for TDP-43 active sites...")
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
import numpy as np

# =9 Step 5: Set Active Site Coordinates for TDP-43
tdp_x, tdp_y, tdp_z = 49.4174, 20.0641, -46.7476  # Active site coordinates

# =9 Function: Compute Distance Between Molecule & Active Site
def compute_distance_to_tdp43(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        try:
            # Generate 3D Conformer
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())  # Generate 3D structure
            AllChem.UFFOptimizeMolecule(mol)  # Optimize the conformation

            # Compute Molecule Centroid
            conf = mol.GetConformer()
            mol_coords = np.mean([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())], axis=0)
            
            # Compute Distance from TDP-43 Active Site
            dist = np.linalg.norm(np.array([tdp_x, tdp_y, tdp_z]) - mol_coords)
            return dist
        except:
            return np.nan  # Return NaN if molecule fails to generate a 3D structure
    return np.nan

# Step 5: Apply TDP-43 Active Site Filtering with Multiprocessing
print("=9 Step 5: Filtering molecules for TDP-43 active site using multiprocessing...")
tdp_x, tdp_y, tdp_z = 49.4174, 20.0641, -46.7476  # Active site coordinates

def compute_distance_to_tdp43(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.UFFOptimizeMolecule(mol)
            conf = mol.GetConformer()
            mol_coords = np.mean([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())], axis=0)
            dist = np.linalg.norm(np.array([tdp_x, tdp_y, tdp_z]) - mol_coords)
            return dist
        except:
            return np.nan
    return np.nan
