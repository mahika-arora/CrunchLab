import numpy as np
import pandas as pd
import rascaline
from skmatter.decomposition import PCovR
from sklearn.linear_model import RidgeCV
from metatensor import TensorMap, Labels
import matplotlib.pyplot as plt
import ase
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import os

np.set_printoptions(threshold=np.inf)

# Read SMILES from a CSV file
input_file = 'input_smiles.csv'
smiles_data = pd.read_csv(input_file)

# Extract the SMILES column
smiles_column_name = 'SMILES'  
smiles_list = smiles_data[smiles_column_name]

# Prepare to store SMILES and SOAP vectors
smiles_and_soap_vectors = []
avg_soap=[]

# Loop over each SMILES string
for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)

    # Convert RDKit molecule to ASE atoms object
    atoms = ase.Atoms(
        symbols=[atom.GetSymbol() for atom in mol.GetAtoms()],
        positions=[(mol.GetConformer().GetAtomPosition(atom.GetIdx()).x,
                    mol.GetConformer().GetAtomPosition(atom.GetIdx()).y,
                    mol.GetConformer().GetAtomPosition(atom.GetIdx()).z)
                   for atom in mol.GetAtoms()],
        cell=[15, 15, 15]
    )

    # Hyperparameters for the SOAP descriptors
    hypers = {
        "cutoff": 4.0,  #TO-DO Change based on cutoff radius being worked with
        "max_radial": 8,
        "max_angular": 6,
        "atomic_gaussian_width": 0.3,
        "radial_basis": {"Gto": {}},
        "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
        "center_atom_weight": 1.0,
    }

    calculator = rascaline.SoapPowerSpectrum(**hypers)
    names_lis = ['center_type', 'neighbor_1_type', 'neighbor_2_type']
    #Carbon-Centered
    combi_knock = [[6, 6, 6], [6, 7, 6], [6, 7, 7]]
    #Uncomment the line below and comment the line above if working with Nitrogen-centered 
    #combi_knock = [[7, 6, 6], [7, 7, 6], [7, 7, 7]]  
    labels = Labels(names=names_lis, values=np.array(combi_knock, dtype=np.int32))

    rho2i = calculator.compute([atoms], selected_keys=labels)
    rho2i_i = rho2i.keys_to_properties(['neighbor_1_type', 'neighbor_2_type'])
    feats_env = rho2i_i.blocks(center_type=6)[0].values #TO-DO change center_type to 7 if calculating for Nitrogen-centered
    soap_list = feats_env.tolist()
    #for nitrogen 
    #if feat_env.shape[0] != 0: then lines 67-68 
    sum_feats = feats_env.sum(axis=0) ## Summation
    sum_feats /= feats_env.shape[0]
    
    print(feats_env.shape[0])

    # Append the SMILES and the corresponding SOAP vectors to the list
    for soap_vector in soap_list:
        smiles_and_soap_vectors.append([smiles] + soap_vector)
        
    avg_soap.append(sum_feats.tolist())
    
# Convert the list to a NumPy array
smiles_and_soap_array = np.array(smiles_and_soap_vectors, dtype=object)
avg_soap_array = np.array(avg_soap,dtype = object)

# Save the NumPy array to a CSV file
np.savetxt("smiles_and_soapVectors.csv", smiles_and_soap_array, fmt='%s', delimiter=",")
np.savetxt("avg_soap_4C.csv",avg_soap_array,delimiter = ",")  
    
    
    