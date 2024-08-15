import skmatter
from skmatter.preprocessing import StandardFlexibleScaler
import numpy as np
import pandas as pd
import rascaline
from skmatter.decomposition import PCovR
from sklearn.linear_model import RidgeCV
from metatensor import TensorMap, Labels
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import ase
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import os

np.set_printoptions(threshold=np.inf)

input_file = 'soapVectors4C.csv'   #change input file name based on the file being worked on
soap_vectors = np.loadtxt(input_file,delimiter = ",")
SOAP_transform = StandardFlexibleScaler(with_mean=True, with_std=True, column_wise=False)
new_frame = SOAP_transform.fit_transform(soap_vectors)

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(soap_vectors)
# Save the SOAP vectors to an output file
pca_df = np.savetxt("pca4C.csv", principalComponents, delimiter = ",")

print("Shape:", principalComponents.shape)