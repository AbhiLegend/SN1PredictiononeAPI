import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import time  # Import time for simulating delays

# Updated function to generate synthetic data with validity checks
def generate_synthetic_data():
    base_smiles = ["CCCl", "CCBr", "CCC", "CCO"]  # Base molecules
    functional_groups = ["Cl", "Br", "I", "O", "C=O"]  # Functional groups to add
    
    synthetic_molecules = []
    for base in base_smiles:
        base_mol = Chem.MolFromSmiles(base)
        if not base_mol:  # Check if the molecule is valid
            continue
        for fg in functional_groups:
            synthetic_smiles = base + "." + fg  # Use "." to denote separate molecules
            synthetic_mol = Chem.MolFromSmiles(synthetic_smiles)
            if synthetic_mol:  # Ensure the resulting molecule is valid
                synthetic_molecules.append(synthetic_smiles)
    
    # Remove duplicates
    synthetic_molecules = list(set(synthetic_molecules))
    
    # Generate descriptors
    data = []
    for smi in synthetic_molecules:
        mol = Chem.MolFromSmiles(smi)
        if mol:  # Additional check for validity
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            sn1_reaction = np.random.choice([0, 1])  # Randomly assign SN1 reaction outcome
            data.append({"SMILES": smi, "Molecular Weight": mw, "logP": logp, "SN1": sn1_reaction})
    
    return pd.DataFrame(data)

# Placeholder function for model training
def train_model(df):
    X = df[['Molecular Weight', 'logP']]
    y = df['SN1']
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y)
    return clf

# Function to calculate descriptors
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:  # Ensure the molecule is valid before calculating descriptors
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        return mw, logp
    else:
        return None, None  # Return None if the molecule is invalid

# Streamlit UI
st.title("SN1 Reaction Predictor")

# Generate and display synthetic data (optional step for demonstration)
if st.checkbox("Generate synthetic data for training"):
    with st.spinner("Generating synthetic data..."):
        df_synthetic = generate_synthetic_data()
        st.write(df_synthetic.head())  # Display a preview of the synthetic data
        time.sleep(1)  # Simulate delay for data generation
        
        # Train model with synthetic data
        with st.spinner("Training model on synthetic data..."):
            model = train_model(df_synthetic)
            st.success("Model trained successfully on synthetic data!")
else:
    model = None  # Placeholder for model, you can load a pre-trained model here

user_input = st.text_input("Enter a SMILES string", "CCCl")

if user_input:
    mw, logp = calculate_descriptors(user_input)
    if mw and logp:  # Check if descriptors were successfully calculated
        st.write(f"Molecular Weight: {mw}, logP: {logp}")
        
        # Visualize the molecule
        mol = Chem.MolFromSmiles(user_input)
        if mol:  # Ensure the molecule is valid before attempting to visualize
            mol_image = Draw.MolToImage(mol)
            st.image(mol_image, caption="Input Molecule")
        
        if model:  # Check if the model exists before predicting
            # Predict the reaction outcome
            prediction = model.predict(np.array([[mw, logp]]))
            outcome = "likely" if prediction[0] == 1 else "unlikely"
            st.write(f"The molecule is {outcome} to undergo an SN1 reaction.")
    else:
        st.error("Invalid SMILES string. Please enter a valid SMILES.")
