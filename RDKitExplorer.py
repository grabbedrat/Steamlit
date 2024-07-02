import streamlit as st
import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem import Draw
import pandas as pd
import numpy as np
from stmol import showmol
import plotly.express as px
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(layout="wide", page_title="Molecular Analyzer")

# Function to toggle code visibility
def toggle_code(code, label="Show/Hide Code", language="python"):
    with st.expander(label):
        st.code(code, language=language)

# Custom CSS for better styling
st.markdown("""
<style>
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stApp > header {
        background-color: transparent;
    }
    .main > div {
        padding-top: 1rem;
    }
    h1, h2, h3 {
        margin-bottom: 0.5rem;
    }
    .stTable td, .stTable th {
        padding: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Advanced Molecular Analysis with Code")

# SMILES input at the top
st.subheader("Molecule Input")
example_smiles = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
    "Custom": ""
}

col_examples, col_smiles = st.columns([1, 2])
with col_examples:
    selected_example = st.selectbox("Choose a molecule or enter custom SMILES:", list(example_smiles.keys()))

with col_smiles:
    if selected_example == "Custom":
        smiles = st.text_input("Enter SMILES string:")
    else:
        smiles = example_smiles[selected_example]
    st.text(f"SMILES: {smiles}")

    # Displaying additional molecule information
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        st.text(f"Formula: {rdMolDescriptors.CalcMolFormula(mol)}")
        st.text(f"Canonical SMILES: {Chem.MolToSmiles(mol, canonical=True)}")
        st.text(f"InChI: {Chem.MolToInchi(mol)}")
        st.text(f"InChI Key: {Chem.InchiToInchiKey(Chem.MolToInchi(mol))}")

# Molecule setup
toggle_code("""
# Create a molecule object from SMILES string
mol = Chem.MolFromSmiles(smiles)
# Add hydrogen atoms to the molecule
mol = Chem.AddHs(mol)
# Generate a 3D conformation for the molecule
AllChem.EmbedMolecule(mol, randomSeed=42)
# Optimize the molecule's geometry using the MMFF94 force field
AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
""", "Code")

mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, randomSeed=42)
AllChem.MMFFOptimizeMolecule(mol, maxIters=200)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("3D Visualization")
    
    toggle_code("""
def render_mol(xyz):
    xyzview = py3Dmol.view(width=400, height=300)
    xyzview.addModel(xyz, "xyz")
    xyzview.setStyle({'stick':{}})
    xyzview.setBackgroundColor('white')
    xyzview.zoomTo()
    showmol(xyzview, height=300, width=400)

xyz = Chem.MolToXYZBlock(mol)
render_mol(xyz)
    """, "Code")
    
    def render_mol(xyz):
        xyzview = py3Dmol.view(width=400, height=300)
        xyzview.addModel(xyz, "xyz")
        xyzview.setStyle({'stick':{}})
        xyzview.setBackgroundColor('white')
        xyzview.zoomTo()
        showmol(xyzview, height=300, width=400)
    
    xyz = Chem.MolToXYZBlock(mol)
    render_mol(xyz)

    st.subheader("2D Structure")
    toggle_code("""
img = Draw.MolToImage(mol)
st.image(img, use_column_width=True)
    """, "Show/Hide 2D Structure Code")
    img = Draw.MolToImage(mol)
    st.image(img, use_column_width=True)

with col2:
    st.subheader("Molecular Properties")
    
    toggle_code("""
props = {
    "Mol Weight": f"{Descriptors.ExactMolWt(mol):.2f}",
    "LogP": f"{Crippen.MolLogP(mol):.2f}",
    "H-Bond Donors": Descriptors.NumHDonors(mol),
    "H-Bond Acceptors": Descriptors.NumHAcceptors(mol),
    "Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
    "TPSA": f"{Descriptors.TPSA(mol):.2f}",
    "Rings": rdMolDescriptors.CalcNumRings(mol),
    "Fraction SP3": f"{rdMolDescriptors.CalcFractionCSP3(mol):.2f}"
}

df = pd.DataFrame.from_dict(props, orient='index', columns=['Value'])
st.table(df)
    """, "Code")
    
    props = {
        "Mol Weight": f"{Descriptors.ExactMolWt(mol):.2f}",
        "LogP": f"{Crippen.MolLogP(mol):.2f}",
        "H-Bond Donors": Descriptors.NumHDonors(mol),
        "H-Bond Acceptors": Descriptors.NumHAcceptors(mol),
        "Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
        "TPSA": f"{Descriptors.TPSA(mol):.2f}",
        "Rings": rdMolDescriptors.CalcNumRings(mol),
        "Fraction SP3": f"{rdMolDescriptors.CalcFractionCSP3(mol):.2f}"
    }
    
    df = pd.DataFrame.from_dict(props, orient='index', columns=['Value'])
    st.table(df)

    st.subheader("Solubility Prediction")
    toggle_code("""
def predict_solubility(mol):
    logp = Crippen.MolLogP(mol)
    mp = Descriptors.MolWt(mol) * 0.25 + 50
    log_solubility = 0.5 - 0.01 * (mp - 25) - logp
    return 10**log_solubility

solubility = predict_solubility(mol)
st.write(f"Est. Aqueous Solubility: {solubility:.2e} mol/L")
    """, "Code")
    def predict_solubility(mol):
        logp = Crippen.MolLogP(mol)
        mp = Descriptors.MolWt(mol) * 0.25 + 50
        log_solubility = 0.5 - 0.01 * (mp - 25) - logp
        return 10**log_solubility

    solubility = predict_solubility(mol)
    st.write(f"Est. Aqueous Solubility: {solubility:.2e} mol/L")

st.subheader("Molecular Fingerprint")

toggle_code("""
def plot_fingerprint(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fp_array = np.array(fp)
    fig = px.bar(y=fp_array, labels={'y': 'Bit Value', 'index': 'Bit Position'})
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
    return fig

fp_fig = plot_fingerprint(mol)
st.plotly_chart(fp_fig, use_container_width=True)
""", "Code")

def plot_fingerprint(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fp_array = np.array(fp)
    fig = px.bar(y=fp_array, labels={'y': 'Bit Value', 'index': 'Bit Position'})
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
    return fig

fp_fig = plot_fingerprint(mol)
st.plotly_chart(fp_fig, use_container_width=True)

st.subheader("Atom Contributions to LogP")

toggle_code("""
def plot_atom_logp_contributions(mol):
    contribs = Crippen._GetAtomContribs(mol)
    atom_logps = [contrib[0] for contrib in contribs]
    atom_symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
    
    fig = px.scatter(x=list(range(mol.GetNumAtoms())), y=atom_logps, 
                     hover_name=atom_symbols, labels={'x': 'Atom Index', 'y': 'LogP Contribution'})
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
    return fig

logp_fig = plot_atom_logp_contributions(mol)
st.plotly_chart(logp_fig, use_container_width=True)
""", "Code")

def plot_atom_logp_contributions(mol):
    contribs = Crippen._GetAtomContribs(mol)
    atom_logps = [contrib[0] for contrib in contribs]
    atom_symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
    
    fig = px.scatter(x=list(range(mol.GetNumAtoms())), y=atom_logps, 
                     hover_name=atom_symbols, labels={'x': 'Atom Index', 'y': 'LogP Contribution'})
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
    return fig

logp_fig = plot_atom_logp_contributions(mol)
st.plotly_chart(logp_fig, use_container_width=True)

st.info("""
This app demonstrates advanced molecular analysis using RDKit, including:
3D and 2D visualization, key molecular properties, Morgan fingerprint visualization,
analysis of atom contributions to LogP, and aqueous solubility prediction.
The code for each section is displayed to help users understand how these analyses are performed.
""")