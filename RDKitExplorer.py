import streamlit as st
import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem import Draw
import pandas as pd
import numpy as np
from stmol import showmol
import plotly.express as px

st.set_page_config(layout="wide", page_title="Molecular Analyzer")

st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .stApp > header {
        background-color: transparent;
    }
    h1, h2, h3 {
        margin-bottom: 0.5rem;
    }
    .stTable td, .stTable th {
        padding: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Advanced Molecular Analysis")

with st.sidebar:
    st.header("Molecule Input")
    smiles = st.text_input("Enter SMILES string", "CC(=O)OC1=CC=CC=C1C(=O)O")

mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, randomSeed=42)
AllChem.MMFFOptimizeMolecule(mol, maxIters=200)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("3D Visualization")
    
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
    img = Draw.MolToImage(mol)
    st.image(img, use_column_width=True)

with col2:
    st.subheader("Molecular Properties")
    
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
    def predict_solubility(mol):
        logp = Crippen.MolLogP(mol)
        mp = Descriptors.MolWt(mol) * 0.25 + 50
        log_solubility = 0.5 - 0.01 * (mp - 25) - logp
        return 10**log_solubility

    solubility = predict_solubility(mol)
    st.write(f"Est. Aqueous Solubility: {solubility:.2e} mol/L")

st.subheader("Molecular Fingerprint")

def plot_fingerprint(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fp_array = np.array(fp)
    fig = px.bar(y=fp_array, labels={'y': 'Bit Value', 'index': 'Bit Position'})
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
    return fig

fp_fig = plot_fingerprint(mol)
st.plotly_chart(fp_fig, use_container_width=True)

st.subheader("Atom Contributions to LogP")

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
These features provide insights into molecular structure and properties.
""")