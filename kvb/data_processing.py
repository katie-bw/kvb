import pandas as pd 
from rdkit import Chem 
from rdkit.Chem import AllChem, PandasTools, DataStructs, rdchem, Draw, rdMolDescriptors, RDConfig
import numpy as np
import selfies as sf
import plotly.express as px
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datamol as dm
import exmol
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, classification_report, confusion_matrix, accuracy_score,roc_auc_score, precision_score, balanced_accuracy_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import time
import os 
import sys
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
sys.path.append(os.path.join(os.environ['CONDA_PREFIX'],'share','RDKit','Contrib'))
from rdkit.Contrib.SA_Score import sascorer
#Import SAScore file
#Import packages


def get_df_from_smi(file_path: str, drop_dups = False):
    """
    Takes the file path for the .smi file and generates a DataFrame for the SMILES. 

    Parameters: 

    file_path (str): The path to where the .smi is located

    drop_dups (bool): Whether of not duplicates should be dropped, defaults to False

    Returns: 

    df (DataFrame): The DataFrame for the molecules 
    """
    with open(file_path, 'r') as f: 
        smi_lis = f.readlines()
    f.close()
    #Read in the data 
    smi_lis = [i.strip() for i in smi_lis]
    #Removes any white space and line breaks 
    df = pd.DataFrame(columns = ["SMILES"], data = smi_lis)
    #Generate DataFrame
    df = get_prop(df, drop_dups = drop_dups)
    #Calculate properties
    return df

def get_prop(df : pd.DataFrame, smi_col = "SMILES", drop_dups = False): 
    """
    Generates the following representations and physiochemical properties 
    for a given DataFrame of molecules, drops duplicates if indicated

    * SMARTS 
    * InChIKey 
    * SA Score 
    * LogP 
    * QED 
    * Molecular weight 
    * SELFIES

    Also applies filters on the molecules and states if they have passed the filters or not 

    Parameters: 

    df (pd.DataFrame): DataFrame containing the molecules, must include a SMILES column

    smi_col (str): The name of the column containing the SMILES string, defaults to "SMILES"

    drop_dups (bool): Indicates whether or not duplicates should be dropped, defaults to False

    current_dir (str) : Deaults to ".", file path for the current directory

    Returns: 

    pd.DataFrame : New DataFrame containing the calcaulted properties

    """
    if "Molecule" not in df.columns:
        PandasTools.AddMoleculeColumnToFrame(df, smilesCol = smi_col, molCol = "Molecule")
    #Add a column with the RDKit mol files in 
    df["SMARTS"] = [Chem.MolToSmarts(mol) for mol in df["Molecule"]]
    #Add a SMARTS column to the DataFrame
    df["InChIKey"] = [Chem.MolToInchiKey(mol) for mol in df["Molecule"]]
    #Add an InChIKey column 
    if drop_dups == True: 
        df.drop_duplicates(subset = "InChIKey")
    #Drop duplicates by InChIKey
    df = check_filters(df)
    #Apply filters
    self_lis = []
    for smi in df[smi_col]: 
        try:
            self_lis.append(sf.encoder(smi))
        except sf.EncoderError:
            self_lis.append(None)
    #Generate SELFIES allowing for encoder errors
    df["SELFIES"] = self_lis
    df['SA_Score'] = [sascorer.calculateScore(i) for i in df['Molecule']]
    #Calculate the synthetic accessibility score
    prop_lis = [Chem.QED.properties(i) for i in df['Molecule']]
    df['LogP'] = [i[1] for i in prop_lis]
    df['QED'] = [Chem.QED.qed(mol, prop) for mol, prop in zip(df['Molecule'], prop_lis)]
    #Calculate LogP and QED
    df["Molecular weight"]  = [round(Chem.Descriptors.ExactMolWt(mol), 2) for mol in df["Molecule"]]
    #Calculate molecular weight 
    return df

def check_filters(df : pd.DataFrame, smarts_name = "SMARTS"): 
    """ 
    Runs rd_filters checks for the molecules in the DataFrame.
    NOTE: rd_filters must be saved to your working directory and installed see https://github.com/PatWalters/rd_filters 

    Parameters: 

    df (DataFrame): DataFrame containing the molecules, must include a SMARTS

    smarts_name (str): The name of the column containing the SMARTS, defaults to "SMARTS"

    Returns: 

    df (DataFrame): DataFrame with the additional "Passed filters" column
    """
    with open("dataset_smarts.smi", 'w') as f: 
        for count, smarts in enumerate(df[smarts_name]):
            f.write(f"{smarts}  MOL{count}\n")
    f.close()
    #Write the SMARTS to a .txt
    bash_command = 'rd_filters filter --in dataset_smarts.smi --prefix filtered --alerts rd_filters/rd_filters/data/alert_collection.csv --rules rd_filters/rd_filters/data/rules.json'
    os.system(bash_command)
    #Run the filtering
    filter_df = pd.read_csv("filtered.csv")
    df["Passed filter"] = [True if res == "OK" else False for res in filter_df["FILTER"]]
    df["LogP"] = filter_df["LogP"]
    df["TPSA"] = filter_df["TPSA"]
    df["Molecular weight"] = filter_df["MW"]
    return df

def dim_reduce(df : pd.DataFrame, run_pca = True, run_tsne = True):
    """ 
    Carries out dimenstionality reduction on the molecular fingerprints 
    allowing for visualisation of the chemical space.
    
    Parameters:
    
    df (DataFrame): Dataframe for the dataset of molecules

    run_pca (bool): Whether PCA dimensionality reduction should be carried out, defaults to True 

    run_tsne (bool): Whether tSNE dimensionality reduction should be carried out, defaults to True 

    Returns: 

    df (DataFrame): The original dataframe with added columns for the dimensionality reduction results

    """
    df['FPs'] = [mol2fp(mol) for mol in df['Molecule']]
    fp = np.array(df['FPs'].tolist())
    #Generate Morgan Fingerprints

    if run_pca == True: 
        #PCA 
        pca = PCA(n_components = 2)
        pca_val = pca.fit(fp) 
        pca_fin = pca.transform(fp)
        pca_fin = pd.DataFrame(pca_fin)
        df['PCA 1'] = pca_fin[0]
        df['PCA 2'] = pca_fin[1]

        if run_tsne == True: 
            #tSNE 
            tsne = TSNE(n_components = 2, init = 'random').fit_transform(fp)
            tsne = pd.DataFrame(tsne)
            df['t-SNE 1'] = tsne[0]
            df['t-SNE 2'] = tsne[1]
    
    elif run_tsne == True:
        #tSNE 
        tsne = TSNE(n_components = 2, init = 'random').fit_transform(fp)
        tsne = pd.DataFrame(tsne)
        df['t-SNE 1'] = tsne[0]
        df['t-SNE 2'] = tsne[1]
    
    return df
        



def clean_data(df : pd.DataFrame, smi_name = "SMILES", drop_dups = True):
    """ 
    Opens the .csv file and cleans the data by removing duplicates, 
    removing null entries and removing SMILES with a wild card.

    Parameters: 

    df (pd.DataFrame) : The DataFrame for the dataset 

    smi_name (str): The name of the column which the SMILES are stored in, defaults to "SMILES"
    
    drop_dups (bool) : Whether or not duplicates should be dropped, defaults to True
    Returns: 

    no_dups (DataFrame): A dataframe of the cleaned dataset

    """
    df_smi = df[(df[smi_name].notnull())]
    df_smi = df_smi[df_smi[smi_name].str.contains("\*") == False]
    #Drop entries without a SMILES string and SMILES with a wild card
    PandasTools.AddMoleculeColumnToFrame(frame = df_smi, smilesCol = smi_name, molCol = 'Molecule')
    df_smi['InChIKey'] = [AllChem.MolToInchiKey(mol) for mol in df_smi['Molecule']]
    #Add InChIKey and molecule column 
    if drop_dups == True: 
        df_smi = df_smi.drop_duplicates(subset = ['InChIKey'])
    #Drop duplicates
    df_smi['SMARTS'] = [AllChem.MolToSmarts(mol) for mol in df_smi['Molecule']]
    self_lis = []
    for mol in df_smi[smi_name]: 
        try: 
            self_lis.append(sf.encoder(mol))
        except sf.EncoderError: 
            self_lis.append("None")
    df_smi["SELFIES"] = self_lis
    #Add SMARTS and SELFIES
    print(f"{len(df) - len(df_smi)} entires have been removed")
    return df_smi

def mol2fp(mol : Chem.rdchem.Mol, radius = 2, bitlength = 2048): 
    """ 
    Generates Morgan Fingerprints as a bit vector. 

    Parameters: 

    mol (Mol): rdkit mol file for the molecule of interest

    radius (int): The radius of the fingerprint, defaults to 2

    bitlength (int): The bit length of the fingerprint, defaults to 2048

    Returns: 

    ar (ndarray): An array of the molecular fingerprint for the molecules
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius = radius, nBits = bitlength)
    #Generate the fingerprints
    ar = np.zeros((1,), dtype = np.int8)
    DataStructs.ConvertToNumpyArray(fp, ar)
    #Create an array from the fingerprints
    return ar
    #Return the array

def cluster(cluster_size : int, nbits : int, fp : str, output : str, txt_name : str, length : int): 
    """
    Cluster the molecules using their fingerprints and the K-means implementation.

    Parameters:   

    cluster_size (int): The number of clusters the data is clustered into 
    
    nbits (int): The bit length of the molecular fingerprints
    
    output (str): The fingerprint type, with this implementation it can be either 
        "morgan2", "morgan3", "ap", "rdkit5"

    file (str): File path to where the .csv should be saved to 

    txt_name (str): A string giving the file path to the .smi file containing the SMILES for the dataset

    length (int): The length of the dataset to be clustered
    
    """
    start_time = time.time()
    bash_command = f"python kmeans.py all --in {txt_name} --clusters {cluster_size} --out {output} --fp_type {fp} --dim {nbits} --sample {length}"
    #Define te bash command for the clustering using the chosen parameters 
    print(bash_command)
    os.system(bash_command)
    print(f"It took {time.time() - start_time} seconds to complete the clustering for {output}")


class Plots: 
    def __init__(self,  atoms_col : str = "Number of atoms", selfies_col : str = "SELFIES length", smiles_col : str = "SMILES length", atom_len : str = "darkturquoise", self_len : str = "deepskyblue", smi_len : str = "dodgerblue"): 
        self.atoms_col = atoms_col 
        self.selfies_col = selfies_col 
        self.smiles_col = smiles_col
        self.atom_len = atom_len 
        self.self_len = self_len 
        self.smi_len = smi_len

    @classmethod
    def plot_lengths(cls, df : pd.DataFrame, dataset : str, atoms_col : str = "Number of atoms", selfies_col : str = "SELFIES length", smiles_col : str = "SMILES length", atom_len : str = "darkturquoise", self_len : str = "deepskyblue", smi_len : str = "dodgerblue", file_name : str = "len.png"): 
        """
        Plots the lenght of SELFIES and SMILES against the number of atoms in the molecule 

        Parameters : 

        df (pd.DataFrame) : DataFrame containing the different lengths 

        dataset (str) : The name of the dataset, used in the title of the plot

        atoms_col (str) : Defaults to "Number of atoms", the name of the column containing the number of atoms in the molecule 

        selfies_col (str) : Defaults to "SELFIES length", the name of the column containing the SELFIES length for the molecule

        smiles_col (str) : Defaults to "SMILES length", the name of the column containing the SMILES length for the molecule 

        atom_len (str) : Defaults to "DarkTurquoise", the CSS colour name for the atom length within the plot 

        self_len (str) : Defaults to "DeepSkyBlue", the CSS colour name for the SELFIES length within the plot 

        smi_len (str) : Defaults to "DodgerBlue", the CSS colour name for the SMILES length

        file_name (str) : Defaults to "len.png", name for the plot to be saved to 
        
        """
        cls.df = df
        cls.dataset = dataset
        cls.atoms_col = atoms_col 
        cls.selfies_col = selfies_col
        cls.smiles_col = smiles_col 
        cls.atom_len = atom_len 
        cls.self_len = self_len 
        cls.smi_len = smi_len 
        cls.file_name = file_name

        count_lis = [mol for mol in df[atoms_col]] + [selfie for selfie in df[selfies_col]] + [smile for smile in df[smiles_col]]
        df_len = len(df)
        col_lis = ["Number of atoms" for x in range(df_len)] + ["SELFIES length" for x in range(df_len)] + ["SMILES length" for x in range(df_len)]
        df_1 = pd.DataFrame({"Size" : count_lis, "Type" : col_lis})
        #Define the DataFrame
        title = f"The distribution of molecule size, the SMILES <br>and SELFIES length for the {dataset}"
        #Define the title
        fig = px.histogram(df_1, x = "Size", title = title, color = "Type", marginal = "box", barmode = "group", color_discrete_sequence = [atom_len, self_len, smi_len])
        #Define the plot 
        fig.write_image(file_name, scale = 10)
    
    @classmethod 
    def cluster_distribution(cls, df : pd.DataFrame, cluster_details : str, colour : str, nbins : int = 50): 
        """
        Plot a histogram for the cluster population distribution 

        Parameters : 

        df (pd.DataFrame) : DataFrame with the cluster information in 

        cluster_details (str) : The details for the clustering to be displayed on the title 

        colour (str) : CSS colour for the plot 

        nbins (int) : Defaults to 50, the 
        """
        cls.df = df 
        cls.cluster_details = cluster_details
        cls.colour = colour
        cls.nbins = nbins 
        
        title = f"Cluster distribution for {cluster_details}"
        #Define the title 
        fig = px.histogram(df, x = "Cluster", nbins = nbins, title = title, color_discrete_sequence = [colour])
        #Define the figure 
        fig.show()
    
    @classmethod
    def prop_histogram(cls, df : pd.DataFrame, dataset : str = "dataset", file_name : str = "prop_histogram.png"): 
        """
        Plot a histogram to see the distribution of a given property in the dataset

        Parameters : 

        df (pd.DataFrame) : DataFrame containing the dataset

        dataset_name (str) : Defaults to 'the dataset', name of the dataset to be used in the title

        file_name (str) : Defaults to 'prop_histogram.png', file name for the plot to be saved to
        """
        cls.df = df
        cls.dataset = dataset
        cls.file_name = file_name
        fig = make_subplots(rows = 2, cols = 2)
        #Define the subplots

        fig.add_trace(go.Histogram(x = df["LogP"], name = "LogP", marker_color = "darkturquoise"), row = 1, col = 1)
        fig["layout"]["xaxis1"].update(title_text = "LogP")
        fig["layout"]["yaxis1"].update(title_text = "Frequency")

        fig.add_trace(go.Histogram(x = df["Molecular weight"], name = "Molecular weight", marker_color = "deepskyblue"), row = 1, col = 2)
        fig["layout"]["xaxis2"].update(title_text = "Molecular weight")
        fig["layout"]["yaxis2"].update(title_text = "Frequency")

        fig.add_trace(go.Histogram(x = df["SA_Score"], name = "SA Score", marker_color = "dodgerblue"), row = 2, col = 1)
        fig["layout"]["xaxis3"].update(title_text = "SA Score")
        fig["layout"]["yaxis3"].update(title_text = "Frequency")

        fig.add_trace(go.Histogram(x = df["QED"], name = "QED", marker_color = "royalblue"), row = 2, col = 2)
        fig["layout"]["xaxis4"].update(title_text = "QED")
        fig["layout"]["yaxis4"].update(title_text = "Frequency")

        fig.update_layout(height = 600, width = 800, title_text = f"Property distributions for the {dataset}")
        fig.write_image(file_name, scale = 10)

    @classmethod
    def scatter_plot_discrete(cls, df : pd.DataFrame, colour : dict, prop : str, method : str = "PCA", dataset : str = "dataset", file_name : str = "scatter.png"):
        """
        Plots a scatter plot for a non continuous parameter

        Parameters : 

        df (pd.DataFrame) : DataFrame to be plotted

        colour (dict) : Dictionary detailing the colours for each of the classifications 

        prop (str) : Property to be visualized

        method (str) : Defaults to PCA, the dimensionality reduction method to be used for the plot

        dataset (str) : Defaults to 'dataset', name of the dataset being visualised 

        file_name (str) : Defaults to 'scatter.png', name for the plot to be saved to 
        """
        cls.df = df 
        cls.colour = colour 
        cls.method = method

        title = f"{dataset} visualised by {prop} using the {method} reduced molecular fingerprints"
        #Define the title
        fig = px.scatter(df, x = f'{method} 1', y = f'{method} 2', color = prop, color_discrete_map = colour, title = title, width = 800, height = 800)
        #Define the plot 
        fig.write_image(file_name, scale = 10)

        def scatter_plot_cont(cls, df : pd.DataFrame, method : str, prop : str, prop_title : str, colour : str): 
            """
            Produces a scatterplot of the chemical space 
            Using the reduced fingerprints.

            Parameters: 

            method (str): the dimenstionality reduction method which will be used to visualise the chemical space

            prop (str): The property which will be visualised

            df (DataFrame): Dataframe for the dataset of molecules
            
            colour (str): colour set to be used for visualisation 

            Returns: 

            fig (str)

            """
            cls.df = df 
            cls.method = method 
            cls.prop = prop 
            cls.prop_title = prop_title 
            cls.colour = colour 
            fig = px.scatter(df, x = f'{method} 1', y = f'{method} 2', color = prop, title = f'Dataset visualised by the {prop_title} of the molecule <br>by {method} reduced Morgan Fingerprints'
                        , color_discrete_sequence = colour)
            fig.update_layout(height = 700, width = 795)
            return fig 
        
        

    