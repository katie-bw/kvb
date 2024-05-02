import argparse
import numpy as np 
import os

import csv
import pandas as pd 
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, PandasTools, Draw, Mol,MACCSkeys, QED, Descriptors, RDConfig, rdCoordGen 
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint, GetBTFingerprint
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem.Fingerprints import FingerprintMols
import gc
import selfies as sf
import matplotlib.pyplot as plt 
from operator import itemgetter
import plotly.graph_objects as go
import random
import warnings
import os
import collections
import sys
import scipy
import time 
import json
sys.path.append(os.path.join(os.environ['CONDA_PREFIX'],'share','RDKit','Contrib'))
from rdkit.Contrib.SA_Score import sascorer
#Import SAScore file
#Import packages


class LECSSVariables:
    """
    Define the variables required
    """
    def __init__(self):
        self.alphabet = list(sf.get_semantic_robust_alphabet()) 
        #Generate a list of the characters in the SELFIES alphabet as a global variable 

        self.ring_char = ['[Ring1]', '[Ring2]', '[Ring3]', '[=Ring1]', '[=Ring2]', '[=Ring3]']
        self.branch_char = ['[Branch1]', '[Branch2]', '[Branch3]', '[=Branch1]', '[=Branch2]', '[=Branch3]', '[#Branch1]', '[#Branch2]', '[#Branch3]']
        self.dropped_exmol = ['[P]', '[#P]', '[=P]', '[B]', '[#B]', '[=B]', '[#N+1]','[N+1]','[B+1]','[#P-1]','[=S-1]','[=P+1]','[S-1]','[#B-1]','[=N+1]','[C-1]','[#P+1]','[B-1]','[#S-1]','[#O+1]','[P+1]','[#C-1]','[=B+1]','[=C-1]','[=B-1]','[=O+1]','[=S+1]','[=C+1]','[S+1]','[N-1]','[#C+1]','[=N-1]','[=P-1]','[P-1]','[C+1]','[#S+1]','[O+1]']
        self.valency_issues = ["[#C]", "[#N]", "[#S]", "[#P]", "[=O]"]
        #Define lists for the ring, branch characters and those removed in the exmol basic alphabet

        self.alphabet_tot = list(sf.get_semantic_robust_alphabet())
        #Define a list of the SELFIES alphabet

        self.no_rings_alphabet = list(set(self.alphabet) - set(self.ring_char))
        self.no_branches_alphabet = list(set(self.alphabet) - set(self.branch_char))
        self.no_both_alphabet = list(set(self.no_rings_alphabet) - set(self.branch_char))
        self.exmol_alphabet_no = list(set(self.no_both_alphabet) - set(self.dropped_exmol))
        self.exmol_alphabet = list(set(self.alphabet) - set(self.dropped_exmol))
        self.replace_alphabet = list(set(self.no_both_alphabet) - set(self.valency_issues))
        #Define the alphabets for the constraints

        self.index_dict = {'[C]' : '0', '[Ring1]' : '1', '[Ring2]' : '2',
                    '[Branch1]' : '3', '[=Branch1]' : '4', '[#Branch1]' : '5', 
                    '[Branch2]' : '6', '[=Branch2]' : '7', '[#Branch2]' : '8', 
                    '[O]' : '9', '[N]' : '10', '[=N]' : '11', '[=C]' : '12',
                    '[#C]' : '13', '[S]' : '14', '[P]' : '15', 'Else' : '0'}
        self.index_lis = list(self.index_dict.keys())

        self.no_rings_alphabet = list(set(self.alphabet) - set(self.ring_char))
        self.no_branches_alphabet = list(set(self.alphabet) - set(self.branch_char))
        self.no_both_alphabet = list(set(self.no_rings_alphabet) - set(self.branch_char))
        self.exmol_alphabet_no = list(set(self.no_both_alphabet) - set(self.dropped_exmol))
        self.exmol_alphabet = list(set(self.alphabet) - set(self.dropped_exmol))
        #Define the alphabets for the constraints

        self.alph_dict = {'1' : self.alphabet, 
                        '2' : self.no_rings_alphabet, 
                        '3' : self.no_branches_alphabet, 
                        '4' : self.no_both_alphabet, 
                        '5' : self.exmol_alphabet_no, 
                        '6' : self.exmol_alphabet}
        #Define the alphabet dictionary 

        self.column_names = ["SELFIES", "Mutate type", "Character removed", "Character added", "Mutation step", "SMILES", "Molecule", "SMARTS", "InChIKey",
                             "Tanimoto", "Dice", "Cosine", "Sokal", "SA Score", "LogP", "QED", "Molecular Weight",
                             "Change in SA Score", "Change in LogP", "Change in QED", "Change in Molecular Weight",
                             "Original molecule"]


    """
    Methods for the molecule optimizations 
    """
def get_json(file_name : str = "Optimization.json"): 
    """
    Generates a .json file in order to save the details of the optimization conditions (for single parameter optimization)

    Parameters: 

    file_name (str): The name for the .json file to be saved to 

    Returns
        
    dict: The dictionary detailing the optimization conditions
    """
    json_dict = {}
    alph = LECSSVariables()
    alph = alph.alphabet
    valid = False
    while valid == False:
        org_smiles = input("Please enter the SMILES for the molecule you wish to optimise: \n")
        if Chem.MolFromSmiles(org_smiles) is None: 
            print(f"{org_smiles} is not a valid smiles")
            print("Enter a valid SMILES")
        else: 
            valid = True 
            json_dict["Original SMILES"] = org_smiles
    #Starting SMILES
    json_dict["Pool size"] = input("Please enter the pool size to be used when generating the new molecules (reccomended to be 50): \n")
    json_dict["Mutation steps"] = input("Please enter the number of mutatation steps to be used when generating the new molecules (reccomended to be 1): \n")
    json_dict["Alphabet"] = input("Please choose which alphabet should be used for the optimization: \n1 - Standard SELFIES alphabet\n2 - Standard alphabet with rings removed\n3 - Standard alphabet with branches removed\n4 - Standard alphabet with rings and branches removed\n5 - Exmol alphabet with rings and branches removed\n6 - Exmol alphabet")
    json_dict["Restrictions"] = input("Please choose which restrictions should be used for the optimization: \n1 - None\n2 - Simple heteroatoms\n3 - Rings\n4 - Branches\n5 - Custom")
    #Details of the mutations
    if json_dict["Restrictions"] == "5": 
        finish = False
        char_list = []
        while finish == False: 
            char = input("Please enter the character you wish to restrict: ")
            if char not in char_list: 
                print(f"{char} is not a valid SELFIES characters please enter a valid character")
            else: 
                char_list.append(char)
            done = input(f"You currently have the following characters denoted for restrictions:\n{char_list}\nDo you wish to add anymore? Y/N")
            if done == "Y": 
                finish = True
    #Get custom restrictions
    json_dict["Limit"] = int(input("What is the number of optimization steps you want to be taken?"))
    prop_dict = {"1" : "QED", "2" : "SA Score", "3" : "LogP", "4" : "Molecular weight", "5" : "Tanimoto"}
    prop = input("Which property would you like to optimize for?\n1 - QED\n2 - SA Score\n3 - LogP\n4 - Molecular weight\n5 - Tanimoto")
    json_dict["Property"] = prop_dict[prop]
    #Which property to be isolated for 
    range_ = input("Is there a range you want your molecule to fall into? Y/N")
    if range_ == "Y": 
        lower = input("\nPlease enter the lower barrier of the range:")
        upper = input("\nPlease enter the upper barrier of the range:")
        json_dict["Range"] = f"{lower} - {upper}"
    else: 
        json_dict["Range"] = None
    #Range for the optimization
    direction = input("\nWhich direction do you want to optimize in? +/-")
    json_dict["Direction"] = direction == "+" 
    #Direction for the optimization
    if prop == "5": 
        valid = False
        while valid == False:
            smiles = input("Please enter the SMILES string for the similarity measurement:\n")
            if Chem.MolFromSmiles(smiles) is None: 
                print(f"{smiles} is not a valid smiles")
                print("Enter a valid SMILES")
            else: 
                valid = True 
                json_dict["SMILES"] = smiles
    else: 
        json_dict["SMILES"] = None
    json_object = json.dumps(json_dict, indent=4)
    # Writing to sample.json
    with open(file_name, "w") as outfile:
        outfile.write(json_object)
    with open(file_name, "r") as json_file:
        data = json.load(json_file)
    return data
    

def define_json(org_smiles : str, prop : str, direction : bool, range_ : bool = False, upper : float = None, lower : float = None, alphabet : int = 1, limit : int = 50, pool_size : int = 50, mutation_steps : int = 1, restriction : int = 1, smi : str = None, restriction_lis : list = None, file_name : str = "Optimization.json"): 
    """
    Generates a .json file in order to save the details of the optimization conditions (for single parameter optimization)

    Parameters: 

    org_smiles (str) : SMILES string for the molecule being optimized 

    prop (str) : Name of the property being optimized for, must be one of: QED, SA Score, LogP, Molecular Weight, Tanimoto

    direction (bool) : Direction for the optimization

    range_ (bool) : Defaults to False, whether there is a desired range for the property

    upper (float) : Defaults to None, upper bound for the range 

    lower (float) : Defaults to None, lower bound for the range

    alphabet (int) : Defaults to 1, number for the alphabet being used for the optimization

    limit (int) : Defaults to 50, maximum number of steps taken

    pool_size (int) : Defaults to 50, the pool size being used for the optimization

    mutation_steps (int) : Defaults to 1, the number of mutation steps being used for the optimization 

    restriction (int) : Defaults to 1, the number for the restriction being used

    smi (str) : Defaults to None, the SMILES string for the Tanimoto property 

    restriction_lis (list) : Defaults to None, list of characters for a custom restriction

    file_name (str): Name for the .json file to be saved to 

    Returns
        
    dict: The dictionary detailing the optimization conditions
    """
    json_dict = {}
    alph = LECSSVariables()
    alph = alph.alphabet
    #Define dictionary and alphabet
    if Chem.MolFromSmiles(org_smiles) is None: 
        print(f"{org_smiles} is not a valid smiles")
        print("try again with a valid SMILES")
        return
    else: 
        json_dict["Original SMILES"] = org_smiles
    #Starting SMILES
    json_dict["Pool size"] = pool_size
    json_dict["Mutation steps"] = mutation_steps
    json_dict["Alphabet"] = alphabet
    json_dict["Restrictions"] = restriction 
    #Details of the mutations
    if json_dict["Restrictions"] == "5": 
        valid = True 
        for char in restriction_lis: 
        #Iterate over the characters 
            valid = valid and (char in alph)
        if valid == True: 
            print(f"This is not a list of valid SELFIES characters {restriction_lis}")
            print("Try again with all valid SELFIES characters")
            return
        else: 
            json_dict["Restriction list"] = restriction_lis  
    else: 
        json_dict["Restriction list"] = None       
    #Get custom restrictions
    json_dict["Limit"] = int(limit)
    json_dict["Property"] = prop
    #Which property to be isolated for 
    if range_ == True: 
        json_dict["Range"] = f"{lower} - {upper}"
    else: 
        json_dict["Range"] = None
    #Range for the optimization
    json_dict["Direction"] = direction 
    #Direction for the optimization
    if prop == "5": 
        if Chem.MolFromSmiles(smi) is None: 
            print(f"{smi} is not a valid smiles")
            print("try again with a valid SMILES")
            return
        else: 
            json_dict["SMILES"] = smi
    else: 
        json_dict["SMILES"] = None
    json_object = json.dumps(json_dict, indent=4)
    # Writing to sample.json
    with open(f"{file_name}.json", "w") as outfile:
        outfile.write(json_object)
    with open(f"{file_name}.json", "r") as json_file:
        data = json.load(json_file)
    return data

def get_multi_opt_json(file_name : str = "Optimization.json"): 
    """
    Generates a .json file in order to save the details of the optimization conditions (for multi parameter optimization)

    Parameters: 

    file_name (str): The name for the .json file to be saved to 

    Returns
        
    dict: The dictionary detailing the optimization conditions
    """
    json_dict = {}
    count = 0 
    for prop in ["QED", "SA Score", "LogP", "Molecular weight"]:
        opt = input(f"Would you like to optimize for {prop}? Y/N")
        json_dict[prop] = {"Optimization" : opt == "Y"}
        if opt == "Y": 
            count = count + 1
            range_ = input("Is there a range you want your molecule to fall into? Y/N")
            if range_ == "Y": 
                lower = input("Please enter the lower barrier of the range:")
                upper = input("Please enter the upper barrier of the range:")
                json_dict[prop]["Range"] = f"{lower} - {upper}"
            else: 
                json_dict[prop]["Range"] = None
            direction = input("Which direction do you want to optimize in? +/-")
            json_dict[prop]["Direction"] = direction == "+"
        else: 
            json_dict[prop]["Range"] = None
            json_dict[prop]["Direction"] = None
    tanimoto = input("Would you like to optimize for similarity to a target molecule? Y/N")
    json_dict["Tanimoto"] = {"Optimization" : tanimoto == "Y"}
    if tanimoto == "Y": 
        count = count + 1
        json_dict["Tanimoto"]["SMILES"] = input("Please enter the SMILES string for the molecule:")
        range_ =  input("Is there a range you want the molecule to fall into? Y/N")
        if range_ == "Y": 
            lower = input("Please enter the lower barrier of the range:")
            upper = input("Please enter the upper barrier of the range:")
            json_dict["Tanimoto"]["Range"] = f"{lower} - {upper}"
        else: 
            json_dict["Tanimoto"]["Range"] = None
        direction = input("Which direction do you want to optimize in? +/-")
        json_dict["Tanimoto"]["Direction"] = direction == "+"
    else: 
        json_dict["Tanimoto"]["Range"] = None
        json_dict["Tanimoto"]["SMILES"] = None
        json_dict["Tanimoto"]["Direction"] = None
    if count > 1: 
        print("This is a multi-objective optimization problem as multiple parameters have been selected")
    json_object = json.dumps(json_dict, indent=4)
    
    # Writing to sample.json
    with open(file_name, "w") as outfile:
        outfile.write(json_object)
    with open(file_name, "r") as json_file:
        data = json.load(json_file)
    return data

def get_pairs_dict(df : pd.DataFrame, pair_id : int, id_col = "Pair ID"):
    """
    Based on the pair ID the relevant dictionary is generated in order to run the optimization 

    Parameters: 

    df (pd.DataFrame): DataFrame of the molecule pairs

    pair_id (int): ID for the pair

    id_col (str): Defaults to "Pair ID", the name of the ID column

    Returns: 

    dict: Dictionary providing the details fo the pair to be optimized 

    {"start" : {"smi" : smi, "LogP" : logp, "QED" : qed, "SA Score" : sa_score}, 
                        "target" : {"smi" : smi, "LogP" : logp, "QED" : qed, "SA Score" : sa_score}}
    """
    pair_df = df[df[id_col] == pair_id]
    #Get the row for the molecule pair 
    return {"start" : {"smi" : pair_df["Mol1"][0], "LogP" : pair_df["LogP 1"][0], "QED" : pair_df["QED 1"][0], "SA Score" : pair_df["SA Score 1"][0]}, 
                    "target" : {"smi" : pair_df["Mol2"][0], "LogP" : pair_df["LogP 2"][0], "QED" : pair_df["QED 2"][0], "SA Score" : pair_df["SA Score 2"][0]}}

def get_target_delta(df : pd.DataFrame, target_dict : dict): 
    """
    Calculates the difference in given physiochemical properties to the target molecule

    Parameters: 

    df (pd.DataFrame): DataFrame of the generated molecules 

    target_dict (dict): Details of the target

    Returns: 

    pd.DataFrame: Updated DataFrame with the relevant columns added
    """
    df["LogP to target"] = [target_dict["LogP"] - logp for logp in df["LogP"]]
    df["QED to target"] = [target_dict["QED"] - qed for qed in df["QED"]]
    df["SA Score to target"] = [target_dict["SA Score"] - sa for sa in df["SA Score"]] 
    fp_lis = [get_fingerprint(d) for d in df["Molecule"]]
    df["Tanimoto to target"] = [DataStructs.TanimotoSimilarity(target_dict["fp"], fp) for fp in fp_lis]
    #Calculate the relevant properties 
    return df 
    
def run_steps_direction(mutate_details : dict, org_smi : str, target_dict : dict, limit : int, prop : str): 
    """
    Run the optimization steps 

    Parameters: 

    mutate_details (dict): Dictionary detailing the conditions for the mutation; 
                            {"num mutations" : num_mutations, pool size" : pool_size, "alphabet" : alph_num, "restricted chars" : restricted_chars} 

    org_smi (str): The starting SMILES string

    target_dict (dict): The dictionary detailing the target and its properties

    limit (int): The maximum number of steps to be taken before terminating the toptmixation 

    prop (str): The property to be optimized for 

    Returns: 

    pd.DataFrame: The DataFrame providing the generated molecules
    """
    step_smi = org_smi
    complete = False
    df_lis = []
    #Set up for loop
    while complete == False: 
        #End run when completed
        for x in range(limit): 
        #Iterate over a set number of times
            df = run_bulk_mutation(step_smi, num_mutations = mutate_details["num mutations"], 
                                        pool_size = mutate_details["pool size"], alphabet_num = mutate_details["alph_num"], 
                                        restricted_chars = mutate_details["restricted chars"])
            #Run the mutations to generate a small chemical space
            df.drop_duplicates(subset = "InChIKey")
            #Drop any duplicate molecules!
            df["Step"] = [f"Step {x + 1}" for i in range(len(df))]
            #Make note of the step of the optimization process
            hit_lis = [target_dict["InChIKey"] == inchikey for inchikey in df["InChIKey"]]
            df["Hit"] = hit_lis
            #Check if any hits are present 
            df = get_target_delta(df, target_dict)
            #Calculate the difference in the relevant properties to the target
            if prop == "Tanimoto": 
                _prop = df[f'{prop} to target'].max()
                df["Optimal molecule"] = [x == df[f'{prop} to target'].max() for x in df[f"{prop} to target"]]
            else: 
                _prop = df[f"{prop} to target"].min()
                df["Optimal molecule"] = [x == df[f"{prop} to target"].min() for x in df[f"{prop} to target"]]
            #Identify which molecule is most similar to the target 
            if hit_lis.count(True) > 0: 
                complete = True
            df_lis.append(df)
            print(f"Step {x + 1} complete, {prop}: {_prop}")
            #Append the DataFrame
            
    df = pd.concat(df_lis, axis = 0)
    print(f"Optimization complete \nafter {x + 1}/limit steps taken") 
    return df

def get_mutate_details_json(file_path : str): 
    """
    Get the mutate details from the .json file

    Parameters : 

    file_path (str) : File path for the .json file

    Returns : 

    mutate_details (dict) : Details for the mutation
    """
    with open(file_path, "r") as json_file:
        optimization_details = json.load(json_file)
    #Load in the data
    mutate_details = {"num mutations" : int(optimization_details["Mutation steps"]), "pool size" : int(optimization_details["Pool size"]), 
                      "alphabet" : int(optimization_details["Alphabet"])}
    if optimization_details["Restrictions"] != "5": 
        restrict_dict = {"1" : None, 
                        "2" : ["[O]", "[N]"],
                        "3" : ['[Ring1]', '[Ring2]', '[Ring3]', '[=Ring1]', '[=Ring2]', '[=Ring3]'], #Ring characters
                        "4" :['[Branch1]', '[Branch2]', '[Branch3]', '[=Branch1]', '[=Branch2]', '[=Branch3]', '[#Branch1]', '[#Branch2]', '[#Branch3]']}
        mutate_details["restricted chars"] = restrict_dict[str(optimization_details["Restrictions"])]
    else: 
        mutate_details["restricted chars"] = optimization_details["Restriction list"]
    #Get the mutation details 
    return mutate_details
    
def run_optimization(file_path : str, gen_image : bool = True, image: str = "steps") -> dict: 
    """
    Run the optimization steps 

    Parameters: 

    file_path (str) : File path for the .json file containig the details of the optimization 

    gen_image (bool) : Defaults to True, whether a scatter plot should be generated showing the property across the steps

    image (str) : Defaults to "steps", the name for the image file name

    Returns: 

    dict : Dictionary containing two DataFrames, one for all the molecules generated and one for the molecules which were optimal at each step
    """
    warnings.simplefilter(action = "ignore")
    with open(file_path, "r") as json_file:
        optimization_details = json.load(json_file)
    json_file.close()
    #Load in the optimization details
    mutate_details = get_mutate_details_json(optimization_details)
    #Get the mutation details for the optimization
    org_smi = optimization_details["Original SMILES"]
    prop = optimization_details["Property"]
    #Get details of the optimization
    step_smi = org_smi
    complete = False
    df_lis = []
    optimal_mols = []
    #Set up for loop
    for x in range(optimization_details["Limit"]): 
    #Iterate over a set number of times
        df = run_bulk_mutation(step_smi, num_mutations = mutate_details["num mutations"], 
                                    pool_size = mutate_details["pool size"], alphabet_num = mutate_details["alph_num"], 
                                    restricted_chars = mutate_details["restricted chars"], ref_smi = org_smi, print_time = False)
        #Run the mutations to generate a small chemical space
        df.drop_duplicates(subset = "InChIKey")
        #Drop any duplicate molecules!
        df["Step"] = [f"Step {x + 1}" for i in range(len(df))]
        #Make note of the step of the optimization process
        if optimization_details["Range"] != None: 
            range_limits = (optimization_details["Range"]).split()
            lower = float(range_limits[0])
            upper = float(range_limits[2])
            #Get the range limits 
            df["Within range"] = [(df[prop] > lower) & (df[prop] < upper)]
            #Get the generated molecules which fall within the defined range
        df["Optimal molecule"] = [[] for x in range(len(df))]
        df_lis.append(df)
        df_tot = pd.concat(df_lis, axis = 0)
        #Combine the DataFrame
        df_tot = get_optimal_mol(optimization_details, df_tot, x)
        opt = df_tot[df_tot["Optimal molecule"].str.len() != 0][["SMILES", "Optimal molecule"]]
        #Get a DataFrame for the molecules which have been or are the optimal molecules
        for idx, row in opt.iterrows(): 
            if row["Optimal molecule"][-1] == f"Step {x + 1}":
                step_smi = row["SMILES"]
    if gen_image: 
        display_steps(df = df_tot, prop = prop, file_name = image)
    #Plot the property over the optimization steps
    return {"Total" : df_tot, "Optimal" : opt}

def display_steps(df : pd.DataFrame, prop : str, file_name : str): 
    """
    Plots a histogram 

    Parameters : 

    df (pd.DataFrame) : DataFrame of the generated molecules 

    prop (str) : Property which the seed molecule has been optimized for

    file_name (str) : File name for the image to be saved to 
    
    """
    fig = go.Figure()
    #Initialize the figure
    fig.add_trace(go.Scatter(x = df["Step"], y = df[prop], mode = "markers"))
    #Add the scatter plot trace
    title = f"{prop} over the optimization steps"
    #Define the plot title
    ticks = (len(set(df["Step"]))/10)
    fig.update_layout(template = "simple_white", title = title, height = 600, width = 800)
    fig.update_xaxes(title_text = "Optimization step")
    fig.update_yaxes(title_text = prop)
    #Add the titles and format the plot
    fig.write_image(f"{file_name}.png")
    #Write the generated image to a .png
    
def get_optimal_mol(optimization_details : dict, df_tot : pd.DataFrame,  step : int):
    """
    Get the optimal molecule for the generated molecules 

    Parameters : 

    optimization_details (dict) : Dictionary detailing the optimization details 

    df_tot (pd.DataFrame) : Total DataFrame for the generated molecules 

    step (int) : The step of the optimization

    Returns : 

    pd.DataFrame : DataFrame detailing the optimal molecule for that step
    
    """
    prop = optimization_details["Property"]
    if optimization_details["Range"] != None:
    #Defined range
        within_range = df_tot[df_tot["Within range"] == True]
        if len(within_range) == 0: 
            poss_range = False
            print(f"None of the generated molecules fall within the defined range of {optimization_details['Range']} for prop\nSo the optimal molecule for change in {prop} will be chosen for the next step")
        else: 
            poss_range = True
    else: 
        poss_range = False
    #Will the optimal molecule be within the range?
    if optimization_details["Direction"] == True:
        if poss_range == True:
            _prop = within_range[f'Change in {prop}'].max()
            df_tot[df_tot[f'Change in {prop}'] == _prop]["Optimal molecule"].iloc[0] = df_tot[df_tot[f'Change in {prop}'] == _prop]["Optimal molecule"].iloc[0].append(f"Step {step + 1}")
        else: 
            _prop = df_tot[f'Change in {prop}'].max()
            df_tot[df_tot[f'Change in {prop}'] == _prop]["Optimal molecule"].iloc[0] = df_tot[df_tot[f'Change in {prop}'] == _prop]["Optimal molecule"].iloc[0].append(f"Step {step + 1}")
    else: 
        if poss_range == True:
            _prop = within_range[f'Change in {prop}'].min()
            df_tot.query(f"'Change in {prop}' == {_prop}")["Optimal molecule"].iloc[0] = df_tot.query(f"'Change in {prop}' == {_prop}")["Optimal molecule"].iloc[0].append(f"Step {step + 1}")
        else: 
            _prop = df_tot[f'Change in {prop}'].min()
            df_tot.query(f"'Change in {prop}' == {_prop}")["Optimal molecule"].iloc[0] = df_tot.query(f"'Change in {prop}' == {_prop}")["Optimal molecule"].iloc[0].append(f"Step {step + 1}") 
    _prop = df_tot[df_tot[f'Change in {prop}'] == _prop]
    prop_val = round(_prop[prop].iloc[0], 3)
    print(f"Step {step + 1} complete, {prop}: {prop_val}")
    return df_tot
    
def get_mutate_details_json(optimization_dict : dict): 
    """
    Gets the mutation details from the dictionary provided from the .json file 

    Parameters : 

    optimization_dict (dict) : Dictionary detailing the optimization details 

    returns : 

    dict : Dictionary detailing the mutation details specifically 
    
    """
    restricted_chars = {"1" : None, 
                        "2" : ["[O]", "[N]"],
                        "3" : ['[Ring1]', '[Ring2]', '[Ring3]', '[=Ring1]', '[=Ring2]', '[=Ring3]'], #Ring characters
                        "4" : ['[Branch1]', '[Branch2]', '[Branch3]', '[=Branch1]', '[=Branch2]', '[=Branch3]', '[#Branch1]', '[#Branch2]', '[#Branch3]']}
                        #Branch characters
    #Define the predefined restriction characters
    mutate_details = {"num mutations" : int(optimization_dict["Mutation steps"]), "pool size" : int(optimization_dict["Pool size"]),
                      "alph_num" : int(optimization_dict["Alphabet"])}
    if optimization_dict["Restrictions"] != "5": 
        mutate_details["restricted chars"] = restricted_chars[str(optimization_dict["Restrictions"])]
    #For a predefined restriction set
    else:
        mutate_details["restricted chars"] = optimization_dict["Restricted list"]  
    return mutate_details  
    

    
#Returns a dearomatized molecule 
def randomize_smiles(mol : Chem.rdchem.Mol, size : int):
    """
    Generates random, kekulized SMILES 
    This is used in the molecular generation in order to increase diversity 

    Parameters:

    mol (Mol): RDKit mol file for the molecule 

    size (int): The number of randomized SMILES to be returned

    Returns: 
    
    lis (list): A list of the randomized SMILES strings
    """
    if not mol:
        return None
    #If an empty RDKit mol file is generated return None
    Chem.Kekulize(mol)
    #Kekulize the molecule
    lis = []
    if size != 1:
        for i in range(size): 
            lis.append(Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False,  kekuleSmiles=True))	
    else: 
        lis = [Chem.MolToSmiles(mol)]
    return lis

    #Return the kekulized SMILES 
	
#Taken from the STONED workflow
class _FingerprintCalculator:
    def get_fingerprint(self, mol: Mol, fp_type: str):
        method_name = 'get_' + fp_type
        method = getattr(self, method_name)
        if method is None:
            raise Exception(f'{fp_type} is not a supported fingerprint type.')
        return method(mol)
    def get_AP(self, mol: Mol):
        return AllChem.GetAtomPairFingerprint(mol, maxLength=10)
    def get_PHCO(self, mol: Mol):
        return Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
    def get_BPF(self, mol: Mol):
        return GetBPFingerprint(mol)
    def get_BTF(self, mol: Mol):
        return GetBTFingerprint(mol)
    def get_PATH(self, mol: Mol):
        return AllChem.RDKFingerprint(mol)
    def get_ECFP4(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 2)
    def get_ECFP6(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 3)
    def get_FCFP4(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 2, useFeatures=True)
    def get_FCFP6(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 3, useFeatures=True)
    
#Returns fingerprint
def get_fingerprint(mol: Chem.rdchem.Mol, fp_type = "ECFP4"):
    """
    Obtains the molecular fingerprint for the molecule, using the _FingerprintCalculator class

    Parameters: 

    mol (Chem.rdchem.Mol) : The molecule used to generate the fingerprint 

    fp_type : The fingerprint type as defined in the class, defaults to "get_ECFP4"
    """
    return _FingerprintCalculator().get_fingerprint(mol = mol, fp_type = fp_type)


def get_delta(org_mols : pd.DataFrame, gen_mols : pd.DataFrame, org_ID : str, calc_props = False):
    """Gets the delta for a number of physiochemical properties: 

        * Molecular weight 
        * SA Score 
        * QED
        * LogP

        Will also direct to calculate properties for seed molecules if needed
        
    Parameters: 

    org_mols (pd.DataFrame): DataFrame of the seed molecules 

    gen_mols (pd.DataFrame): DataFrame of the generated molecules

    org_ID (str): Name of the column where the molecular identifier is stored

    calc_props (bool): Defaults to False, indicates whether or not the seed molecules 
                       need their properties calculated

    Returns: 

    pd.DataFrame : DataFrame for the generated molecules giving the delta for the given properties
    """
    if calc_props == True: 
        df = calc_props(df)
    #Calculate properties if required
    delta_mw = []
    delta_sa_score = []
    delta_qed = []
    delta_logp = []
    #Initialize lists
    for idx, row in gen_mols.iterrows():
        props = org_mols.loc[org_mols[org_ID] == row["Original molecule"]]
        delta_mw.append(row["Molecular Weight"] - props["Molecular weight"].values[0])
        delta_sa_score.append(row["SA Score"] - props["SA Score"].values[0])
        delta_qed.append(row["QED"] - props["QED"].values[0])
        delta_logp.append(row["LogP"] - props["LogP"].values[0])
        #Calculate the difference
        del props
        gc.collect()
    gen_mols["Change in Molecular Weight"] = delta_mw
    gen_mols["Change in SA Score"] = delta_sa_score
    gen_mols["Change in QED"] = delta_qed
    gen_mols["Change in LogP"] = delta_logp
    del delta_mw, delta_sa_score, delta_qed, delta_logp
    gc.collect()
    #Add columns to the DataFrame
    return gen_mols

def get_nearest_neighbour_sim(df : pd.DataFrame, smi_column = "SMILES", mol_column = "Molecule", calc_mol = False): 
    """
    Calculates the similarities to the nearest neighbour within the dataset using Tanimoto

    Parameters: 

    df (pd.DataFrame): DataFrame of the generated molecules

    smi_column (str): Defaults to "SMILES", the name of the column containing the SMILES strings 

    mol_column (str): Defaults to "Molecule", the name of the column containing the RDKit mol files 

    calc_mol (bool): Defaults to False, whether or not the RDKit mol files need to be calculated

    Returns: 

    pd.DataFrame: DataFrame for the generated molecules giving the nearest neighbour similarity
    """
    if calc_mol == True: 
        df[mol_column] = [Chem.MolFromSmiles(smi) for smi in df[smi_column]]
        #Generate the RDKit mol

    
    df["FP"]= [get_fingerprint(mol) for mol in df[mol_column]]
    #Calculate the fingerprint
    fp_lis = df["FP"].tolist()
    
    max_sim = []
    #List for similarity measurements

    for x, fp in enumerate(fp_lis): 
        fps = df["FP"].tolist()
        fps.pop(x)
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
        #Calculate the similarities
        max_sim.append(max(sims))
        #Add the maximum
    df["Nearest neighbour sim"] = max_sim
    #Append to the DataFrame

    df = df.drop(["Molecule", "FP"], axis = 1)
    return df


def check_filters(df : pd.DataFrame, smiles_col = "SMILES", smarts_col = "SMARTS", gen_smarts = False, save_props = True): 
    """
    Runs the rd_filters on a set of molecules

    Paramters:

    df (pd.DataFrame): The DataFrame of the molecules 

    smiles_col (str): Defaults to "SMILES", the name of the column which contains the SMILES strings of the molecules

    smarts_col (str): Defualts to "SMARTS", the name of the column which contains the SMARTS strings of the molecules

    gen_smarts (bool): Defaults to False, whether or not SMARTS for the molecule needs to be generated

    save_props (bool): Defaults to True, whether or not the properties calculated are saved

    Returns: 

    pd.DataFrame: The DataFrame of molecules indicating which passed the filters
    
    """
    if gen_smarts == True: 
        df[smarts_col] = [Chem.MolToSmarts(Chem.MolFromSmiles(smi)) for smi in df[smiles_col]]
        #Generate SMARTS for the molecules if needed

    with open('dataset.smi', 'w') as f: 
        for count, smarts in enumerate(df[smarts_col]): 
            f.write(smarts + ' MOL{}'.format(count) + '\n')
    f.close()
    #Write the SMARTS to a .txt file for filtering 

    bash_command = f'rd_filters filter --in dataset.smi --prefix filtered'
    os.system(bash_command)
    #Run the bash command

    with open('filtered.smi', 'r') as f: 
            val_lis = f.readlines()
    f.close()
    filter_len = (len(val_lis))
    #Write the SMILES which passed the filters

    if save_props == True:
        rs_df = pd.read_csv("filtered.csv")
        rs_df = rs_df.drop(["SMILES", "NAME"], axis = 1)
        df = pd.concat([df, rs_df], axis = 1)
    #Save the calculated properties if needed

    filters = []

    for result in df["FILTER"]:
        if result == "OK": 
            filters.append(True)
        else: 
            filters.append(False)

    df["FILTER"] = filters
    #Generate a filter column for the DataFrame
    return df

def calculate_KL(df : pd.DataFrame, props : list, org_filepath = "dataset.csv"): 
    """
    Calculates the KL divergence between the generated molecules and the original dataset
    
    Parameters: 

    df (pd.DataFrame) : The DataFrame with the generated molecules in 

    org_filepath (str) : The file path to the original dataset from which the molecules have been generated

    props (list) : The list of properties for which the KL divergence should be caculated for

    Returns: 

    float : The KL divergence between the generated molecules and the original dataset for the given properties
    """
    org_mols = pd.read_csv(org_filepath)
    #Read in the original molecules
    kl_dict = {}
    #Set up a dictionary for the results
    for prop in props: 
        
        #Iterate over the properies 
        org = np.array(org_mols[prop])
        gen = np.array(df[prop])
        #Get arrays

        kde_o = scipy.stats.gaussian_kde(org)
        kde_g = scipy.stats.gaussian_kde(gen)

        _eval = np.linspace(np.hstack([org, gen]).min(), np.hstack([org, gen]).max())

        org = kde_o(_eval)
        gen = kde_g(_eval)

        kl_dict[prop] = scipy.stats.entropy(org, gen)
        #Calculate the KL divergence for the property
    kl_lis = kl_dict.values()
    #Get the computed KL divergence 
    kl_lis = [np.exp(-kl) for kl in kl_lis]
    #Exp
    score = (sum(kl_lis)/len(kl_lis))
    #calculate score
    return score


header_names = ['SELFIES', 'SMILES', 'Mutation', 'Mutation step', 'InChIKey', 'Tanimoto', 'Dice', 'Cosine', 'Sokal', 'Run', 'SA Score', 'LogP', 'QED', 'Molecular Weight', 'Constraint', 'Molecule number']

def get_properties(df : pd.DataFrame, org_fp, org_mol : Mol): 
    """
    Add properties of the generated molecules to the DataFrame 

    Parameters: 

    df (DataFrame): The DataFrame containing the SMILES strings for the molcules 

    Returns: 

    df (DataFrame): The DataFrame with the additional calculated properties
    """
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol = 'SMILES', molCol = 'Molecule')
    #Add a column with Mol files in 
    df['SMARTS'] = [Chem.MolToSmarts(i) for i in df['Molecule']]
    df['InChIKey'] = [Chem.MolToInchiKey(b) for b in df['Molecule']]
    #Add SMARTS for filter checks and InChIKeys for duplicate searching
    fp_lis = [AllChem.GetMorganFingerprint(c, 2) for c in df['Molecule']]
    #Generate fingerprints in order to calcualte the similarities to the seed molecules
    df['Tanimoto'] = [DataStructs.TanimotoSimilarity(org_fp, d) for d in fp_lis]
    df['Dice'] = [DataStructs.DiceSimilarity(org_fp, d) for d in fp_lis]
    df['Cosine'] = [DataStructs.DiceSimilarity(org_fp, d) for d in fp_lis]
    df['Sokal'] = [DataStructs.DiceSimilarity(org_fp, d) for d in fp_lis]
    #Add similarities
    df['SA Score'] = [sascorer.calculateScore(i) for i in df['Molecule']]
    #Calculate the synthetic accessibility score
    prop_lis = [Chem.QED.properties(i) for i in df['Molecule']]
    df['LogP'] = [i[1] for i in prop_lis]
    df['QED'] = [Chem.QED.qed(mol, prop) for mol, prop in zip(df['Molecule'], prop_lis)]
    df['Molecular Weight'] = [round(Chem.Descriptors.ExactMolWt(mol), 2) for mol in df['Molecule']]
    #Calculate physiochemical properties of the molecules
    org_SAScore = sascorer.calculateScore(org_mol)
    org_props = Chem.QED.properties(org_mol)
    org_logp = org_props[1]
    org_qed = Chem.QED.qed(org_mol, org_props)
    org_mw = round(Chem.Descriptors.ExactMolWt(org_mol), 2)
    #Calculate the properties for the seed molecule
    df["Change in SA Score"] = [new - org_SAScore for new in df["SA Score"]]
    df["Change in LogP"] = [new - org_logp for new in df["LogP"]]
    df["Change in QED"] = [new - org_qed for new in df["QED"]]
    df["Change in Molecular Weight"] = [new - org_mw for new in df["Molecular Weight"]]
    #df = get_nearest_neighbour_sim(df)
    #Calculate nearest neighbour similarity 
    return df

def run_anova(file_lis : list, prop_lis : list): 
    """
    Run ANOVA between the similarities/delta prop for the different constraints 

    Parameters: 

    file_lis (list): List of file paths for the generated molecules 

    prop_lis (list): List of the properties to be used

    Returns: 

    pd.DataFrame : DataFrame detailing the ANOVA results 
    """
    df_lis = [pd.read_csv(file_path, usecols = prop_lis) for file_path in file_lis]
    #Read in the generated molecules
    props = [[df[prop] for df in df_lis] for prop in prop_lis]
    #Get a list of lists of the properties
    res_dict = {"Property" : prop_lis,
                "F-Statistic" : [],
                "p-Value" : [], 
                "Significant" : []}
    #Set up the dictionary
    for prop in props: 
    #Iterate over the properties
        res = scipy.stats.f_oneway(*prop)
        #Run the ANOVA
        res_dict["F-Statistic"].append(res[0])
        res_dict["p-Value"].append(res[1])
        #Add the statistics to the dictionary
        res_dict["Significant"].append(res[1] < 0.05)
        #Add the significance to the dictionary
    df = pd.DataFrame.from_dict(res_dict)
    #Convert to a DataFrame
    return df


#Mutates the SELFIE string
def mutate_selfie(selfie : str, selfies_indices : list, alphabet : list):
    """
    Mutates a single SELFIES by either insertion, replacement or deletion

    Parameters: 

    selfie (str) : The SELFIES string to be mutated 

    selfies_indices (list) : Indices which can be mutated under the set constraint 

    alphabet (list) : The list of the SELFIES characters to be used for the mutatations

    Returns: 

    dict : Dictionary detailing the following
            {"SELFIES" : The mutated SELFIES, 
             "Mutate type" : The type of mutation that was carried out, 
             "Character removed" : The character which has been removed, None if nothing is removed, 
             "Character inserted" : The character which has been added, None if nothing is added}
    
    """
    valid = False
    selfies_chars = list(sf.split_selfies(selfie))
    #Get a list of the SELFIES characters in the string
    choice_ls = ["Insert", "Replace", "Delete"]
    if len(selfies_chars) <= 4:
        random_choice = "Insert"
        #Force the mutatation type to be Insert for a small SELFIES string 
    else:
        random_choice = np.random.choice(choice_ls)
    #Randomly pick a mutation type

    while valid == False:

        # Insert a character in a Random Location
        if random_choice == "Insert": 

            insert_chars = selfies_indices
            insert_chars.append((selfies_indices[-1] + 1))
            #Get the SELFIES indices 

            random_index = np.random.choice(insert_chars)
            random_character = np.random.choice(alphabet, size = 1)[0]
            #Generate a random index and character

            add_char = random_character
            del_char = None
            #Note the character added

            selfie_mutated_chars = selfies_chars[:random_index] + [random_character] + selfies_chars[random_index:]
            #Insert the character

            

        # Replace a random character 
        elif random_choice == "Replace":                        
            random_index = np.random.choice(selfies_indices)
            random_character = np.random.choice(alphabet, size=1)[0]
            if random_index == 0:
                add_char = random_character
                del_char = None
                selfie_mutated_chars = [random_character] + selfies_chars[random_index+1:]
                
            else:
                add_char = random_character 
                del_char = None
                selfie_mutated_chars = selfies_chars[:random_index] + [random_character] + selfies_chars[random_index+1:]
                
        
        # Delete a random character
        elif random_choice == "Delete": 
            random_index = np.random.choice(selfies_indices)

            if random_index == 0:
                del_char = selfies_chars[random_index]
                add_char = None
                selfie_mutated_chars = selfies_chars[random_index + 1:]
            else:
                del_char = selfies_chars[random_index]
                add_char = None
                selfie_mutated_chars = selfies_chars[:random_index] + selfies_chars[random_index+1:]
                

        else: 
                raise Exception('Invalid Operation trying to be performed')
        
        selfie_mutated = "".join(x for x in selfie_mutated_chars)
        #Combine the SELFIES into a single string

        try:
            if len(selfie_mutated_chars) > 3:
                smiles = sf.decoder(selfie_mutated)
                if Chem.MolFromSmiles(smiles) == None:
                    valid = False
                else:
                    valid = True
            else: 
                valid = False 

        except sf.DecoderError: 
            valid = False 
            
    
    return {"SELFIES" : selfie_mutated,
            "Mutate type" : random_choice, 
            "Character removed" : del_char, 
            "Character added" : add_char}

def get_val_indices(selfies : str, restricted_chars : list):
    """
    Get the indices of the SELFIES string which can be mutated 

    Parameters: 

    selfies (str) : The SELFIES string 

    restricted_chars (list) : The list of SELFIES characters which should not be mutated 

    Returns: 

    list : A list of the indices for the SELFIES characters which can be mutated 
    """
    selfie_list = list(sf.split_selfies(selfies))
    #Get a list of the SELFIES characters 
    indices_list = list(range(len(selfie_list)))
    #Get a list of the possible indices in the SELFIES string
    val_indices = indices_list
    #Set up a list for the valid indices

    for index in indices_list: 
    #Iterate over the indices in the list 
        char = selfie_list[index]
        #Get the character for the index
        if char in restricted_chars:
            if index in val_indices:
                val_indices.remove(index)
                #Remove the index if it refers to a restricted character
                if ("Ring" or "Branch") in char: 
                    try:
                        val_indices.remove(index + 1)
                        #Remove the index if it follows a Ring or Branch character
                    except ValueError: 
                        continue
            else: 
                continue
            
                    
    return val_indices
    #Return the indices list

def run_bulk_mutation(smiles: str, num_mutations : int, pool_size : int, alphabet_num : int, restricted_chars = None, ref_smi = None, print_time : bool = True) -> pd.DataFrame: 
    """
    Runs a bulk mutatation

    Parameters: 

    smiles (str) : The SMILES of the seed molecule

    num_mutations (int) : The number of mutation steps to be carried out 

    pool_size (int) : The pool size of randomized SMILES to be generated and used as a base for the mutatations

    alphabet_num (int) : The number for which alphabet should be used 

    restricted_chars (list): Defaults to None, a list of the characters which cannot be mutated under the constraint

    ref_smi (str) : Defaults to None, needed if the reference molecule is not the same as the seed molecule

    print_time (bool) : Defaults to True, whether the time taken should be printed

    Returns: 

    pd.DataFrame : DataFrame for the generated molecules
    
    """
    start_time = time.time()
    alphabet = (LECSSVariables()).alph_dict[str(alphabet_num)]
    #Get the alphabet corresponding to the number
    ran_self_lis = [sf.encoder(smi) for smi in randomize_smiles(Chem.MolFromSmiles(smiles), pool_size)]
    #Generate a randomized pool of SELFIES
    dict_lis = []
    #Set up a list to add the dictionaries to 

    for selfies in ran_self_lis:
    #Iterate over the randomized SELFIES 
            for _ in range(num_mutations): 
                #Iterate over a set number of mutation steps
                if type(restricted_chars) == list:
                    selfies_indices = get_val_indices(selfies, restricted_chars)
                else: 
                    selfies_indices = list(range(len(list(sf.split_selfies(selfies)))))

                mutate_dict = mutate_selfie(selfies, selfies_indices, alphabet)
                del selfies_indices
                gc.collect()
                mutate_dict["Mutation step"] = _
                #Append the mutation step to the dictionary
                selfies = mutate_dict["SELFIES"]
                #Use the new SELFIES for the next mutation
                dict_lis.append(mutate_dict)
                #Add the dictionary to the list

    df = pd.DataFrame(dict_lis)
    #Generate a DataFrame from the list of dictionaries
    df["SMILES"] = [sf.decoder(smi) for smi in df["SELFIES"]]
    #Get the SMILES for the generated SELFIES
    if ref_smi == None: 
        mol = Chem.MolFromSmiles(smiles)
    else: 
        mol = Chem.MolFromSmiles(ref_smi)
    #Generate an RDKit mol file for the original or reference molecule
    fp = get_fingerprint(mol)
    #Generate a Morgan fingerprint for the original or reference molecule
    df = get_properties(df, fp, mol)
    #Calculate various physiochemical properties for the molecules
    end_time = time.time() - start_time
    if print_time: 
        print(f"It took {round(end_time/60, 2)} minutes to complete this mutation run")
    return df

def run_mutation_on_set(smi_lis : list, num_mutations : int, pool_size : int, alphabet_num : int, restricted_chars = None, id_lis = None, file_path = "mutated.csv"): 
    """
    Runs the mutation on a set of molecules using the provided conditions 

    Parameters: 

    smi_lis (list): The list of SMILES for the molecules to be mutated  

    num_mutations (int) : The number of mutation steps to be carried out 

    pool_size (int) : The pool size of randomized SMILES to be generated and used as a base for the mutatations

    alphabet_num (int) : The number for which alphabet should be used 

    restricted_chars (list) : Defaults to None, A list of the characters which cannot be mutated under the constraint

    id_lis (list) : Defaults to None and the indices are used instead, a list of identifiers for the SMILES

    file_path (str) : Defaults to "mutated.csv", the file path for the outputted molecules to be written to

    Returns: 

    pd.DataFrame : DataFrame containing the generated molecules 
    """
    if type(id_lis) != list: 
    #Generate an ID list if one is not provided
        id_lis = [f"MOL{index}" for index in range(len(smi_lis))]

    df = pd.DataFrame(columns = (LECSSVariables()).column_names)
    df.to_csv(file_path, header = True, index = False)
    #Set up inital .csv

    for smi, smi_id in zip(smi_lis, id_lis):
        start_time = time.time()
        print(f"Currently mutating {smi_id} \n")
        #Iterate over the SMILES and IDs
        df_1 = run_bulk_mutation(smi, num_mutations, pool_size, alphabet_num , restricted_chars)
        #Run the mutation 
        df_1["Original molecule"] = [smi_id for x in range(len(df_1))]
        #Add a column detailing the original molecule
        print(f"Completed mutating {smi_id}, which took {round(((time.time() - start_time)/60), 2)} minutes \n")
        df_1.to_csv(file_path, header = False, index = False, mode = "a")
        #Append the DataFrame to the working .csv
        del df_1
        gc.collect()

props = ["Tanimoto", "Cosine", "Dice", "Sokal", "Change in LogP", "Change in QED", "Change in SA Score", "FILTER"]
        
def change_count(prop_series : list): 
    """
    Counts the number of molecules which have caused an increase or decrease in a certain physiochemical property

    Parameters: 

    prop_series (pd.Series): The Pandas Series of the property being counted for, in this instance should be one of
                         Change in LogP, Change in QED, Change in SA Score

    Returns: 

    dict : Dictionary containing the results
        """
    increase_count = prop_series.between(0, 10).sum()
    decrease_count = prop_series.between(-10, -0.00001).sum()
    #Count the number of molecules which saw an increase in their physiochemical property from the seed molecule

    increase_percent = round(((increase_count/len(prop_series)) * 100), 2)
    decrease_percent = round(((decrease_count/len(prop_series)) * 100), 2)
    #Get the percentages molecules which saw an increase in their physiochemical property from the seed molecule
    return {"Increase" : {"Count" : increase_count, "Percentage" : increase_percent}, 
            "Decrease" : {"Count" : decrease_count, "Percentage" : decrease_percent}}
        
def sim_count(prop_series : pd.Series): 
    """
    Counts the number of molecules in a certain range for the similarity measures

    Parameters: 

    prop_series (pd.Series): The Pandas Series of the property being counted for, in this instance should be one of
                            Tanimoto, Cosine, Dice, Sokal

    Returns: 

    dict : Dictionary containing the results
    """
    low_count = prop_series.between(0.4, 1).sum()
    medium_count = prop_series.between(0.6, 1).sum()
    high_count = prop_series.between(0.75, 1).sum()
    #Get the count for the similarities in the relevant ranges
    low_percent = round(((low_count/len(prop_series)) * 100), 2)
    medium_percent = round(((medium_count/len(prop_series)) * 100), 2)
    high_percent = round(((high_count/len(prop_series)) * 100), 2)
    #Get the percentages for the molecules in the relevant range
    return {"x > 0.75" : {"Count" : high_count, "Percentage" : high_percent}, 
            "x > 0.6" : {"Count" : medium_count, "Percentage" : medium_percent}, 
            "x > 0.4" : {"Count" : low_count, "Percentage" : high_percent}}

def filter_count(prop_series : pd.Series):
    """
    Counts the number of molecules in the generated dataset which have passed the compound quality filters

    Parameters: 

    prop_series (pd.Series): The Pandas Series of the property being counted for, in this instance should be FILTER

    Returns: 

    dict : Dictionary containing the results
        
    """
    prop_lis = prop_series.tolist()
    #Generate a list from the series
    true_count = prop_lis.count(True)
    false_count = prop_lis.count(False)
    #Get the count for the molecules which passed or failed the filters
    true_percent = round(((true_count/len(prop_lis)) * 100), 2)
    false_percent = round(((false_count/len(prop_lis)) * 100), 2)
    #Get the percentage of molecules which passed or failed the filters
    return {"Passed" : {"Count" : true_count, "Percentage" : true_percent}, 
            "Failed" : {"Count" : false_count, "Percentage" : false_percent}}

def get_used_details(file_path : str): 
    """
    Takes the file path for the generated molecules and breaks it down into the following features: 

    * Pool size 
    * Alphabet used
    * Restrictions uses

    Paratemeters: 
     
    file_path (str): The file path for the generated molecules

    Returns: 

    dict: Dictionary providing the mutation details
    """
    restriction_key = {"B" : "Heteroatoms", "C" : "Rings", "D" : "Branches", "none" : "None", None : "None", "" : "None"}
    #Define the key required for the restrictions
    det_dict = {}
    #Initialize a dictionary
    file_path = file_path.split("/")[-1]
    #Get just the specific file name
    splits = file_path.split("_")
    #Split by underscore
    det_dict["Pool size"] = splits[0]
    #Append the pool size
    det_dict["Alphabet"] = splits[1]
    #Append the alphabet number
    restriction_char = splits[2][:-4]
    det_dict["Restriction"] = restriction_key[restriction_char]
    #Append the restriction type
    return det_dict

def run_count(file_path : str, filters : bool = True, unique : bool = True):
    """
    Bin the properties of the generated molecules for a more readable table of results

    Parameters: 

    file_path (str): The file path for the .csv containing the generated molecules

    filters (bool) : Whether or not the rd_filters should be carried out on the generated molecules and reported

    unique (bool) : Whether the number of unique molecules generated should be calculated and reported
    
    Returns: 

    pd.DataFrame: DataFrame of the results
    """
    count_dict = get_used_details(file_path)
    #Get the details of the mutation
    
    for sims in ["Tanimoto", "Cosine", "Dice", "Sokal"]: 
        prop_series = pd.Series(pd.read_csv(file_path, usecols = [sims])[sims])
        #Read the property in as a DataFrame that can be used as a series
        res_dict = sim_count(prop_series)
        #Count the values
        count_dict[f"{sims} x > 0.75"] = f"{res_dict['x > 0.75']['Count']} ({res_dict['x > 0.75']['Percentage']}%)"
        count_dict[f"{sims} x > 0.6"] = f"{res_dict['x > 0.6']['Count']} ({res_dict['x > 0.6']['Percentage']}%)"
        count_dict[f"{sims} x > 0.4"] = f"{res_dict['x > 0.4']['Count']} ({res_dict['x > 0.4']['Percentage']}%)"
        #Append results to the dictionary
        del prop_series, res_dict
        #Delete the series to manage memory 

    for change in ["Change in LogP", "Change in QED", "Change in SA Score"]: 
        prop_series = pd.Series(pd.read_csv(file_path, usecols = [change])[change])
        #Read the property in as a DataFrame that can be used as a series
        res_dict = change_count(prop_series)
        #Count the values       
        count_dict[f"Increase in {change[10:]}"] =  f"{res_dict['Increase']['Count']} ({res_dict['Increase']['Percentage']}%)"
        count_dict[f"Decrease in {change[10:]}"] =  f"{res_dict['Decrease']['Count']} ({res_dict['Decrease']['Percentage']}%)"
        #Append results to the dictionary 
        del prop_series, res_dict
        #Delete the series to manage memory

    if filters: 
        df = pd.read_csv(file_path, usecols = ["SMARTS", "SMILES"])
        df = check_filters(df)
        filter_count = df["FILTER"].tolist().count(True)
        filter_count_percent = round(((filter_count/len(df)) * 100), 2)
        count_dict["Passed filters"] = f"{filter_count} ({filter_count_percent}%)"
        
        
    return count_dict

def get_table(dir_name : str, filters : bool = True, unique : bool = True): 
    """
    Generate a table of results 

    Parameters : 

    dir_name (str) : Name of the directory where the generated molecules are stored 

    filters (bool) : Whether or not the rd_filters should be carried out on the generated molecules and reported

    unique (bool) : Whether the number of unique molecules generated should be calculated and reported

    Returns : 

    pd.DataFrame : DataFrame of results 
    
    """
    files = os.listdir(dir_name)
    #Get a list of the file names for the generated molecules 
    files = [f"{dir_name}/{file}" for file in files]
    dicts = [run_count(file_path = file, filters = filters, unique = unique) for file in files]
    #Generate the dictionaries 
    df = pd.DataFrame(dicts)
    return df

def get_properties_single(smiles : str): 
    """
    Add properties of the generated molecules to the DataFrame 

    Parameters: 

    smiles (str) : The SMILES string for the molecule 

    Returns: 

    df (DataFrame): DataFrame with calculated properties
    """
    df = pd.DataFrame()
    #Create DataFrame 
    df["SMILES"] = [smiles]
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol = 'SMILES', molCol = 'Molecule')
    #Add a column with Mol files in 
    df['SMARTS'] = [Chem.MolToSmarts(i) for i in df['Molecule']]
    df['InChIKey'] = [Chem.MolToInchiKey(b) for b in df['Molecule']]
    #Add SMARTS for filter checks and InChIKeys for duplicate searching
    df["SELFIES"] = [sf.encoder(s) for s in df["SMILES"]]
    #Add SELFIES
    df['SA Score'] = [sascorer.calculateScore(i) for i in df['Molecule']]
    #Calculate the synthetic accessibility score
    prop_lis = [Chem.QED.properties(i) for i in df['Molecule']]
    df['LogP'] = [i[1] for i in prop_lis]
    df['QED'] = [Chem.QED.qed(mol, prop) for mol, prop in zip(df['Molecule'], prop_lis)]
    df['Molecular Weight'] = [round(Chem.Descriptors.ExactMolWt(mol), 2) for mol in df['Molecule']]
    #Calculate physiochemical properties of the molecules 
    return df

def get_legends(df : pd.DataFrame, mutation_steps : int, org_mol_df : pd.DataFrame, representation : str = "SELFIES", prop : str = None): 
    """
    Defines the legend names for the image

    Parameters : 

    df (pd.DataFrame) : DataFrame containing the generated molecules to be displayed
    
    mutation_steps (int) : The number of mutation steps which have been carried out
    
    org_mol_df (str) : DataFrame for the seed molecule

    representation (str) : Defaults to "SELFIES", the representation to be displayed on the image 

    prop (str) : Defaults to None, the property to be displayed 
    """
    if prop != None: 
        if prop not in ["Tanimoto", "Dice", "Sokal", "Cosine"]: 
            legends = [f"Seed molecule: \n \n{prop} : {round(org_mol_df[prop].loc[org_mol_df.index[0]], 2)}"]
        else: 
            legends = ["Seed molecule"]
        for x in range(mutation_steps): 
            prop_val = round(df[prop].loc[df.index[x]], 2)
            #Get the property for the generated molecule 
            legend = f"Step {x + 1}: \n{prop} : {prop_val}"
            legends.append(legend)

    else: 
        legends = [f"Seed molecule: \n \n{org_mol_df[representation]}"]
        for x in range(mutation_steps):
            string = df[representation].loc[df.index[x]]
            #Get the representation for the molecule 
            legend = f"Step {x + 1}: \n{string}"
            legends.append(legend)
    return legends

def draw_mutation_example(df : pd.DataFrame, seed_molecule : str, representation : str = "SELFIES", display_prop : bool = False, prop : str = None, file_path : str = "mutation_example"):
    """
    Draw out an example of the mutation steps taken

    Parameters : 

    df (pd.DataFrame) : DataFrame containing the generated molecules 
    
    seed_molecule (str) : The SMILES string for the seed molecule 

    representation (str) : Defaults to "SELFIES", the representation to be displayed on the image 

    display_prop (bool) : Defaults to False, whether one of the calculated properties should be displayed in the image 

    prop (str) : Defaults to None, the property to be displayed 

    file_path (str) : Defaults to "mutation_example", file path for the image to be saved to
    """
    mutation_steps = df["Mutation step"].tolist()
    num_mutations = max(mutation_steps)
    #Get the number of mutation steps carried out
    start = (example - 1)*num_mutations
    end = start + num_mutations
    mols_to_display = df.loc[start : end]
    #Get the example
    PandasTools.AddMoleculeColumnToFrame(mols_to_display, molCol = "Molecule", smilesCol = "SMILES")
    #Ensure there is a Molecule column 
    org_mol_df = get_properties_single(smiles = seed_molecule)
    #Get a DataFrame of properties for the starting molecule
    legends = get_legends(df = mols_to_display, mutation_steps = num_mutations, org_mol_df = org_mol_df, representation = representation, prop = prop)
    #Get the legend names
    mols = mols_to_display["Molecule"].tolist()
    mols.insert(0, Chem.MolFromSmiles(seed_molecule))
    #Get a list of the molecules to be displayed
    [rdCoordGen.AddCoords(mol) for mol in mols]
    img = Draw.MolsToGridImage(mols, molsPerRow = round((num_mutations + 1)/2), legends = legends, returnPNG = False)
    #Generate image
    img.save(f"./{file_path}.png")

