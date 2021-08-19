# %%
import os
import sys
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import argparse
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from extractFeatures import extractCT,get_resname_char


def load_ss2cs_model(nucleus, DIR_PATH):
  ''' load save model '''
  filename = DIR_PATH + '/model/RF_' + nucleus + '.sav'
  model = pickle.load(open(filename, 'rb'))
  return(model)

def processfile(inFile, outFile, DIR_PATH, save = False):
    
    # initialize    
    rna = "user"
    nuclei = ["C1'", "C2'", "C3'", "C4'", "C5'","C2","C5","C6","C8", "H1'", "H2'", "H3'","H4'", "H5'","H5''","H2","H5","H6","H8"]

    # featurization
    features = extractCT(inFile, rna)
    features.drop('i_resname_char', axis=1, inplace=True)

    # fit one hot encoder
    train_X = pd.read_csv(DIR_PATH+"/data/train_X_NEW.csv",sep=' ',header=0)
    train_X = train_X.drop(['id','length','resid'],axis = 1)
    enc = preprocessing.OneHotEncoder(sparse = False)
    enc.fit(train_X)
    
    # fit model for each nucleus type
    results = pd.DataFrame([])
    for nucleus in nuclei:
    # one hot encoding testing data
        features_resname = features.drop(['id', 'length', 'resid'],axis=1)
        features_info = features['length']
        features_resname_enc = pd.DataFrame(enc.transform(features_resname))
        features_enc = pd.concat([features_info, features_resname_enc],axis = 1)

        # model prediction
        model = load_ss2cs_model(nucleus, DIR_PATH)
        y_pred = model.predict(features_enc)

        # format prediction
        output_resname = features['i_resname'].apply(lambda x: get_resname_char(x))
        output_resid = features['resid']
        output_nucleus = pd.Series([nucleus]*len(features))
        output_cs = pd.Series(y_pred)
        output_error = pd.Series(["."]*len(features))
        result = pd.concat([output_resname, output_resid, output_nucleus, output_cs, output_error],axis=1)
        results = pd.concat([results, result],ignore_index=True)
    
    if save:
        results.to_csv(outFile, sep=' ', header=None, index=False)
    return results
# %%    
def concat(inFile, outFile, DIR_PATH):
    ctfiles = [f for f in os.listdir(inFile) if f.endswith(".ct")]  
    prefix = []
    for file in ctfiles:
        prefix.append(file[:4])
    prefix = list(set(prefix))
    df_list = []
    for x in prefix:
        print(x)
        ctfiles_x = [f for f in ctfiles if f[:4] == x]
        df = pd.DataFrame()
        orders = [int(file.split(".")[0].split("_")[1]) for file in ctfiles_x]
        orders.sort()
        for order in orders:
            file = x + "_" + str(order) + ".ct"
            inFileName = os.path.join(inFile, file)
            outFileName = os.path.join(outFile, file.replace(".ct", ".csv"))
            df_cur = processfile(inFileName, outFileName, DIR_PATH, False)
            df_cur["id"] = order
            df = pd.concat([df, df_cur])
        df = df.drop(columns = [2]).reset_index(drop = True).rename(columns = {0:"nucleus", 1: "shift"})
        df_list.append(df)
    return list(zip(df_list, prefix))

def main():
    # configure parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input", help="path folder containing CT files", required = True)
    parser.add_argument("-o","--output", help="path to folder to save CSV files", required = True)
    parser.add_argument("-s","--ss2cs_path", help="path to SS2CS repo", required = True)

    # parse command line
    a = parser.parse_args()  

    # initialize    
    inFile = a.input
    outFile = a.output
    DIR_PATH = a.ss2cs_path   
    ctfiles = [f for f in os.listdir(inFile) if f.endswith(".ct")]
    print("Current Output:", os.path.abspath(outFile))
    print("Processing Files:", len(ctfiles))

    csv_files = concat(inFile, outFile, DIR_PATH)
    for df, file_name in csv_files:
        df.to_csv(os.path.join(outFile, file_name + ".csv"))
    
    print("Done")
# %%

    
if __name__ == "__main__":
    main()