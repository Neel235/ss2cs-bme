import pandas as pd
import numpy as np
import os
import BME.bme_reweight as bme

import matplotlib
from matplotlib.pyplot import hist2d
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Tahoma']

def BME_find_weights(expcs, sim, theta, conformers):
  # generate random weights 
  w0 = np.random.random(conformers)
  # normalize
  w0 /= np.sum(w0)

  # initialize reweighting class with weights                                                                                                                                
  bmea = bme.Reweight(verbose=False, w0=list(w0))
    
  bmea.load(expcs, sim)
  chi2_before,chi2_after, srel = bmea.optimize(theta=theta)
  return list(np.round_(bmea.get_weights(), 5)), list(np.round_(chi2_after, 5))
 
def save_bme_results(tmp, filename):
  mean_w, sd_w = [], []
  for i in range(0, tmp.shape[0]):
    w = []
    tmp2 = tmp[i,:,:]    
    for j in range(0, tmp2.shape[1]):
      w.append(np.mean(tmp2[:,j]))
      w.append(np.std(tmp2[:,j]))
    mean_w.append(w)
  pd.DataFrame.from_records(mean_w).to_csv(filename, sep=' ', header=None, index=False)

def add_accuracy(row):
  if "C" in row['nucleus']:
    return 0.84
  else:
    return 0.11

def BME_find_theta(expcs, sim, theta, w0 = None):
    
  if w0 is None:
    bmea = bme.Reweight(verbose=False)
  else:
      bmea = bme.Reweight(verbose=False, w0=w0)    

  # load data
  bmea.load(expcs, sim)
  chi2_before,chi2_after, srel = bmea.optimize(theta=theta)
  return chi2_after

  # initialize reweighting class with weights                                                                                                                                
  bmea = bme.Reweight(verbose=False)
  bmea.load(expcs, sim)
  chi2_before,chi2_after, srel = bmea.optimize(theta=theta)
  return list(np.round_(bmea.get_weights(), 5))

def null_weights(N=18):
  return([float(1/N) for i in range(0, N)])

def load_ss2cs_model(nucleus, rna = '', DIR_PATH = '/content/drive/My Drive/RESEARCH/ss2cs/'):
  ''' load save model '''
  filename = DIR_PATH + 'output/RF_'+ rna + nucleus + '.sav'
  model = pickle.load(open(filename, 'rb'))
  return(model)

def read_exp_cs(input_exp):    
    # read in experimental chemical shift file
    names = ['resname', 'resid', 'nucleus', 'expcs', 'NA']
    expcs = pd.read_csv(input_exp, sep = "\s+", header = None, names = names)
    return(expcs)

def read_computed_cs(input_sim, names = ['model', 'resid', 'resname', 'nucleus', 'simcs', 'id']):
    # read in computed (simulated) chemical shift file    
    simcs = pd.read_csv(input_sim, sep = "\s+", header = None, names = names)
    return(simcs)
        
def merge_exp_computed_cs(input_exp, input_sim):
    # read in experimental chemical shift file
    expcs = read_exp_cs(input_exp)

    # read in computed (simulated) chemical shift file
    simcs = read_computed_cs(input_sim)

    # merge predicted, measured, and accuracy information
    cs = simcs.merge(expcs, on = ['resid', 'resname', 'nucleus'])
    cs = cs.sort_values(by = ['model', 'resid', 'nucleus'])
    return(cs)

def pair_cs(cs, csname = "expcs", debug = False, pairing = {"H1'" : "C1'", "H2'" : "C2'", "H3'" : "C3'", "H4'" : "C4'", "H5'" : "C5'", "H5''" : "C5'", "H2" : "C2", "H5" : "C5", "H6" : "C6", "H8" : "C8"}):
    ''' Function to pair proton and carbon chemical shift file '''
    resid_list, resname_list, proton_name_list, carbon_name_list, proton_cs_list, carbon_cs_list  = [],[],[],[],[],[]
    for resid in cs['resid'].unique():
        # get resname
        tmp = cs[(cs['resid']==resid)]
        for resname in tmp['resname'].unique():
            for h in pairing.keys():
                c = pairing[h]    
                try:
                    h_cs = tmp[(tmp['resid']==resid) & (tmp['resname']==resname) &  (tmp['nucleus']==h)]
                    c_cs = tmp[(tmp['resid']==resid) & (tmp['resname']==resname) &  (tmp['nucleus']==c)]

                    # append
                    if c_cs[csname].values[0] and h_cs[csname].values[0]:
                        carbon_cs_list.append(c_cs[csname].values[0])
                        proton_cs_list.append(h_cs[csname].values[0])                
                        resid_list.append(resid)
                        resname_list.append(resname)
                        proton_name_list.append(h)
                        carbon_name_list.append(c)
                        if debug:
                            print(resid, resname, h, c, h_cs[csname].values[0], c_cs[csname].values[0])
                except:
                    pass

    cs = pd.DataFrame({"resid": resid_list, 
                  "resname": resname_list, 
                  "proton_name": proton_name_list, 
                  "carbon_name": carbon_name_list,
                  "proton_cs": proton_cs_list,
                  "carbon_cs": carbon_cs_list})   
    return(cs)

def create_paired_data(cs):
    ''' generates paired data set for experimental and simulated data, respectively '''
    # pair data from experiments
    expcs_paired = pair_cs(cs[cs['model']==1] , csname = "expcs")

    # paired data from simulated/computation
    simcs_paired = {}
    for model in cs['model'].unique():
        simcs_paired[model] = pair_cs(cs[cs['model']==model], csname = "simcs")
    
    return(expcs_paired, simcs_paired)

def generate_2D_histogram(data, xrange = [63,161], xgrid_size = 0.90, yrange = [3.5, 9], ygrid_size = 0.17, x_name = "carbon_cs", y_name = "proton_cs"):
    xbins = int((xrange[1] - xrange[0])/xgrid_size)
    ybins = int((yrange[1] - yrange[0])/ygrid_size)
    # returns: h, xedge, yedge, image = 
    return(hist2d(x=data[x_name], y = data[y_name], bins = [xbins, ybins], range = [xrange, yrange]))

def create_histogram_data(expcs_paired, simcs_paired):
    ''' generates paired data set for experimental and simulated data, respectively '''
    # pair data from experiments
    h, xedge, yedge, image = generate_2D_histogram(expcs_paired)
    expcs_hist = h.flatten()

    # paired data from simulated/computation
    simcs_hist = {}
    for model in simcs_paired.keys():        
        h, xedge, yedge, image = generate_2D_histogram(simcs_paired[model])
        simcs_hist[model] = h.flatten()
    
    return(expcs_hist, simcs_hist)

def write_BME_chemical_shifts_unassigned(expcs_hist, simcs_hist, output_name_exp = "data/bme_experimental_7JU1_unassigned.dat", output_name_sim = "data/bme_simulated_7JU1_unassigned.dat", error = 1):
    ''' writes output flattened, BME-ready histogram files '''
    expcs = pd.DataFrame({"0": expcs_hist})
    expcs["1"] = error
    expcs.columns = ['DATA=JCOUPLINGS', 'PRIOR=GAUSS']
    expcs.index = [i for i in range(0, expcs.shape[0])]
    simcs = pd.DataFrame.from_dict(simcs_hist, orient='index')
    expcs.to_csv(output_name_exp, sep = " ", index = True, index_label = "#")
    simcs.to_csv(output_name_sim, sep = " ", index=True, header=None)
    return(expcs, simcs)
    


def write_BME_chemical_shifts(input_exp = "data/chemical_shifts/measured_shifts_ 7JU1.dat", 
                              input_sim = "data/chemical_shifts/computed_shifts_7JU1.dat",
                              input_accuracy = "data/chemical_shifts/uncertainity.dat",
                              output_name_exp = "data/chemical_shifts/bme_experimental_7JU1.dat", 
                              output_name_sim = "data/chemical_shifts/bme_simulated_7JU1.dat"):
    """ Writes out files chemical shift files needed for BME 
            input_exp: path to input experimental chemical shift file
            input_sim: path to input computed (simulated) chemical shift file
            input_accuracy: path to input accuracy (uncertainty) file
            output_name_exp: path to output experimental chemical shift file formatted for BME analysis
            output_name_sim: path to output simulated chemical shift file formatted for BME analysis
    """
    # read in experimental chemical shift file
    names = ['resname', 'resid', 'nucleus', 'expCS', 'NA']
    expcs = pd.read_csv(input_exp, sep = "\s+", header = None, names = names)

    # read in computed (simulated) chemical shift file
    names = ['model', 'resid', 'resname', 'nucleus', 'simcs', 'id']
    simcs = pd.read_csv(input_sim, sep = "\s+", header = None, names = names)

    # read in accuracy file
    names = ['nucleus', 'error']
    accuracy = pd.read_csv(input_accuracy, sep = "\s+", header = None, names = names)

    # merge predicted, measured, and accuracy information
    cs = simcs.merge(expcs, on = ['resid', 'resname', 'nucleus']).merge(accuracy, on = ['nucleus'])
    cs = cs.sort_values(by = ['model', 'resid', 'nucleus'])

    # output files for BME
    # experimental
    expcs = cs[['model', 'expCS', 'error']]
    expcs = expcs[expcs.model==1]
    expcs = expcs[['expCS', 'error']]
    expcs.columns = ['DATA=JCOUPLINGS', 'PRIOR=GAUSS']
    expcs.index = [i for i in range(0, expcs.shape[0])]
    expcs.to_csv(output_name_exp, sep = " ", index = True, index_label = "#")
    
    # computed (simulated)
    conformers = cs.model.unique()
    simcs = np.zeros((len(conformers), expcs.shape[0]))
    for j,conformer in enumerate(cs.model.unique()):
        simcs[j,:] = cs[cs.model==conformer].simcs.values        
    simcs = pd.DataFrame.from_records(simcs)
    simcs.to_csv(output_name_sim, sep = " ", index=True, header=None)
    return(expcs, simcs)


def find_weights(exp_file, sim_file, theta, w0 = None, verbose = False):
    """ Find weights using BME 
            exp_file: path to experimental observable file formatted for BME analysis
            sim_file: path to simulated observable file formatted for BME analysis
            theta: scale factor for entropy
            w0 = initial (prior) weights
            verbosity of output
        Returns:
            weights: optimal weights
            chi2_before: chi^2 before reweighting
            chi2_after: chi^2 after reweighting
            srel: the relative entropy of these optimized weights relative to the initial (prior) weights
    """
    if w0 is None:
        bmea = bme.Reweight(verbose=verbose)
    else:
        bmea = bme.Reweight(verbose=verbose, w0=w0)        
    bmea.load(exp_file, sim_file)
    chi2_before, chi2_after, srel = bmea.optimize(theta=theta)
    return bmea.get_weights(), chi2_before, chi2_after, srel

def boltzmann_weight(energy, kB = 0.001987, T = 298):
    return(np.exp(-energy/(kB*T)))