import copy
import csv
import fileinput
import json
import os
import glob
import subprocess
import sys
from collections import defaultdict, namedtuple
from pathlib import Path
from os import listdir
from os.path import isfile, join

import hjson
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skopt
from ase.calculators.lammpslib import LAMMPSlib
from ase.calculators.lammpsrun import LAMMPS
from ase.io import iread, read, write
from xpot.transfer_funcs.general import *
from xpot.potential_parsers.snap import *
from ase.data import chemical_symbols, atomic_masses

path = os.getcwd()
print(path)

this_file = Path(__file__).resolve()

def new_val_snap_standard(obj, sweep_name, iteration, method="mae", xval_set=0):
        # Calculate errors for current GAP iteration through LAMMPS + ASE.
        
        potential = obj.snap_total["[OUTFILE]"]["potential"]
        print(path + " new path")
    
        if "pair_coeff1" in obj.snap_total["[REFERENCE]"]:
            hybrid_overlay = obj.snap_total["[REFERENCE]"]["pair_style"].split()
            for i in hybrid_overlay:
                if i == "zero":
                    index = hybrid_overlay.index(i)
                    #print(index)
                    hybrid_overlay.remove(obj.snap_total["[REFERENCE]"]["pair_style"].split()[index+1])
                    hybrid_overlay.remove(obj.snap_total["[REFERENCE]"]["pair_style"].split()[index])
                    hybrid_overlay.append("snap")
            hybrid_overlay = " ".join(hybrid_overlay)
            cmds = [f"pair_style {hybrid_overlay}", 
                    f"pair_coeff * * snap ./{potential}.snapcoeff ./{potential}.snapparam {chemical_symbols[obj.elements]}"]
            for k in obj.snap_total["[REFERENCE]"].items():
                if "pair_coeff" in k[0] and k[0] != "pair_coeff1":
                    cmds.append(f"pair_coeff {k[1]}")
                else:
                    pass
            cmds += [cmds.pop(1)]
            print(cmds)
        else:
            #cmds = ["pair_style snap", 
            #        f"pair_coeff * * ./{potential}.snapcoeff ./{potential}.snapparam {chemical_symbols[obj.elements]}"]
            cmds = ["pair_style snap", 
                    f"pair_coeff * * ./{potential}.snapcoeff ./{potential}.snapparam {chemical_symbols[obj.elements]}"]
        #print(cmds)
        #lmp = LAMMPSlib(lmpcmds=cmds, lammps_header=lmp_header)
        print("continuing")
        Structure = namedtuple("Structure", 
                ["label", "dft_e", "lmp_e", "dft_f", "lmp_f"])

        structs = []
        train_list_files = [k[0] for k in obj.snap_total["[GROUPS]"].items() if "train" in k[0]]
        for i in train_list_files:
            train_list = iread(f"{input_loc}/{i}.{obj.snap_total['[SCRAPER]']['scraper'].lower()}")
            
            for atoms in train_list:
                atoms.info["config_type"] = i.split("/")[-1].split(".")[0]
                dft_e = atoms.info["energy"]/np.sqrt(len(atoms))
                dft_f = atoms.get_forces()
                try:
                    dft_s = atoms.get_stress()
                except:
                    dft_s = np.zeros(6)
                # lmp = LAMMPS(parameters=cmds, files=files)
                lmp = LAMMPSlib(lmpcmds=cmds)
                atoms.calc = lmp
                elems_list = atoms.get_chemical_symbols()
                offset_list = []
                for k,v in obj.snap_total['[ESHIFT]'].items():
                    if k in elems_list:
                        offset_list.append(v*elems_list.count(k))
                offset = sum(offset_list)
                lmp_e = (atoms.get_potential_energy() - offset)/np.sqrt(len(atoms))
                lmp_f = atoms.get_forces()
                structs.append(Structure(atoms.info['config_type'], dft_e, 
                        lmp_e, dft_f, lmp_f))


        test_list_files = [k[0] for k in obj.snap_total["[GROUPS]"].items() if "test" in k[0]]
        for i in test_list_files:
            test_list = iread(f"{input_loc}/{i}.{obj.snap_total['[SCRAPER]']['scraper'].lower()}")
            for atoms in test_list:
                atoms.info["config_type"] = i.split("/")[-1].split(".")[0]
                dft_e = atoms.info["energy"]/np.sqrt(len(atoms))
                dft_f = atoms.get_forces()
                try:
                    dft_s = atoms.get_stress()
                except:
                    dft_s = np.zeros(6)
                lmp = LAMMPSlib(lmpcmds=cmds)
                # lmp = LAMMPS(parameters=cmds, files=files)
                atoms.calc = lmp
                elems_list = atoms.get_chemical_symbols()
                offset_list = []
                for k,v in obj.snap_total['[ESHIFT]'].items():
                    if k in elems_list:
                        offset_list.append(v*elems_list.count(k))
                offset = sum(offset_list)
                lmp_e = (atoms.get_potential_energy() - offset)/np.sqrt(len(atoms))
                lmp_f = atoms.get_forces()
                structs.append(Structure(atoms.info['config_type'], dft_e, 
                        lmp_e, dft_f, lmp_f))
                #print(lmp_e, dft_e)

        print("finished_loop - now validate")
        errors = validate_errors(obj, structs, sweep_name, iteration, method=method)
        combined_error = obj.loss_ratio * errors.test_e + (1 - obj.loss_ratio) * errors.test_f
        return combined_error
        
def validate_errors(obj, error_list, sweep_name, iteration, method="mae"):
    err = namedtuple("err", "test_e train_e test_f train_f")
    if method == "mae":
        mae = namedtuple("mae", "label e f") # stress")
        maes = []
        for i in error_list:
            e = np.abs(i.dft_e - i.lmp_e)
            f = np.square(np.subtract(i.dft_f,i.lmp_f)) # squres of the force components
            f = np.sqrt(np.sum(f, axis=1)) # the magnitude of force error for each atom
            #s = np.abs(i.dft_s - i.lmp_s)
            maes.append(mae(i.label, e, f)) #, s))
        
        test_e = np.mean([i.e for i in maes if "test" in i.label])
        train_e = np.mean([i.e for i in maes if "train" in i.label])
        test_f_tmp = np.vstack([i.f for i in maes if "test" in i.label])
        train_f_tmp = np.vstack([i.f for i in maes if "train" in i.label])
        test_f = np.mean([np.abs(i) for i in test_f_tmp])
        train_f = np.mean([np.abs(i) for i in train_f_tmp])
        #test_s = np.mean([i.s for i in maes if i.label == "test"])
        #train_s = np.mean([i.s for i in maes if i.label == "train"])

        #print(os.getcwd() + " Where the errors are being printed")
        #new_path = Path(path / obj.project_name / sweep_name / "errors.csv")
        with open(f"{path}/{obj.project_name}/{sweep_name}/errors.csv","a+") as f:
            f.writelines(
                f"{iteration},{train_e},{test_e},{train_f},"
                f"{test_f},NaN,NaN\n"
            )
        #return err(test_e, train_e, test_f, train_f) #, test_s, train_s)
    
    elif method == "rmse":
        rmse = namedtuple("rmse", "label e f") # stress")
        rmses = []
        for i in error_list:
            e = (i.dft_e - i.lmp_e)**2
            f = np.square(np.subtract(i.dft_f,i.lmp_f)) # squres of the force components
            f = np.sqrt(np.sum(f, axis=1)) # the magnitude of force error for each atom
            #s = (i.dft_s - i.lmp_s)**2
            rmses.append(rmse(i.label, e, f)) #, s))
        
        test_e = np.sqrt(np.mean([i.e for i in rmses if "test" in i.label]))
        train_e = np.sqrt(np.mean([i.e for i in rmses if "train" in i.label]))
        test_f_tmp = np.concatenate([i.f for i in rmses if "test" in i.label])
        train_f_tmp = np.concatenate([i.f for i in rmses if "train" in i.label])
        test_f = np.sqrt(np.mean(np.square(test_f_tmp)))
        train_f = np.sqrt(np.mean(np.square(train_f_tmp)))
        #test_s = np.sqrt(np.mean([i.s for i in rmses if i.label == "test"]))
        #train_s = np.sqrt(np.mean([i.s for i in rmses if i.label == "train"]))

        #print(os.getcwd() + " Where the errors are being printed")
        #new_path = Path(path / obj.project_name / sweep_name / "errors.csv")
        with open(f"{path}/{obj.project_name}/{sweep_name}/errors.csv","a+") as f:
            f.writelines(
                f"{iteration},{train_e},{test_e},{train_f},"
                f"{test_f},NaN,NaN\n"
            )
        print("errors printed")

    return err(test_e, train_e, test_f, train_f) #, test_s, train_s)

if __name__ == "__main__":
    # load in SNAP class
    infile = sys.argv[1]
    path = sys.argv[2]
    input_loc = sys.argv[3]
    ml_obj = SNAP(infile)
    if ml_obj.xval_sets == 1:
        iteration = os.getcwd().split("/")[-1]
    elif ml_obj.xval_sets > 1:
        iteration = os.getcwd().split("/")[-2]

    # Run error analysis separately so as not to interfere with fitSNAP lammps
    loss = new_val_snap_standard(ml_obj, ml_obj.sweep_name, iteration, method="rmse")
    with open("./loss.txt", "w+") as f:
        f.write(f"{loss}")
    

