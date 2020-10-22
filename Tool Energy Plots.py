#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import copy
import random
import time
from collections import defaultdict
from tqdm.notebook import tqdm
import h5py


numberFilesLabel_1=99
numberFilesLabel_2=99

#### Path to Energy
# "C:/Users/../Label_"  
DATADIR_Label1 ="C:/Users/Deep Thought/Documents/Python/CNN_Masterarbeit/DSNB/DSNB_lappd5x5_Energy/DSNB_Energy_"
DATADIR_Label2 ="C:/Users/Deep Thought/Documents/Python/CNN_Masterarbeit/DSNB/Atmo_lappd5x5_Energy/Atmo_Energy_"


#### Want to check:

ElectronNumber = True # _EnergyElectron.csv
ElectronEnergy = True # _EnergyElectron.csv

MuonNumber     = True # _EnergyMuon.csv
MuonEnergy     = True # _EnergyMuon.csv

PionNumber     = True # _EnergyPion.csv
PionEnergy     = True # _EnergyPion.csv

KaonNumber     = True # _EnergyKaon.csv
KaonEnergy     = True # _EnergyKaon.csv


NeutronNumber  = True # _EnergyNeutrino.csv
VisibleEnergy  = True # _NeutronNumber.csv
ParentEnergy   = True # _VisibleEnergy.csv



#### Electron
#Number

if ElectronNumber ==True:
    all_values=np.zeros((0,2))
    for i in tqdm(range(numberFilesLabel_1)):
        try: 
            training_data_list = np.loadtxt(DATADIR_Label1+str(i)+"_EnergyElectron.csv",delimiter=',',usecols=(0,1))
            #print(training_data_list.shape,all_values.shape)

            all_values = np.concatenate([all_values,training_data_list],axis=0)
        except OSError:
            print("File {} not found".format(i))
            continue

    for i in tqdm(range(numberFilesLabel_2)):
        try:
            training_data_list = np.loadtxt(DATADIR_Label2+str(i)+"_EnergyElectron.csv",delimiter=',',usecols=(0,1))
            #print(training_data_list.shape,all_values.shape)
            all_values = np.concatenate([all_values,training_data_list],axis=0)
        except OSError:
            print("File {} not found".format(i))
            continue

    all_values=np.array(all_values)   
    np.save('Temp/NumberElectron', all_values)
    print("Finished Electron-Number")
    
#Energy
if ElectronEnergy ==True:
    all_values = []
    for i in tqdm(range(numberFilesLabel_1)):
        try: 
            training_data_file = open(DATADIR_Label1+str(i)+"_EnergyElectron.csv",'r')
            training_data_list = training_data_file.readlines()
            training_data_file.close()



            for zeile in training_data_list:

                zeile = zeile.replace(",\n", "")
                #zeile = zeile.replace("\n", "")
                zeile = zeile.split(',')
                #print(len(zeile))
                if len(zeile) <=2:
                    all_values.append(0)
                if len(zeile)>2:
                    all_values.append(zeile[2:])
        except OSError:
            print("File {} not found".format(i))
            continue


    for i in tqdm(range(numberFilesLabel_2)):
        try: 
            training_data_file = open(DATADIR_Label2+str(i)+"_EnergyElectron.csv",'r')
            training_data_list = training_data_file.readlines()
            training_data_file.close()



            for zeile in training_data_list:

                zeile = zeile.replace(",\n", "")
                #zeile = zeile.replace("\n", "")
                zeile = zeile.split(',')
                #print(len(zeile))
                if len(zeile) <=2:
                    all_values.append(0)
                if len(zeile)>2:
                    all_values.append(zeile[2:])
        except OSError:
            print("File {} not found".format(i))
            continue


    all_values=np.array(all_values)   
    np.save('Temp/EnergyElectron', all_values)
    print("Finished Electron-Energy")

###Muon
#Number

if MuonNumber ==True:
    all_values=np.zeros((0,2))

    for i in tqdm(range(numberFilesLabel_1)):
        try: 
            training_data_list = np.loadtxt(DATADIR_Label1+str(i)+"_EnergyMuon.csv",delimiter=',',usecols=(0,1))
            #print(training_data_list.shape,all_values.shape)

            all_values = np.concatenate([all_values,training_data_list],axis=0)
        except OSError:
            print("File {} not found".format(i))
            continue

    for i in tqdm(range(numberFilesLabel_2)):
        try:
            training_data_list = np.loadtxt(DATADIR_Label2+str(i)+"_EnergyMuon.csv",delimiter=',',usecols=(0,1))
            #print(training_data_list.shape,all_values.shape)
            all_values = np.concatenate([all_values,training_data_list],axis=0)
        except OSError:
            print("File {} not found".format(i))
            continue

    all_values=np.array(all_values)   
    np.save('Temp/NumberMuon', all_values)
    print("Finished Muon-Number")

#Energy
if MuonEnergy == True:
    all_values = []

    for i in tqdm(range(numberFilesLabel_1)):
        try: 
            training_data_file = open(DATADIR_Label1+str(i)+"_EnergyMuon.csv",'r')
            training_data_list = training_data_file.readlines()
            training_data_file.close()



            for zeile in training_data_list:

                zeile = zeile.replace(",\n", "")
                #zeile = zeile.replace("\n", "")
                zeile = zeile.split(',')
                #print(len(zeile))
                if len(zeile) <=2:
                    all_values.append(0)
                if len(zeile)>2:
                    all_values.append(zeile[2:])
        except OSError:
            print("File {} not found".format(i))
            continue

    for i in tqdm(range(numberFilesLabel_2)):
        try: 
            training_data_file = open(DATADIR_Label2+str(i)+"_EnergyMuon.csv",'r')
            training_data_list = training_data_file.readlines()
            training_data_file.close()



            for zeile in training_data_list:

                zeile = zeile.replace(",\n", "")
                #zeile = zeile.replace("\n", "")
                zeile = zeile.split(',')
                #print(len(zeile))
                if len(zeile) <=2:
                    all_values.append(0)
                if len(zeile)>2:
                    all_values.append(zeile[2:])
        except OSError:
            print("File {} not found".format(i))
            continue


    all_values=np.array(all_values)   
    np.save('Temp/EnergyMuon', all_values)
    print("Finished Muon-Energy")


### Pion
#Number
if PionNumber == True:

    all_values=np.zeros((0,2))

    for i in tqdm(range(numberFilesLabel_1)):
        try: 
            training_data_list = np.loadtxt(DATADIR_Label1+str(i)+"_EnergyPion.csv",delimiter=',',usecols=(0,1))
            #print(training_data_list.shape,all_values.shape)

            all_values = np.concatenate([all_values,training_data_list],axis=0)
        except OSError:
            print("File {} not found".format(i))
            continue 


    for i in tqdm(range(numberFilesLabel_2)):
        try: 
            training_data_list = np.loadtxt(DATADIR_Label2+str(i)+"_EnergyPion.csv",delimiter=',',usecols=(0,1))
            #print(training_data_list.shape,all_values.shape)

            all_values = np.concatenate([all_values,training_data_list],axis=0)
        except OSError:
            print("File {} not found".format(i))
            continue 

    all_values=np.array(all_values)   
    np.save('Temp/NumberPion', all_values)
    print("Finished Pion-Number")

#Energy
if PionEnergy == True:
    all_values = []

    for i in tqdm(range(numberFilesLabel_1)):
        try: 
            training_data_file = open(DATADIR_Label1+str(i)+"_EnergyPion.csv",'r')
            training_data_list = training_data_file.readlines()
            training_data_file.close()



            for zeile in training_data_list:

                zeile = zeile.replace(",\n", "")
                #zeile = zeile.replace("\n", "")
                zeile = zeile.split(',')
                #print(len(zeile))
                if len(zeile) <=2:
                    all_values.append(0)
                if len(zeile)>2:
                    all_values.append(zeile[2:])
        except OSError:
            print("File {} not found".format(i))
            continue


    for i in tqdm(range(numberFilesLabel_2)):
        try: 
            training_data_file = open(DATADIR_Label2+str(i)+"_EnergyPion.csv",'r')
            training_data_list = training_data_file.readlines()
            training_data_file.close()



            for zeile in training_data_list:

                zeile = zeile.replace(",\n", "")
                #zeile = zeile.replace("\n", "")
                zeile = zeile.split(',')
                #print(len(zeile))
                if len(zeile) <=2:
                    all_values.append(0)
                if len(zeile)>2:
                    all_values.append(zeile[2:])
        except OSError:
            print("File {} not found".format(i))
            continue


    all_values=np.array(all_values)   
    np.save('Temp/EnergyPion', all_values)
    print("Finished Pion-Energy")

### Kaon
#Number
if KaonNumber == True:

    all_values=np.zeros((0,2))

    for i in tqdm(range(numberFilesLabel_1)):
        try: 
            training_data_list = np.loadtxt(DATADIR_Label1+str(i)+"_EnergyKaon.csv",delimiter=',',usecols=(0,1))
            #print(training_data_list.shape,all_values.shape)

            all_values = np.concatenate([all_values,training_data_list],axis=0)
        except OSError:
            print("File {} not found".format(i))
            continue
    for i in tqdm(range(numberFilesLabel_2)):
        try: 
            training_data_list = np.loadtxt(DATADIR_Label2+str(i)+"_EnergyKaon.csv",delimiter=',',usecols=(0,1))
            #print(training_data_list.shape,all_values.shape)

            all_values = np.concatenate([all_values,training_data_list],axis=0)
        except OSError:
            print("File {} not found".format(i))
            continue 
    all_values=np.array(all_values)   
    np.save('Temp/NumberKaon', all_values)
    print("Finished Kaon-Number")

#Energy
if KaonEnergy == True:
    
    all_values = []

    for i in tqdm(range(numberFilesLabel_1)):
        try: 
            training_data_file = open(DATADIR_Label1+str(i)+"_EnergyKaon.csv",'r')
            training_data_list = training_data_file.readlines()
            training_data_file.close()



            for zeile in training_data_list:

                zeile = zeile.replace(",\n", "")
                #zeile = zeile.replace("\n", "")
                zeile = zeile.split(',')
                #print(len(zeile))
                if len(zeile) <=2:
                    all_values.append(0)
                if len(zeile)>2:
                    all_values.append(zeile[2:])
        except OSError:
            print("File {} not found".format(i))
            continue
    for i in tqdm(range(numberFilesLabel_2)):
        try: 
            training_data_file = open(DATADIR_Label2+str(i)+"_EnergyKaon.csv",'r')
            training_data_list = training_data_file.readlines()
            training_data_file.close()



            for zeile in training_data_list:

                zeile = zeile.replace(",\n", "")
                #zeile = zeile.replace("\n", "")
                zeile = zeile.split(',')
                #print(len(zeile))
                if len(zeile) <=2:
                    all_values.append(0)
                if len(zeile)>2:
                    all_values.append(zeile[2:])
        except OSError:
            print("File {} not found".format(i))
            continue

    all_values=np.array(all_values)   
    np.save('Temp/EnergyKaon', all_values)
    print("Finished Kaon-Energy")

#Parent Energy

if ParentEnergy == True:
    all_values=np.zeros((0))

    for i in tqdm(range(numberFilesLabel_1)):
        try: 
            training_data_list = np.loadtxt(DATADIR_Label1+str(i)+"_EnergyNeutrino.csv",delimiter=',',usecols=(0))
            #print(training_data_list.shape,all_values.shape)

            all_values = np.concatenate([all_values,training_data_list],axis=0)
        except OSError:
            print("File {} not found".format(i))
            continue    

    for i in tqdm(range(numberFilesLabel_2)):
        try:
            #print(i)
            training_data_list = np.loadtxt(DATADIR_Label2+str(i)+"_EnergyNeutrino.csv",delimiter=',',usecols=(0))
            #print(training_data_list.shape,all_values.shape)

            all_values = np.concatenate([all_values,training_data_list],axis=0)
        except OSError:
            print("File {} not found".format(i))
            continue 

    all_values=np.array(all_values)   
    np.save('Temp/EnergyMother', all_values)
    print("Finished Parent Energy")

### Neutron Counting
if NeutronNumber == True:
    all_values=np.zeros((0))

    for i in tqdm(range(numberFilesLabel_1)):
        try: 
            training_data_list = np.loadtxt(DATADIR_Label1+str(i)+"_NeutronNumber.csv",delimiter=',',usecols=(0))
            #print(training_data_list.shape,all_values.shape)

            all_values = np.concatenate([all_values,training_data_list],axis=0)
        except OSError:
            print("File {} not found".format(i))
            continue

    for i in tqdm(range(numberFilesLabel_2)):
        try: 
            training_data_list = np.loadtxt(DATADIR_Label2+str(i)+"_NeutronNumber.csv",delimiter=',',usecols=(0))
            #print(training_data_list.shape,all_values.shape)

            all_values = np.concatenate([all_values,training_data_list],axis=0)
        except OSError:
            print("File {} not found".format(i))
            continue 

    all_values=np.array(all_values)   
    np.save('Temp/NN', all_values)
    print("Finished Neutron Counting")

### Visible Energy
if VisibleEnergy == True:
    all_values=np.zeros((0,2))

    for i in tqdm(range(numberFilesLabel_1)):
        try: 
            training_data_list = np.loadtxt(DATADIR_Label1+str(i)+"_VisibleEnergy.csv",delimiter=',',usecols=(0,1))
            #print(training_data_list.shape,all_values.shape)

            all_values = np.concatenate([all_values,training_data_list],axis=0)
        except OSError:
            print("File {} not found".format(i))
            continue    

    for i in tqdm(range(numberFilesLabel_2)):
        try:

            training_data_list = np.loadtxt(DATADIR_Label2+str(i)+"_VisibleEnergy.csv",delimiter=',',usecols=(0,1))
            #print(training_data_list.shape,all_values.shape)

            all_values = np.concatenate([all_values,training_data_list],axis=0)
        except OSError:
            print("File {} not found".format(i))
            continue 

    all_values=np.array(all_values)   
    np.save('Temp/VisibleEnergy', all_values)
    print("Finished Visible Energy")

