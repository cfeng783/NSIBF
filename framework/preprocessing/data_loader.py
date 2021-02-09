import pandas as pd
import numpy as np
import random
import math
import zipfile
from .signals import ContinousSignal,DiscreteSignal,SignalSource


def _process_model(u, t, noise_std=0.1):
    s = math.sin(t/u) + np.random.normal(0,noise_std)
    return s

def _measure_model(s, noise_std=0.2):
    x = s*2 + np.random.normal(0,noise_std)
    return x

def get_simulation_data():
    """
    get simulation data for qualitative experiment
    """
    np.random.seed(123)
    random.seed(1234)
    
    T = 10000
    L = 30
    ulist = []
    xlist = []
    u = 3
    for t in range(T):        
        if t%L==0:
            u = 9-u
           
        s = _process_model(u,t)
        x = _measure_model(s)
        ulist.append(u)
        xlist.append(x)
    
    train_dict = {'x':xlist,'u':ulist}
    train_df = pd.DataFrame(data=train_dict)
    
    N = 10000
    ulist = []
    xlist = []
    labels = []
    slist = []
    for t in range(T,T+N):
        if t%L==0:
            u = 9-u
        
        if t%1000>100 and t%1000 < 200:
            s = _process_model(u,t,noise_std=0.6)
            x = _measure_model(s)
            l = 1
        else:
            s = _process_model(u,t)
            x = _measure_model(s)
            l = 0
            
        ulist.append(u)
        xlist.append(x)
        slist.append(s)
        labels.append(l)
    
    test_dict = {'x':xlist,'u':ulist,'s':slist,'label':labels}
    test_df = pd.DataFrame(data=test_dict)

    signals = []
    signals.append(ContinousSignal('x', SignalSource.sensor, isInput=True, isOutput=True, 
                                                    min_value=train_df['x'].min(), max_value=train_df['x'].max(),
                                                    mean_value=train_df['x'].mean(), std_value=train_df['x'].std()))
    signals.append( DiscreteSignal('u', SignalSource.controller, isInput=True, isOutput=False, 
                                                values=train_df['u'].unique()) )
    
    return train_df,test_df,signals
    
def load_wadi_data():
    z_tr = zipfile.ZipFile('../datasets/WADI/WADI_train.zip', "r")
    f_tr = z_tr.open(z_tr.namelist()[0])
    train_df=pd.read_csv(f_tr)
    f_tr.close()
    z_tr.close()
    
    z_tr = zipfile.ZipFile('../datasets/WADI/WADI_test.zip', "r")
    f_tr = z_tr.open(z_tr.namelist()[0])
    test_df=pd.read_csv(f_tr)
    f_tr.close()
    z_tr.close()
    
    train_df=train_df.fillna(method='ffill')
    test_df.loc[test_df['label']>=1,'label']=1
    test_df=test_df.fillna(method='ffill')
    
    sensors = ['1_AIT_001_PV', '1_AIT_002_PV', '1_AIT_003_PV', '1_AIT_004_PV', 
               '1_AIT_005_PV', '1_FIT_001_PV', '1_LT_001_PV', '2_DPIT_001_PV', 
               '2_FIC_101_CO', '2_FIC_101_PV', '2_FIC_101_SP', '2_FIC_201_CO', 
               '2_FIC_201_PV', '2_FIC_201_SP', '2_FIC_301_CO', '2_FIC_301_PV', 
               '2_FIC_301_SP', '2_FIC_401_CO', '2_FIC_401_PV', '2_FIC_401_SP', 
               '2_FIC_501_CO', '2_FIC_501_PV', '2_FIC_501_SP', '2_FIC_601_CO', 
               '2_FIC_601_PV', '2_FIC_601_SP', '2_FIT_001_PV', '2_FIT_002_PV', 
               '2_FIT_003_PV', '2_FQ_101_PV', '2_FQ_201_PV', '2_FQ_301_PV', '2_FQ_401_PV', 
               '2_FQ_501_PV', '2_FQ_601_PV', '2_LT_001_PV', '2_LT_002_PV', '2_MCV_101_CO', 
               '2_MCV_201_CO', '2_MCV_301_CO', '2_MCV_401_CO', '2_MCV_501_CO', '2_MCV_601_CO', 
               '2_P_003_SPEED', '2_P_004_SPEED', '2_PIC_003_CO', '2_PIC_003_PV', '2_PIT_001_PV', 
               '2_PIT_002_PV', '2_PIT_003_PV', '2A_AIT_001_PV', '2A_AIT_002_PV', '2A_AIT_003_PV', 
               '2A_AIT_004_PV', '2B_AIT_001_PV', '2B_AIT_002_PV', '2B_AIT_003_PV', '2B_AIT_004_PV', 
               '3_AIT_001_PV', '3_AIT_002_PV', '3_AIT_003_PV', '3_AIT_004_PV', '3_AIT_005_PV', 
               '3_FIT_001_PV', '3_LT_001_PV', 'LEAK_DIFF_PRESSURE', 'TOTAL_CONS_REQUIRED_FLOW']


    actuators = ['1_MV_001_STATUS', '1_MV_004_STATUS', '1_P_001_STATUS', '1_P_003_STATUS', 
                 '1_P_005_STATUS', '2_LS_101_AH', '2_LS_101_AL', '2_LS_201_AH', '2_LS_201_AL', 
                 '2_LS_301_AH', '2_LS_301_AL', '2_LS_401_AH', '2_LS_401_AL', '2_LS_501_AH', 
                 '2_LS_501_AL', '2_LS_601_AH', '2_LS_601_AL', '2_MV_003_STATUS', '2_MV_006_STATUS',
                 '2_MV_101_STATUS', '2_MV_201_STATUS', '2_MV_301_STATUS', '2_MV_401_STATUS', 
                 '2_MV_501_STATUS', '2_MV_601_STATUS', '2_P_003_STATUS']

    signals = []
    for name in sensors:
        signals.append( ContinousSignal(name, SignalSource.sensor, isInput=True, isOutput=True, 
                                        min_value=train_df[name].min(), max_value=train_df[name].max(),
                                        mean_value=train_df[name].mean(), std_value=train_df[name].std()) )
    for name in actuators:
        signals.append( DiscreteSignal(name, SignalSource.controller, isInput=True, isOutput=False, 
                                            values=train_df[name].unique()) )
    
    
    pos = len(train_df)*3//4
    val_df = train_df.loc[pos:,:]
    val_df = val_df.reset_index(drop=True)
    
    train_df = train_df.loc[:pos,:]
    train_df = train_df.reset_index(drop=True)
    return train_df,val_df,test_df,signals


def load_swat_data():
    z_tr = zipfile.ZipFile('../datasets/SWAT/SWaT_train.zip', "r")
    f_tr = z_tr.open(z_tr.namelist()[0])
    train_df=pd.read_csv(f_tr)
    f_tr.close()
    z_tr.close()
    
    z_tr = zipfile.ZipFile('../datasets/SWAT/SWaT_test.zip', "r")
    f_tr = z_tr.open(z_tr.namelist()[0])
    test_df=pd.read_csv(f_tr)
    f_tr.close()
    z_tr.close()
    
    test_df['label'] = 0
    test_df.loc[test_df['Normal/Attack']!='Normal', 'label'] = 1
    
    sensors = ['FIT101','LIT101','AIT201','AIT202','AIT203','FIT201',
           'DPIT301','FIT301','LIT301','AIT401','AIT402','FIT401',
           'LIT401','AIT501','AIT502','AIT503','AIT504','FIT501',
           'FIT502','FIT503','FIT504','PIT501','PIT502','PIT503','FIT601',]

    actuators = ['MV101','P101','P102','MV201','P201','P202',
                   'P203','P204','P205','P206','MV301','MV302',
                   'MV303','MV304','P301','P302','P401','P402',
                   'P403','P404','UV401','P501','P502','P601',
                   'P602','P603']
    signals = []
    for name in sensors:
        signals.append( ContinousSignal(name, SignalSource.sensor, isInput=True, isOutput=True, 
                                        min_value=train_df[name].min(), max_value=train_df[name].max(),
                                        mean_value=train_df[name].mean(), std_value=train_df[name].std()) )
    for name in actuators:
        signals.append( DiscreteSignal(name, SignalSource.controller, isInput=True, isOutput=False, 
                                            values=train_df[name].unique()) )
    
    pos = len(train_df)*3//4
    val_df = train_df.loc[pos:,:]
    val_df = val_df.reset_index(drop=True)
    
    train_df = train_df.loc[:pos,:]
    train_df = train_df.reset_index(drop=True)
    return train_df,val_df,test_df,signals

def load_pump_data():
    
    z_tr = zipfile.ZipFile('../datasets/PUMP/PUMP_train.zip', "r")
    f_tr = z_tr.open(z_tr.namelist()[0])
    train_df=pd.read_csv(f_tr)
    f_tr.close()
    z_tr.close()
    
    z_tr = zipfile.ZipFile('../datasets/PUMP/PUMP_test.zip', "r")
    f_tr = z_tr.open(z_tr.namelist()[0])
    test_df=pd.read_csv(f_tr)
    f_tr.close()
    z_tr.close()
    
    signals = []
    for name in train_df:
        if name.startswith('sensor'): 
            signals.append( ContinousSignal(name, SignalSource.sensor, isInput=True, isOutput=True, 
                                            min_value=train_df[name].min(), max_value=train_df[name].max(),
                                            mean_value=train_df[name].mean(), std_value=train_df[name].std() ) ) 
 
    
    pos = len(train_df)*3//4
    val_df = train_df.loc[pos:,:]
    val_df = val_df.reset_index(drop=True)
    
    train_df = train_df.loc[:pos,:]
    train_df = train_df.reset_index(drop=True)
    return train_df,val_df,test_df,signals
