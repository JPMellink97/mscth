import os
from scipy.io import loadmat, savemat


def load_data(pc_path=None, set=None):
    try:
        if pc_path==None:
            dir = r"C:\Users\Joost\OneDrive - TU Eindhoven\Documents\Master\thesis\mscth\data\\"
        else:
            dir = pc_path

        path = dir+set
        
        x_data = loadmat(os.path.join(path, set+"_x_data.mat"))
        x_data = x_data['x']

        u_data = loadmat(os.path.join(path, set+'_u_data.mat'))
        u_data = u_data['u']

        y_data = loadmat(os.path.join(path, set+'_y_data.mat'))
        y_data = y_data['y']

        eq_data = loadmat(os.path.join(path, set+'_eq_data.mat'))
        eq_data = eq_data['eq']

        T_path = os.path.join(path, set+'_T_data.mat')
        if os.path.exists(T_path):
            T_data = loadmat(T_path)
            T_data = T_data['T']
        else:
            T_data = None

        sU_path = os.path.join(path, set+'_sU_data.mat')
        if os.path.exists(sU_path):
            sU_data = loadmat(sU_path)
            sU_data = sU_data['sU']
        else:
            sU_data = None

        set_idx_path = os.path.join(path, set+'_set_idx_data.mat')
        if os.path.exists(set_idx_path):
            set_idx_data = loadmat(set_idx_path)
            set_idx_data = set_idx_data['set_idx']
        else:
            set_idx_data = None 
        
        print(f"Data loaded from {set}!")

        return x_data, u_data, y_data, eq_data, T_data, sU_data, set_idx_data
    except:
        print("failed?")
        return

def save_data( set, u, x, Ts, eq, set_idx=None, y=None, T=None, sU=None, data_dir="data\\"):
    try:
        if y is None:
            y = x[:,0]

        # storing in dicts
        x_mdic  = {"x" :  x}
        y_mdic  = {"y" :  y}
        u_mdic  = {"u" :  u}
        Ts_mdic = {"Ts": Ts}
        eq_mdic = {"eq": eq}

        directory = data_dir+set
        os.makedirs(directory, exist_ok=True)

        savemat(directory+"\\"+set+"_x_data.mat",  x_mdic)
        savemat(directory+"\\"+set+"_y_data.mat",  y_mdic)
        savemat(directory+"\\"+set+"_u_data.mat",  u_mdic)
        savemat(directory+"\\"+set+"_Ts_data.mat",Ts_mdic)
        savemat(directory+"\\"+set+"_eq_data.mat", eq_mdic)

        if T is not None:
            T_mdic  = {"T" :  T}
            savemat(directory+"\\"+set+"_T_data.mat",  T_mdic)

        if sU is not None:
            sU_mdic  = {"sU" :  sU}
            savemat(directory+"\\"+set+"_sU_data.mat",  sU_mdic)

        if set_idx is not None:
            set_idx_mdic  = {"set_idx" :  set_idx}
            savemat(directory+"\\"+set+"_set_idx_data.mat",  set_idx_mdic)

        print(f"Saved succesfully to {set}!")
    except:
        print("Unsuccesful")
    return