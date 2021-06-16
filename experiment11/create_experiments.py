import pkg_resources
import os
import subprocess
import json
from config import Config
from src.preprocess import build as build_dataset


if __name__ == '__main__':
    args = Config()
    
    folder = pkg_resources.resource_filename(__name__, 'experiments')
        
    if not os.path.exists(folder):
        print("Creating a folder 'experiments/' where all experiments will be stored.")
        os.mkdir(folder)
    
    folder = os.path.join(folder, args.name)
    
    if os.path.exists(folder):
        raise ValueError('An experiment with this name already exists')
        
    os.mkdir(folder)
    if args.copy_data: subprocess.Popen('cp -r '+args.copy_data+' '+folder, shell=True, stdout=subprocess.PIPE)  
    else: os.mkdir(os.path.join(folder, 'data'))
    os.mkdir(os.path.join(folder, 'models'))
    os.mkdir(os.path.join(folder, 'logs'))
    os.mkdir(os.path.join(folder, 'outputs'))
    os.mkdir(os.path.join(folder, 'outputs', 'test'))
    os.mkdir(os.path.join(folder, 'outputs', 'valid'))