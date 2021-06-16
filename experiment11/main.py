import torch
import random
import os
import numpy as np
from config import Config

from src.controller import Controller

if __name__ == '__main__':
    config = Config()
    
    seed = config.seed
    random.seed(seed)        
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.empty_cache()   
    
    
    controller = Controller(config)    
    controller.train()
    
    print("DONE")