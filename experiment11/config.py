class Config(object):
    def __init__(self):
        
        # experiment setting
        self.name = "exp-5"
        self.copy_data = "./experiments/exp-3/data"#None#
        self.load = None#'./experiments/exp-3/models/model.pt'#
        
        # controller parameter
        self.seed = 5278
        self.device = 'cuda:2' #'cpu'#
        self.num_workers=0
        self.max_grad_norm = 1
        self.optimizer = 'SGD'# Adam, SGD, RMSprop
        self.momentum = 0.9
        self.lr = 0.01
        self.scheduler = 'CyclicLR'# None, ReduceLROnPlateau, CyclicLR
        self.max_iterations = 500000
        self.interval = 10000
        self.batch_size = 8
        self.max_tlen = 400 # for inference
        
        ## model encoder
        self.emb_size = 256
        self.hidden_size = 256
        self.attention_size = 64
        self.encoder_type = 'brnn'# 'brnn' or 'rnn'
        self.lstm_layers = 1 # 2 will be error lol
        self.dropout = 0.5
        
        ## TTD 
        self.gaussian_size = 64
        
        # vocab size
        self.vocab_size = None # modify by dataset class
        
        # high level control conditions
        ## dataset condition
        self.is_template_data = True#
        self.is_description_data = False#True
        
        ## model condition
        self.baseline = True
        self.template_distribution_decoder = False
        
        ## Additional loss
        self.is_bce_loss = False
        self.is_coverage_loss = False