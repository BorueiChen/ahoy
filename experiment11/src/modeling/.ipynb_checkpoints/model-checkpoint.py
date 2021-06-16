import torch
import numpy as np

from torch import nn
from torch.optim import Adam, SGD, RMSprop
from torch.nn.utils.clip_grad import clip_grad_norm_

from .baseline import PointerGenerator
from .proposed import TemplateDistributionDecoder


class DataToTextModel(object):
    """
    attributes:
        * model:class
        * optimizer: 
        * scheduler: TODO
    methods:
        * train_step: 
        * valid_step:
        * test_step:
    """
    def __init__(self, config):
        # high level control conditions
        self.is_template_data = config.is_template_data
        self.is_description_data = config.is_description_data
        self.baseline = config.baseline
        self.template_distribution_decoder = config.template_distribution_decoder
        self.is_bce_loss = config.is_bce_loss
        self.is_coverage_loss = config.is_coverage_loss
        
        # main atrributes
        self.device = config.device
        if self.baseline: self.model = PointerGenerator(config).to(self.device)
        elif self.template_distribution_decoder: self.model = TemplateDistributionDecoder(config).to(self.device)
            
        def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'The model has {count_parameters(self.model):,} trainable parameters')
        
        self.optimizer = self._get_optimizer(config)
        self.max_grad_norm = config.max_grad_norm

    def _get_optimizer(self, config):
        if config.optimizer == 'Adam':
            optimizer = Adam(self.model.parameters(), lr=config.lr)
        elif config.optimizer == 'SGD':
            optimizer = SGD(self.model.parameters(), lr=config.lr, momentum=config.momentum)        
        elif config.optimizer == 'RMSprop':
            optimizer = RMSprop(self.model.parameters(), lr=config.lr, momentum=config.momentum)

        return optimizer
    def save(self, path, iterations):
        torch.save({'state_dict':self.model.state_dict(), 'iterations':iterations}, path)        
        return
    
    def load(self, path):
        load_dict = torch.load(path)
        self.model.load_state_dict(load_dict['state_dict'])            
        return load_dict['iterations']
    
    def train_step(self, batch):   
        #self.optimizer.zero_grad()   
        for param in self.model.parameters(): param.grad = None
        
        if self.is_description_data and not self.is_template_data:
            src_k = batch['src_k'].to(self.device)
            src_v = batch['src_v'].to(self.device)
            src_lengths = batch['src_lengths']
            tgt = batch['tgt'].to(self.device)
            tgt_lengths = batch['tgt_lengths']
            alignment = batch['alignment'].to(self.device)     
      
            out_dict = self.model.train_single(src_k, src_v, src_lengths, tgt, tgt_lengths, alignment)
  
        elif not self.is_description_data and self.is_template_data:
            src_k = batch['src_k'].to(self.device)
            src_v = batch['src_v'].to(self.device)
            src_lengths = batch['src_lengths']
            template = batch['template'].to(self.device)
            template_lengths = batch['template_lengths']
            alignment = batch['alignment'].to(self.device)      
      
            out_dict = self.model.train_single(src_k, src_v, src_lengths, template, template_lengths, alignment)
    
        elif self.is_template_data and self.is_description_data and self.template_distribution_decoder: 
            """ Only in the TemplateDistributionDecoder and accepting two dataset, this branch will be started."""
            src_k = batch['src_k'].to(self.device)
            src_v = batch['src_v'].to(self.device)
            src_lengths = batch['src_lengths']
            tgt = batch['tgt'].to(self.device)
            tgt_lengths = batch['tgt_lengths']
            template = batch['template'].to(self.device)
            template_lengths = batch['template_lengths']
            alignment = batch['alignment'].to(self.device)    

            out_dict = self.model.train_pair(src_k, src_v, src_lengths, tgt, tgt_lengths, template, template_lengths, alignment)

        out_dict['total_loss'].backward()
        out_dict['total_loss'] = out_dict['total_loss'].item() # just for logging
        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()                 
        
        return out_dict
    
    def valid_step(self, batch):           
        with torch.no_grad():    
            if self.is_description_data and not self.is_template_data:
                src_k = batch['src_k'].to(self.device)
                src_v = batch['src_v'].to(self.device)
                src_lengths = batch['src_lengths']
                tgt = batch['tgt'].to(self.device)
                tgt_lengths = batch['tgt_lengths']
                alignment = batch['alignment'].to(self.device)        

                out_dict = self.model.valid_single(src_k, src_v, src_lengths, tgt, tgt_lengths, alignment)

            elif not self.is_description_data and self.is_template_data:
                src_k = batch['src_k'].to(self.device)
                src_v = batch['src_v'].to(self.device)
                src_lengths = batch['src_lengths']
                template = batch['template'].to(self.device)
                template_lengths = batch['template_lengths']
                alignment = batch['alignment'].to(self.device)    

                out_dict = self.model.valid_single(src_k, src_v, src_lengths, template, template_lengths, alignment)

            elif self.is_template_data and self.is_description_data and self.template_distribution_decoder: #TODO
                src_k = batch['src_k'].to(self.device)
                src_v = batch['src_v'].to(self.device)
                src_lengths = batch['src_lengths']
                tgt = batch['tgt'].to(self.device)
                tgt_lengths = batch['tgt_lengths']
                template = batch['template'].to(self.device)
                template_lengths = batch['template_lengths']
                alignment = batch['alignment'].to(self.device)     

                out_dict = self.model.valid_pair(src_k, src_v, src_lengths, tgt, tgt_lengths, template, template_lengths, alignment)

            out_dict['total_loss'] = out_dict['total_loss'].item() # just for logging
        
        return out_dict
    
    def test_step(self, batch): # beam search TODO
        """
        predictions: [batch, max_tlen]
        attetion_maps: [batch, max_tlen, max_slen]
        p_gens: [batch, max_tlen]
        """        
        src_k = batch['src_k'].to(self.device)
        src_v = batch['src_v'].to(self.device)
        src_lengths = batch['src_lengths']
        max_tlen = batch['max_tlen']
        
        with torch.no_grad():
            out_dict = self.model.inference(src_k, src_v, src_lengths, max_tlen)
            
        return out_dict        