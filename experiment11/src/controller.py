import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
import torch
import logging
import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from .modeling.model import DataToTextModel
from .dataset import Rotowire

# https://github.com/PacktPublishing/Hands-On-TensorBoard-for-PyTorch-Developers


class Controller(object):
    """
    atrributes:
        * dataset: class
        * model: class
        * evaluation: class
        * logger: class
    methods:
        * train: void 
        ## loop over the training set, get a batch of data, train the model,
        log the loss and other metrics.
        After each epoch it calls the validate() function, check if the model has
        improved, and store the model if it is.
        
        * validate: void 
        ## loop over the dev set, get a batch of data, get the
        predictions, log the metrics.
        
        * test: void
        ## loop over the test set, get a batch of data, get the
        predictions, log the metrics.
        
    """
    def __init__(self, config):
        # high level control conditions
        self.is_template_data = config.is_template_data
        self.is_description_data = config.is_description_data
        self.baseline = config.baseline
        self.template_distribution_decoder = config.template_distribution_decoder
        
        # create dataset and model
        self.dataset = Rotowire(config)
        self.name = config.name
        config.vocab_size = self.dataset.vocab_size
        self.model = DataToTextModel(config)
        
        ## add tensorboard
        self.tb = SummaryWriter(comment="_"+config.name)
        logging.basicConfig(filename='./experiments/'+config.name+'/logs/'+str(datetime.datetime.now())+'.log', level=logging.DEBUG)
            
        # train or infer initialization
        self.max_iterations = config.max_iterations
        self.interval = config.interval
        self.batch_size = config.batch_size   
        self.scheduler_name = config.scheduler
        self.scheduler = self._get_scheduler(self.model.optimizer, config)   
        self.max_tlen = config.max_tlen
        
        # laod model
        self.save_path = './experiments/'+config.name
        self.start_iteration = 0
        try:
            if config.load: self.start_iteration = self.model.load(config.load)   
        except:
            logging.info("Initailize new model...")
            print("Initailize new model...")
        self.hyperparameter = config.__dict__
            
    def _get_scheduler(self, optimizer, config):
        if config.scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=50, verbose=True)
        elif config.scheduler == 'CyclicLR':
            scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.01) 
        else: 
            scheduler = None
            
        return scheduler
    
    def train(self):
        logging.info("Start training")
        best_valid_loss, train_loss, valid_loss, valid_bleu, test_bleu = 1e12, None, None, None, None       
        # start training
        with trange(self.max_iterations, desc='Training stage:', unit="ahoy!!!") as pbar:
            for iterations in pbar:   
                iterations += self.start_iteration #offset
                # feed batch
                batch = next(iter(self.dataset.trainloader))
                out_dict = self.model.train_step(batch)
                train_loss = out_dict['total_loss']
                # logging
                pbar.set_description(f"Iterations:[{iterations}], Training loss(batch): {train_loss}")
                out_dict['lr']=self.scheduler.get_last_lr()[0]
                out_dict['iterations']=iterations
                self.logging_w_tensorboard("train",out_dict)   
                # scheduler
                if self.scheduler_name == 'ReduceLROnPlateau': self.scheduler.step(train_loss)
                if self.scheduler_name == 'CyclicLR': self.scheduler.step()                                                        
                                
                # valid and test
                if iterations%self.interval == 0:   
                    valid_loss, valid_bleu, test_bleu = self.valid_and_test(iterations)
                    # model save
                    if valid_loss < best_valid_loss:
                        logging.info(f"Model saving... valid loss drops from {best_valid_loss} to {valid_loss}")
                        best_valid_loss = valid_loss
                        self.model.save(self.save_path+'/models/'+'model.pt', iterations)



        # final validate and test        
        valid_loss, valid_bleu, test_bleu = self.valid_and_test(iterations+1)
        self.tb.add_hparams(self.hyperparameter, {'Valid Loss(iteration)': valid_loss, 'BLEU (valid)': valid_bleu, 'BLEU (test)': test_bleu})
        
        return
    
    def _get_bleu(self, file_path, name):
        
        if self.is_description_data and not self.is_template_data:       
            ## calculate bleu
            if name == "valid":
                result = subprocess.Popen('cat '+file_path+' | sacrebleu --force ./data/valid_output.txt', shell=True, stdout=subprocess.PIPE)  
            elif name == "test":
                result = subprocess.Popen('cat '+file_path+' | sacrebleu --force ./data/test_output.txt', shell=True, stdout=subprocess.PIPE)

            result = result.stdout.read()
            bleu = float(result.split()[2])      
            return bleu
        elif not self.is_description_data and self.is_template_data:
            ## calculate template bleu
            if name == "valid":
                result = subprocess.Popen('cat '+file_path+' | sacrebleu --force ./experiments/'+self.name+'/data/valid_template.txt', shell=True, stdout=subprocess.PIPE)  
            elif name == "test":
                result = subprocess.Popen('cat '+file_path+' | sacrebleu --force ./experiments/'+self.name+'/data/test_template.txt', shell=True, stdout=subprocess.PIPE)

            result = result.stdout.read()
            bleu = float(result.split()[2])      
            return bleu
        elif self.is_description_data and self.is_template_data:       
            ## calculate bleu
            if name == "valid":
                result = subprocess.Popen('cat '+file_path+' | sacrebleu --force ./data/valid_output.txt', shell=True, stdout=subprocess.PIPE)  
            elif name == "test":
                result = subprocess.Popen('cat '+file_path+' | sacrebleu --force ./data/test_output.txt', shell=True, stdout=subprocess.PIPE)

            result = result.stdout.read()
            bleu = float(result.split()[2])        
            
            ## calculate template bleu
            if name == "valid":
                result = subprocess.Popen('cat '+file_path+' | sacrebleu --force ./experiments/'+self.name+'/data/valid_template.txt', shell=True, stdout=subprocess.PIPE)  
            elif name == "test":
                result = subprocess.Popen('cat '+file_path+' | sacrebleu --force ./experiments/'+self.name+'/data/test_template.txt', shell=True, stdout=subprocess.PIPE)

            result = result.stdout.read()
            template_bleu = float(result.split()[2])                              
            return bleu, template_bleu

    def valid_and_test(self, iterations): 
        logging.info("Start validating")
        valid_loss, valid_bleu = self.validate(iterations)   
        logging.info("Start testing")
        test_bleu = self.test(iterations)       
        
        return valid_loss, valid_bleu, test_bleu
    
    def validate(self, iterations):
        file_path = self.save_path+'/outputs/valid/'+'prediction_valid_'+str(iterations)+'.txt'
        with open(file_path, 'w') as file:
            valid_loss, valid_bce_loss, valid_loss_std, valid_kl_loss, valid_template_loss = 0.0, 0.0, 0.0, 0.0,0.0
            pbar = tqdm(self.dataset.validloader)
            num_batches = len(pbar)
            for idx, batch in enumerate(pbar):    
                # feed batch
                out_dict = self.model.valid_step(batch)                   
                valid_loss += out_dict['total_loss']    
                valid_bce_loss += out_dict['bce_loss']    
                valid_loss_std += out_dict['total_loss_std']    
                if self.template_distribution_decoder:
                    valid_kl_loss+= out_dict['kl_loss']
                    valid_template_loss+= out_dict['template_loss']
                # post process
                out_dict['tgt_lengths']=batch['tgt_lengths']
                out_dict = self.dataset.post_process(out_dict, name="valid")                                        
                # write file
                predictions = out_dict['post_process']
                for line in predictions: file.write(" ".join(line)+'\n')
                # logging
                pbar.set_description(f"Iterations:[{iterations}]: Valid loss(batch): {out_dict['total_loss']}, Valid loss(iterations): {valid_loss/(idx+1)}")
            
        # average the valid loss
        valid_loss/=num_batches
        valid_bce_loss/=num_batches
        valid_loss_std/=num_batches
        valid_kl_loss/=num_batches
        valid_template_loss/=num_batches
        
        ## calculate bleu
        bleu = self._get_bleu(file_path, "valid")
        ## logging into tensorboard
        logging_dict={'total_loss':valid_loss, 
                      'bce_loss':valid_bce_loss, 
                      'total_loss_std':valid_loss_std,
                      'kl_loss':valid_kl_loss,
                      'template_loss':valid_template_loss,
                      'bleu':bleu, 
                      'iterations':iterations}
        if self.template_distribution_decoder and self.is_description_data and self.is_template_data:
            logging_dict['bleu']=bleu[0]
            logging_dict['template_bleu']=bleu[1]
        self.logging_w_tensorboard("valid",logging_dict)
                        
        return valid_loss, bleu
    
    def test(self, iterations):  
        file_path = self.save_path+'/outputs/test/'+'prediction_test_'+str(iterations)+'.txt'
        with open(file_path, 'w') as file:
            pbar = tqdm(self.dataset.testloader, desc='Testing inference:', unit="ahoy!!!")
            for idx, batch in enumerate(pbar): 
                # feed batch
                batch['max_tlen'] = self.max_tlen
                out_dict = self.model.test_step(batch)
                # post process
                out_dict = self.dataset.post_process(out_dict)
                # write file
                predictions = out_dict['post_process']
                for line in predictions: file.write(" ".join(line)+'\n')
        
        ## calculate bleu
        bleu = self._get_bleu(file_path, "test")          
        ## logging
        logging_dict={'bleu':bleu, 'iterations':iterations}        

        if self.template_distribution_decoder and self.is_description_data and self.is_template_data:
            logging_dict['bleu']=bleu[0]
            logging_dict['template_bleu']=bleu[1]

        self.logging_w_tensorboard("test", logging_dict)
        
        return bleu

    
    def logging_w_tensorboard(self, name, logging_dict):        
        if name == "valid":
            out_dict=logging_dict
            self.tb.add_scalar("Valid Loss(iteration)", out_dict['total_loss'], out_dict['iterations'])
            self.tb.add_scalar("Valid BCE Loss(iteration)", out_dict['bce_loss'], out_dict['iterations'])
            self.tb.add_scalar("Valid Loss STD(iteration)", out_dict['total_loss_std'], out_dict['iterations'])
            if self.template_distribution_decoder and self.is_description_data and self.is_template_data:
                self.tb.add_scalar("BLEU template (valid)", out_dict['template_bleu'], out_dict['iterations'])
                
            self.tb.add_scalar("BLEU (valid)", out_dict['bleu'], out_dict['iterations'])
            print(f"\nIterations:[{out_dict['iterations']}] || BLEU (valid): {out_dict['bleu']}")
            logging.info(f"Iterations:[{out_dict['iterations']}] Valid loss(iterations): {out_dict['total_loss']}, BLEU: {out_dict['bleu']}")     
        
        elif name == "test":
            out_dict=logging_dict
            self.tb.add_scalar("BLEU (test)", out_dict['bleu'], out_dict['iterations'])
            if self.template_distribution_decoder and self.is_description_data and self.is_template_data:
                self.tb.add_scalar("BLEU template (valid)", out_dict['template_bleu'], out_dict['iterations'])
            print(f"\nIterations:[{out_dict['iterations']}] || BLEU (test): {out_dict['bleu']}")
            logging.info(f"Iterations:[{out_dict['iterations']}]: BLEU: {out_dict['bleu']}")       
            
        elif name == "train":
            out_dict=logging_dict
            self.tb.add_scalar("Train Loss (iterations)", out_dict['total_loss'], out_dict['iterations'])
            self.tb.add_scalar("Train Loss Std(iteration)", out_dict['total_loss_std'], out_dict['iterations'])
            self.tb.add_scalar("Train BCE Loss(iteration)", out_dict['bce_loss'], out_dict['iterations'])            
            self.tb.add_scalar("LR (iterations)", out_dict['lr'], out_dict['iterations']) 
            if self.template_distribution_decoder: 
                self.tb.add_scalar("Template Loss(iteration)", out_dict['template_loss'], out_dict['iterations'])
                self.tb.add_scalar("KL Loss(iteration)", out_dict['kl_loss'], out_dict['iterations'])
            if out_dict['iterations']%self.interval==0: logging.info(f"Iterations:[{out_dict['iterations']}], Training loss(batch): {out_dict['total_loss']}")