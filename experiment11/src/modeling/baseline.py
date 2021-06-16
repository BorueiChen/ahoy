import torch
import torch.nn as nn
import random

from .encoder import Encoder
from .decoder import Decoder

class PointerGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        # high level control conditions
        self.is_bce_loss = config.is_bce_loss
        self.is_coverage_loss = config.is_coverage_loss        
        
        self.device = config.device
        self.embedding = nn.Embedding(config.vocab_size, config.emb_size)
        self.record_linear = nn.Linear(2*config.emb_size, config.emb_size)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.zeros = torch.zeros(1, 1).to(self.device)
        self.dropout = nn.Dropout(config.dropout)
        
    def Record_Embedding_layer(self, src_k, src_v):
        src_k_embedding = self.embedding(src_k) # [batch, slen, emb_size]
        src_v_embedding = self.embedding(src_v) # [batch, slen, emb_size]
        relu = nn.ReLU()
        out = relu(self.record_linear(torch.cat((src_k_embedding, src_v_embedding), dim=2))) # [batch, slen, emb_size]
        
        return out          
        
    def train_single(self, src_k, src_v, src_lengths, tgt, tgt_lengths, alignment):
        """
        Args:
            src_k: [batch, slen]
            src_v: [batch, slen]
            src_lengths: [batch]
            tgt: [batch, tlen]
            tgt_lengths; [batch]
            alignment: [batch, tlen]
        Returns:
            loss: float
            predictions: [batch, max_tlen]
            attetion_maps: [batch, max_tlen, max_slen]
            p_gens: [batch, max_tlen]
        """
        
        batch_size = tgt.size(0)
        tlen = tgt.size(1)
        
        ## Record representation
        record_representation = self.Record_Embedding_layer(src_k, src_v) # [batch, slen, emb_size]
        
        ## Encoder
        # [batch, slen, hidden_size], [batch, hidden_size] , [batch, hidden_size]
        encoder_outputs, hidden, control = self.encoder(record_representation, src_lengths) 
               
        ## Decoder    
        bos = torch.ones_like(tgt[:,0]).to(self.device) # bos
        decoder_inputs = torch.cat((bos.unsqueeze(1), tgt[:, :-1]), dim=1) # [batch, tlen]
        decoder_inputs = self.dropout(self.embedding(decoder_inputs)) # [batch, tlen ,emb_size]
        p_vocab, p_copy, p_gen, attention_map = self.decoder(decoder_inputs, tgt_lengths, encoder_outputs, hidden, control)
        
        ## calculate description loss    
        likelihood = torch.gather(p_vocab, 2, tgt.unsqueeze(2)) + torch.gather(torch.cat((p_copy, torch.zeros(batch_size, tlen, 1).to(self.device)), dim=2), 2 , alignment.unsqueeze(2))        
        mask = tgt!=0      
        likelihood_mean, likelihood_std = likelihood.log().masked_select(mask.unsqueeze(2)).mean(), likelihood.log().masked_select(mask.unsqueeze(2)).std()
        total_loss = -likelihood_mean
        total_loss_std = -likelihood_std
        
        ## calculate bce loss
        _BCE_LOSS = nn.BCELoss()
        bce_target = (alignment != alignment.max()).float()
        bce_loss = _BCE_LOSS(p_gen.squeeze(2), bce_target)
        if self.is_bce_loss: total_loss+=bce_loss      
       
        
        return {'total_loss':total_loss, 
                'total_loss_std':total_loss_std.item(), 
                'bce_loss': bce_loss.item(),
               }
    
    def valid_single(self, src_k, src_v, src_lengths, tgt, tgt_lengths, alignment):
        """
        Args:
            src_k: [batch, slen]
            src_v: [batch, slen]
            src_lengths: [batch]
            tgt: [batch, tlen]
            tgt_lengths; [batch]
            alignment: [batch, tlen]
        Returns:
            loss: float
            predictions: [batch, max_tlen]
            attetion_maps: [batch, max_tlen, max_slen]
            p_gens: [batch, max_tlen]
        """
        
        batch_size = tgt.size(0)
        tlen = tgt.size(1)
        
        ## Record representation
        record_representation = self.Record_Embedding_layer(src_k, src_v) # [batch, slen, emb_size]
        
        ## Encoder
        # [batch, slen, hidden_size], [batch, hidden_size] , [batch, hidden_size]
        encoder_outputs, hidden, control = self.encoder(record_representation, src_lengths) 
               
        ## Decoder    
        bos = torch.ones_like(tgt[:,0]).to(self.device) # bos
        decoder_inputs = torch.cat((bos.unsqueeze(1), tgt[:, :-1]), dim=1) # [batch, tlen]
        decoder_inputs = self.dropout(self.embedding(decoder_inputs)) # [batch, tlen ,emb_size]
        p_vocab, p_copy, p_gen, attention_map = self.decoder(decoder_inputs, tgt_lengths, encoder_outputs, hidden, control)
        
        ## calculate description loss     
        likelihood = torch.gather(p_vocab, 2, tgt.unsqueeze(2)) + torch.gather(torch.cat((p_copy, torch.zeros(batch_size, tlen, 1).to(self.device)), dim=2), 2 , alignment.unsqueeze(2))        
        mask = tgt!=0      
        likelihood_mean, likelihood_std = likelihood.log().masked_select(mask.unsqueeze(2)).mean(), likelihood.log().masked_select(mask.unsqueeze(2)).std()
        total_loss = -likelihood_mean
        total_loss_std = -likelihood_std
        
        ## calculate bce loss
        _BCE_LOSS = nn.BCELoss()
        bce_target = (alignment != alignment.max()).float()
        bce_loss = _BCE_LOSS(p_gen.squeeze(2), bce_target)
        if self.is_bce_loss: total_loss+=bce_loss
        
        ## predictions
        p_vocab_final = p_vocab.scatter_add(2, src_v.unsqueeze(1).repeat(1, tlen, 1), p_copy)            
        predictions = p_vocab_final.max(2)[1]        

        
        
        return {'total_loss':total_loss, 
                'total_loss_std':total_loss_std.item(), 
                'bce_loss': bce_loss.item(),
                'predictions':predictions.clone().detach().cpu().numpy(), 
                'attetion_maps':attention_map.clone().detach().cpu().numpy(), 
                'p_gens':p_gen.squeeze(2).clone().detach().cpu().numpy(),
               }
    
    def inference(self, src_k, src_v, src_lengths, max_tlen):
        """
        Args:
            src_k: [batch, slen]
            src_v: [batch, slen]
            src_lengths: [batch]
        Return:
            predictions: [batch, max_tlen]
            attetion_maps: [batch, max_tlen, max_slen]
            p_gens: [batch, max_tlen]
        """
        
        batch_size = src_v.size(0)
        max_slen = src_v.size(1)
        
        ## Record representation
        record_representation = self.Record_Embedding_layer(src_k, src_v) # [batch, slen, emb_size]
        
        ## Encoder
        # [batch, slen, hidden_size], [1, batch, hidden_size] , [1, batch, hidden_size]
        encoder_outputs, hidden, control = self.encoder(record_representation, src_lengths) 
               
        ## Record    
        predictions = torch.zeros(batch_size, max_tlen)
        attention_maps = torch.zeros(batch_size, max_tlen, max_slen)
        p_gens = torch.zeros(batch_size, max_tlen)
        
        ## Decoder
        # first input to the decoder is the <sos> token
        decoder_input = torch.ones_like(src_k[:,0]).unsqueeze(1).to(self.device) # [batch, 1]
        for t in range(0, max_tlen):
            decoder_input = self.dropout(self.embedding(decoder_input)) # [batch, 1, emb_size]
            p_vocab, p_copy, p_gen, hidden, control, attention_map = self.decoder.inference(decoder_input, hidden, control, encoder_outputs)           
            
            ## get next one
            p_vocab_final = p_vocab.scatter_add(1, src_v, p_copy)            
            top1 = p_vocab_final.max(1)[1]
            decoder_input = top1.unsqueeze(1) # [batch, 1]
            
            ## logging
            predictions[:,t] = top1
            attention_maps[:,t,:] = attention_map
            p_gens[:,t] = p_gen.squeeze(1)

        return {'predictions':predictions.numpy(), 'attetion_maps':attention_maps.numpy(), 'p_gens':p_gens.numpy()}        
      
    