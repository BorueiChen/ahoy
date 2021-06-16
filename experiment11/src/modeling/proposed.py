import torch
import torch.nn as nn
import random

from .encoder import Encoder, TemplateDistribution
from .decoder import TemplateDecoder

class TemplateDistributionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # high level control conditions
        self.is_bce_loss = config.is_bce_loss
        self.is_coverage_loss = config.is_coverage_loss        
        
        self.device = config.device
        self.embedding = nn.Embedding(config.vocab_size, config.emb_size)
        self.record_linear = nn.Linear(2*config.emb_size, config.emb_size)
        self.encoder = Encoder(config)
        self.template_distribution = TemplateDistribution(config)
        self.decoder = TemplateDecoder(config)
        self.zeros = torch.zeros(1, 1).to(self.device)
        self.dropout = nn.Dropout(config.dropout)

    def gaussian_kld(self, recog_mu, recog_logvar, prior_mu, prior_logvar):
        kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar) -
                               torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar)) -
                               torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), 1)
        return kld
    
    def Record_Embedding_layer(self, src_k, src_v):
        src_k_embedding = self.embedding(src_k) # [batch, slen, emb_size]
        src_v_embedding = self.embedding(src_v) # [batch, slen, emb_size]
        relu = nn.ReLU()
        out = relu(self.record_linear(torch.cat((src_k_embedding, src_v_embedding), dim=2))) # [batch, slen, emb_size]
        
        return out          

    def train_pair(self, src_k, src_v, src_lengths, tgt, tgt_lengths, template, template_lengths, alignment):
        """
        Args:
            src_k: [batch, slen]
            src_v: [batch, slen]
            src_lengths: [batch]
            tgt: [batch, tlen]
            tgt_lengths; [batch]
            template: [batch, tlen]
            template_lengths; [batch]
            alignment: [batch, tlen]
        Returns:
            loss: float
            predictions: [batch, max_tlen]
            attetion_maps: [batch, max_tlen, max_slen]
            p_gens: [batch, max_tlen]
        """
        
        batch_size = tgt.size(0)
        tlen = tgt.size(1)
        
        # Record representation
        record_representation = self.Record_Embedding_layer(src_k, src_v) # [batch, slen, emb_size]
        
        # Encoder
        ## [batch, slen, hidden_size], [batch, hidden_size] , [batch, hidden_size]
        encoder_outputs, hidden, control = self.encoder(record_representation, src_lengths) 
        
        # Template distribution
        ## data prepare
        template_distribution_inputs = self.dropout(self.embedding(tgt)) # [batch, tlen, emb_size]
        ## feed
        gaussian_distribution, (mu_post_z, logvar_post_z) = self.template_distribution(template_distribution_inputs, tgt_lengths, encoder_outputs) # [batch, gaussian_size]
               
        # Decoder   
        bos = torch.ones_like(tgt[:,0]).to(self.device) # bos
        ## data prepare        
        decoder_inputs = torch.cat((bos.unsqueeze(1), tgt[:, :-1]), dim=1) # [batch, tlen]
        decoder_inputs = self.dropout(self.embedding(decoder_inputs)) # [batch, tlen ,emb_size]
        decoder_inputs = torch.cat((decoder_inputs, gaussian_distribution.unsqueeze(1).repeat(1,tlen,1)), dim=2) # [batch, tlen , emb_size+gaussian_size]
        ## template data prepare
        template_decoder_inputs = torch.cat((bos.unsqueeze(1), template[:, :-1]), dim=1) # [batch, tlen]
        template_decoder_inputs = self.dropout(self.embedding(template_decoder_inputs)) # [batch, tlen ,emb_size]
        template_decoder_inputs = torch.cat((template_decoder_inputs, gaussian_distribution.unsqueeze(1).repeat(1,tlen,1)), dim=2)
        ## feed
        (p_vocab, p_copy, p_gen, attention_map), (p_vocab_template, p_copy_template, p_gen_template, attention_map_template) = self.decoder(decoder_inputs, tgt_lengths, template_decoder_inputs,
                                                                                                                                            template_lengths, encoder_outputs, hidden, control)
        
        # calculate total loss
        ## description loss          
        likelihood = torch.gather(p_vocab, 2, tgt.unsqueeze(2)) + torch.gather(torch.cat((p_copy, torch.zeros(batch_size, tlen, 1).to(self.device)), dim=2), 2 , alignment.unsqueeze(2))        
        mask = tgt!=0      
        total_loss = -likelihood.log().masked_select(mask.unsqueeze(2)).mean()
        total_loss_std = -likelihood.log().masked_select(mask.unsqueeze(2)).std()

        ## template loss          
        likelihood = torch.gather(p_vocab_template, 2, template.unsqueeze(2)) + torch.gather(torch.cat((p_copy_template , torch.zeros(batch_size, tlen, 1).to(self.device)), dim=2), 2 , alignment.unsqueeze(2))        
        mask = template!=0    
        template_loss = -likelihood.log().masked_select(mask.unsqueeze(2)).mean()
        total_loss+=template_loss # add template loss
        
        ## bce loss
        _BCE_LOSS = nn.BCELoss()
        bce_target = (alignment != alignment.max()).float()
        bce_loss = _BCE_LOSS(p_gen.squeeze(2), bce_target)
        if self.is_bce_loss: total_loss+=bce_loss      
            
        ## kl loss
        mu_prior_z = self.zeros.expand(gaussian_distribution.size())
        logvar_prior_z = self.zeros.expand(gaussian_distribution.size())
        kl_loss = torch.mean(self.gaussian_kld(mu_post_z, logvar_post_z, mu_prior_z, logvar_prior_z), dim=0)
        total_loss+=kl_loss
       
        
        return {'total_loss': total_loss, 
                'total_loss_std': total_loss_std.item(), 
                'template_loss': template_loss.item(),           
                'bce_loss': bce_loss.item(),
                'kl_loss': kl_loss.item(),
               }
    
    def valid_pair(self, src_k, src_v, src_lengths, tgt, tgt_lengths, template, template_lengths, alignment):
        """
        Args:
            src_k: [batch, slen]
            src_v: [batch, slen]
            src_lengths: [batch]
            tgt: [batch, tlen]
            tgt_lengths; [batch]
            template: [batch, tlen]
            template_lengths; [batch]
            alignment: [batch, tlen]
        Returns:
            loss: float
            predictions: [batch, max_tlen]
            attetion_maps: [batch, max_tlen, max_slen]
            p_gens: [batch, max_tlen]
        """
        
        batch_size = tgt.size(0)
        tlen = tgt.size(1)
        
        # Record representation
        record_representation = self.Record_Embedding_layer(src_k, src_v) # [batch, slen, emb_size]
        
        # Encoder
        ## [batch, slen, hidden_size], [batch, hidden_size] , [batch, hidden_size]
        encoder_outputs, hidden, control = self.encoder(record_representation, src_lengths) 
        
        # Template distribution
        ## data prepare
        template_distribution_inputs = self.dropout(self.embedding(tgt)) # [batch, tlen, emb_size]
        ## feed
        gaussian_distribution, (mu_post_z, logvar_post_z) = self.template_distribution(template_distribution_inputs, tgt_lengths, encoder_outputs) # [batch, gaussian_size]
               
        # Decoder   
        bos = torch.ones_like(tgt[:,0]).to(self.device) # bos
        ## data prepare        
        decoder_inputs = torch.cat((bos.unsqueeze(1), tgt[:, :-1]), dim=1) # [batch, tlen]
        decoder_inputs = self.dropout(self.embedding(decoder_inputs)) # [batch, tlen ,emb_size]
        decoder_inputs = torch.cat((decoder_inputs, gaussian_distribution.unsqueeze(1).repeat(1,tlen,1)), dim=2) # [batch, tlen , emb_size+gaussian_size]
        ## template data prepare
        template_decoder_inputs = torch.cat((bos.unsqueeze(1), template[:, :-1]), dim=1) # [batch, tlen]
        template_decoder_inputs = self.dropout(self.embedding(template_decoder_inputs)) # [batch, tlen ,emb_size]
        template_decoder_inputs = torch.cat((template_decoder_inputs, gaussian_distribution.unsqueeze(1).repeat(1,tlen,1)), dim=2)
        ## feed
        (p_vocab, p_copy, p_gen, attention_map), (p_vocab_template, p_copy_template, p_gen_template, attention_map_template) = self.decoder(decoder_inputs, tgt_lengths, template_decoder_inputs,
                                                                                                                                            template_lengths, encoder_outputs, hidden, control)
        
        # calculate total loss
        ## description loss          
        likelihood = torch.gather(p_vocab, 2, tgt.unsqueeze(2)) + torch.gather(torch.cat((p_copy, torch.zeros(batch_size, tlen, 1).to(self.device)), dim=2), 2 , alignment.unsqueeze(2))        
        mask = tgt!=0      
        total_loss = -likelihood.log().masked_select(mask.unsqueeze(2)).mean()
        total_loss_std = -likelihood.log().masked_select(mask.unsqueeze(2)).std()

        ## template loss          
        likelihood = torch.gather(p_vocab_template, 2, template.unsqueeze(2)) + torch.gather(torch.cat((p_copy_template , torch.zeros(batch_size, tlen, 1).to(self.device)), dim=2), 2 , alignment.unsqueeze(2))        
        mask = template!=0    
        template_loss = -likelihood.log().masked_select(mask.unsqueeze(2)).mean()
        total_loss+=template_loss # add template loss
        
        ## bce loss
        _BCE_LOSS = nn.BCELoss()
        bce_target = (alignment != alignment.max()).float()
        bce_loss = _BCE_LOSS(p_gen.squeeze(2), bce_target)
        if self.is_bce_loss: total_loss+=bce_loss      
            
        ## kl loss
        mu_prior_z = self.zeros.expand(gaussian_distribution.size())
        logvar_prior_z = self.zeros.expand(gaussian_distribution.size())
        kl_loss = torch.mean(self.gaussian_kld(mu_post_z, logvar_post_z, mu_prior_z, logvar_prior_z), dim=0)
        total_loss+=kl_loss
        
        ## predictions
        p_vocab_final = p_vocab.scatter_add(2, src_v.unsqueeze(1).repeat(1, tlen, 1), p_copy)            
        predictions = p_vocab_final.max(2)[1]        
        
        
        return {'total_loss':total_loss, 
                'total_loss_std': total_loss_std.item(), 
                'template_loss': template_loss.item(),           
                'bce_loss': bce_loss.item(),
                'kl_loss': kl_loss.item(),
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
            gaussian_distribution = self.template_distribution.sample(encoder_outputs) # [batch, gaussian_size]
            decoder_input = torch.cat((decoder_input, gaussian_distribution.unsqueeze(1)), dim=2) # [batch, 1 , emb_size+gaussian_size]            
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
      
    