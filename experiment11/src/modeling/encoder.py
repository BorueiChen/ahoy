import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.encoder_type = config.encoder_type
        self.hidden_size = config.hidden_size

        # Bidirectional learned initial state
        if self.encoder_type == 'brnn':
            self.init_hidden_h = torch.nn.Parameter(
                nn.init.normal_(torch.zeros(2*config.lstm_layers, 1, config.hidden_size,requires_grad=True), 
                             mean=0, std=0.01)).to(config.device)
            self.init_hidden_c = torch.nn.Parameter(
                nn.init.normal_(torch.zeros(2*config.lstm_layers, 1, config.hidden_size,requires_grad=True), 
                             mean=0, std=0.01)).to(config.device)
        else:
            self.init_hidden_h = torch.nn.Parameter(
                nn.init.normal_(torch.zeros(config.lstm_layers, 1, config.hidden_size,requires_grad=True),
                             mean=0, std=0.01)).to(config.device)
            self.init_hidden_c = torch.nn.Parameter(
                nn.init.normal_(torch.zeros(config.lstm_layers, 1, config.hidden_size,requires_grad=True), 
                             mean=0, std=0.01)).to(config.device)
            
        self.init_hidden = (self.init_hidden_h, self.init_hidden_c)
        
        ## lstm cell        
        if self.encoder_type == 'brnn':
            self.cell = nn.LSTM(num_layers=config.lstm_layers, input_size=config.emb_size, hidden_size=config.hidden_size, bidirectional=True, batch_first=True)
        else: #'lstm'
            self.cell = nn.LSTM(num_layers=config.lstm_layers, input_size=config.emb_size, hidden_size=config.hidden_size, bidirectional=False, batch_first=True)

        self.bridge_h = nn.Linear(config.hidden_size, config.hidden_size)
        self.bridge_c = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def init_state(self, batch_size):
        
        state = (self.init_hidden[0].repeat(1, batch_size, 1),
                 self.init_hidden[1].repeat(1, batch_size, 1))
        return state
    
    def forward(self, record_representation, src_lengths):
        """
        Args:
            record_representation: [batch, slen, emb_size]
            src_lengths: [batch]
        Returns:
            outputs: [batch, slen, hidden_size]
            hidden: [1, batch, hidden_size]
            control: [1, batch, hidden_size]
        """    
        batch_size = record_representation.size(0)
        record_representation = pack_padded_sequence(record_representation, src_lengths, batch_first=True, enforce_sorted=False)
        if self.encoder_type == 'brnn':
            outputs, (hidden, control)  = self.cell(record_representation, self.init_state(batch_size)) # [batch, slen, hidden_size]
            outputs, lens_unpacked = pad_packed_sequence(outputs, batch_first=True)
            
            max_len = outputs.size(1)
            outputs = outputs.view(batch_size, max_len, 2, self.hidden_size)
            outputs = outputs.sum(dim=2)            
            
            hidden = torch.sum(self.bridge_h(hidden), dim=0, keepdim=True)
            control = torch.sum(self.bridge_c(control), dim=0, keepdim=True)

        else:
            outputs, (hidden, control)  = self.cell(record_representation, self.init_state(batch_size)) # [batch, slen, hidden_size] 
            outputs, lens_unpacked = pad_packed_sequence(outputs, batch_first=True)
            hidden = torch.sum(self.bridge_h(hidden), dim=0, keepdim=True)
            control = torch.sum(self.bridge_c(control), dim=0, keepdim=True)            
        
        
        return outputs, hidden, control
    
    
class TemplateDistribution(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.device = config.device

        # Bidirectional learned initial state
        self.init_hidden_h = torch.nn.Parameter(nn.init.normal_(torch.zeros(2*config.lstm_layers, 1, config.hidden_size,requires_grad=True), mean=0, std=0.01)).to(config.device)
        self.init_hidden_c = torch.nn.Parameter(nn.init.normal_(torch.zeros(2*config.lstm_layers, 1, config.hidden_size,requires_grad=True), mean=0, std=0.01)).to(config.device)            
        self.init_hidden = (self.init_hidden_h, self.init_hidden_c)
        
        self.gaussian_size = config.gaussian_size
        self.z_posterior = nn.Linear(config.hidden_size, config.gaussian_size * 2)  # for mu_z & logvar_z
        self.weight_final_z_sample = nn.Linear(config.gaussian_size * 2, config.gaussian_size)
        self.weight_encoder_outputs = nn.Linear(config.hidden_size, config.gaussian_size)
        
        ## lstm cell        
        self.cell = nn.LSTM(num_layers=config.lstm_layers, input_size=config.emb_size, hidden_size=config.hidden_size, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(config.dropout)
    
    def init_state(self, batch_size):
        
        state = (self.init_hidden[0].repeat(1, batch_size, 1), self.init_hidden[1].repeat(1, batch_size, 1))
        return state

    def sample_from_gaussian(self, mu, logvar):
        epsilon = logvar.new_empty(logvar.size()).normal_()
        std = torch.exp(0.5 * logvar)
        z = mu + std * epsilon
        return z

    def fusion(self, z_sample, encoder_outputs):
        """
        args:
            z_sample: [batch, emb_size]
            encoder_outputs: [batch, slen, hidde_size]
        returns:
            final_z_sample: [batch, emb_size]
        """
        final_z_sample = self.weight_final_z_sample(torch.cat((z_sample, self.weight_encoder_outputs(encoder_outputs.mean(1))), dim=1))
        
        return final_z_sample
        
    def forward(self, template_distribution_inputs, tgt_lengths, encoder_outputs):
        """
        Args:
            template_distribution_inputs: [batch, tlen, emb_size]
            tgt_lengths: [batch]
            encoder_outputs: [batch, slen, hidde_size]
        Returns:
            gaussian_distribution: [batch, slen, hidden_size]
            gaussian_distribution: [1, batch, hidden_size]
        """    
        batch_size = template_distribution_inputs.size(0)
        template_distribution_inputs = pack_padded_sequence(template_distribution_inputs, tgt_lengths, batch_first=True, enforce_sorted=False)

        outputs, (hidden, control)  = self.cell(template_distribution_inputs, self.init_state(batch_size)) # [batch, slen, hidden_size]
        outputs, lens_unpacked = pad_packed_sequence(outputs, batch_first=True)

        max_len = outputs.size(1)
        outputs = outputs.view(batch_size, max_len, 2, self.hidden_size)
        outputs = outputs.sum(dim=2)# [batch, tlen, hidden_size]            
          
        # get posterior
        posterior_input = outputs.mean(1) # [batch, hidden_size]
        posterior_out_z = self.z_posterior(posterior_input)  # [batch, gaussian_size]
        mu_post_z, logvar_post_z = torch.chunk(posterior_out_z, 2, 1)  # both has size [batch, gaussian_size]
        # sample z from the posterior
        z_sample = self.sample_from_gaussian(mu_post_z, logvar_post_z)  # [batch, gaussian_size]
        
        final_z_sample = self.fusion(z_sample, encoder_outputs)
        
        return final_z_sample, (mu_post_z, logvar_post_z)

    def sample(self, encoder_outputs):
        """
        Args:
            template_distribution_inputs: [batch, tlen, emb_size]
            tgt_lengths: [batch]
            encoder_outputs: [batch, slen, hidde_size]
        Returns:
            gaussian_distribution: [batch, slen, hidden_size]
            gaussian_distribution: [1, batch, hidden_size]
        """    
        batch_size = encoder_outputs.size(0)
        mu_prior = torch.zeros(batch_size, self.gaussian_size).to(self.device)  # [batch, gaussian_size]
        logvar_prior = torch.zeros(batch_size, self.gaussian_size).to(self.device) # [batch, gaussian_size]
        
        z_sample = self.sample_from_gaussian(mu_prior, logvar_prior)  # [batch, gaussian_size]
               
        final_z_sample = self.fusion(z_sample, encoder_outputs)
        
        return final_z_sample