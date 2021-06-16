import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.weight_h = nn.Linear(config.hidden_size, config.attention_size)
        self.weight_s = nn.Linear(config.hidden_size, config.attention_size)
        self.weight_v = nn.Linear(config.attention_size, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        """
        Args:
            encoder_outputs: [batch, slen, hidden]
            decoder_hidden: [batch, tlen, hidden]
        Returns:
            attention_map: [batch, tlen, slen]
        """
        slen = encoder_outputs.size(1)

        decoder_hidden = decoder_hidden.unsqueeze(2).repeat(1, 1, slen, 1) # [batch, tlen, slen, hidden]

        energy = self.weight_v(torch.tanh(self.weight_h(encoder_outputs).unsqueeze(1) + self.weight_s(decoder_hidden))).squeeze(3) # [batch, tlen, slen]

        return F.softmax(energy, dim=2)
    
    
class TemplateDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.attention = Attention(config)
        self.template_rnn = nn.LSTM(config.emb_size+config.gaussian_size, config.hidden_size, batch_first=True) 
        self.word_rnn = nn.LSTM(config.emb_size+config.gaussian_size, config.hidden_size, batch_first=True)        

        self.vocab_size = config.vocab_size        
        self.out = nn.Sequential(nn.Linear(config.hidden_size * 2, self.vocab_size),
                                 nn.Linear(self.vocab_size, self.vocab_size)
                                )
        
        self.weight_h = nn.Linear(config.hidden_size, 1)
        self.weight_s = nn.Linear(config.hidden_size, 1)
        self.weight_x = nn.Linear(config.emb_size+config.gaussian_size, 1)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, decoder_inputs, tgt_lengths, template_decoder_inputs, template_lengths, encoder_outputs, decoder_hidden, decoder_control):
        """
        Args:
            decoder_inputs: [batch, tlen, emb_size+gaussian_size]
            tgt_lengths: [batch]
            template_decoder_inputs: [batch, tlen, emb_size+gaussian_size]
            template_lengths: [batch]
            encoder_outputs: [batch, slen, hidden]
            decoder_hidden: [1, batch, hidden_size]
            decoder_control: [1, batch, hidden_size]            
            
        Returns:
            p_vocab: [batch, tlen, vocab_size]
            p_copy: [batch, tlen, extend_size]
            p_gen: [batch, tlen, 1]
            attention_map: [batch, tlen, slen]
        """
        # description
        decoder_inputs = pack_padded_sequence(decoder_inputs, tgt_lengths, batch_first=True, enforce_sorted=False)
        decoder_outputs, _ = self.word_rnn(decoder_inputs, (decoder_hidden, decoder_control)) # [batch, tlen, hidden_size]
        
        decoder_outputs, _ = pad_packed_sequence(decoder_outputs, batch_first=True)
        decoder_inputs, _ = pad_packed_sequence(decoder_inputs, batch_first=True)
        
        # copygenerator
        attention_map = self.attention(encoder_outputs, decoder_outputs) # [batch, tlen, slen]
        context = torch.bmm(attention_map, encoder_outputs) # [batch, tlen, hidden_size]
        
        p_vocab = F.softmax(self.out(torch.cat((decoder_outputs, context), dim=2)), dim=2) # [batch, tlen, vocab_size]
        p_gen = torch.sigmoid(self.weight_h(context) + self.weight_s(decoder_outputs) + self.weight_x(decoder_inputs)) # [batch, tlen, 1]
        
        p_vocab = torch.mul(p_vocab, p_gen) # [batch, tlen, vocab_size]
        p_copy = torch.mul(attention_map, 1 - p_gen) # [batch, tlen, extend_size]
        
        ###############################################################################################################################
        ###############################################################################################################################
        
        # template
        template_decoder_inputs = pack_padded_sequence(template_decoder_inputs, template_lengths, batch_first=True, enforce_sorted=False)
        template_decoder_outputs, _ = self.template_rnn(template_decoder_inputs, (decoder_hidden, decoder_control)) # [batch, tlen, hidden_size]
        
        template_decoder_outputs, _ = pad_packed_sequence(template_decoder_outputs, batch_first=True)
        template_decoder_inputs, _ = pad_packed_sequence(template_decoder_inputs, batch_first=True)
        
        # copygenerator
        template_attention_map = self.attention(encoder_outputs, template_decoder_outputs) # [batch, tlen, slen]
        context = torch.bmm(template_attention_map, encoder_outputs) # [batch, tlen, hidden_size]
        
        template_p_vocab = F.softmax(self.out(torch.cat((template_decoder_outputs, context), dim=2)), dim=2) # [batch, tlen, vocab_size]
        template_p_gen = torch.sigmoid(self.weight_h(context) + self.weight_s(template_decoder_outputs) + self.weight_x(template_decoder_inputs)) # [batch, tlen, 1]
        
        template_p_vocab = torch.mul(template_p_vocab, template_p_gen) # [batch, tlen, vocab_size]
        template_p_copy = torch.mul(template_attention_map, 1 - template_p_gen) # [batch, tlen, extend_size]
        
        
        return (p_vocab, p_copy, p_gen, attention_map), (template_p_vocab, template_p_copy, template_p_gen, template_attention_map)
    
    def inference(self, decoder_input, decoder_hidden, decoder_control, encoder_outputs):
        """
        Args:
            decoder_input: [batch, 1, emb_size+gaussian_size]
            src_lengths: [batch]
            decoder_hidden: [1, batch, hidden_size]
            decoder_control: [1, batch, hidden_size]
            encoder_outputs: [batch, slen, hidden]
            
        Returns:
            p_vocab: [batch, vocab_size]
            p_copy: [batch, slen]
            p_gen: [batch, 1]
            hidden: [1, batch, hidden]
            control: [1, batch, hidden]
            attention_map: [batch, slen]
        """
        decoder_outputs, (decoder_hidden, decoder_control) = self.word_rnn(decoder_input, (decoder_hidden, decoder_control)) # [1, batch, hidden_size]
        
        attention_map = self.attention(encoder_outputs, decoder_outputs) # [batch, 1, slen]
        context = torch.bmm(attention_map, encoder_outputs) # [batch, 1, hidden_size]
        
        p_vocab = F.softmax(self.out(torch.cat((decoder_outputs, context), dim=2)), dim=2) # [batch, 1, vocab_size]
        p_gen = torch.sigmoid(self.weight_h(context) + self.weight_s(decoder_outputs) + self.weight_x(decoder_input)) # [batch, 1, 1]
        
        p_vocab = torch.mul(p_vocab, p_gen) # [batch, 1, vocab_size]
        p_copy = torch.mul(attention_map, 1 - p_gen) # [batch, 1, extend_size]
               
        return p_vocab.squeeze(1), p_copy.squeeze(1), p_gen.squeeze(1), decoder_hidden, decoder_control, attention_map.squeeze(1)
    
    
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.attention = Attention(config)
        self.rnn = nn.LSTM(config.emb_size, config.hidden_size, batch_first=True)        

        self.vocab_size = config.vocab_size        
        self.out = nn.Sequential(nn.Linear(config.hidden_size * 2, self.vocab_size),
                                 nn.Linear(self.vocab_size, self.vocab_size)
                                )
        
        self.weight_h = nn.Linear(config.hidden_size, 1)
        self.weight_s = nn.Linear(config.hidden_size, 1)
        self.weight_x = nn.Linear(config.emb_size, 1)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, decoder_inputs, tgt_lengths, encoder_outputs, decoder_hidden, decoder_control):
        """
        Args:
            decoder_input: [batch, tlen, emb_size]
            tgt_lengths: [batch]
            encoder_outputs: [batch, slen, hidden]
            decoder_hidden: [1, batch, hidden_size]
            decoder_control: [1, batch, hidden_size]            
            
        Returns:
            p_vocab: [batch, tlen, vocab_size]
            p_copy: [batch, tlen, extend_size]
            p_gen: [batch, tlen, 1]
            attention_map: [batch, tlen, slen]
        """
        decoder_inputs = pack_padded_sequence(decoder_inputs, tgt_lengths, batch_first=True, enforce_sorted=False)
        decoder_outputs, _ = self.rnn(decoder_inputs, (decoder_hidden, decoder_control)) # [batch, tlen, hidden_size]
        
        decoder_outputs, _ = pad_packed_sequence(decoder_outputs, batch_first=True)
        decoder_inputs, _ = pad_packed_sequence(decoder_inputs, batch_first=True)
        
        attention_map = self.attention(encoder_outputs, decoder_outputs) # [batch, tlen, slen]
        context = torch.bmm(attention_map, encoder_outputs) # [batch, tlen, hidden_size]
        
        p_vocab = F.softmax(self.out(torch.cat((decoder_outputs, context), dim=2)), dim=2) # [batch, tlen, vocab_size]
        p_gen = torch.sigmoid(self.weight_h(context) + self.weight_s(decoder_outputs) + self.weight_x(decoder_inputs)) # [batch, tlen, 1]
        
        p_vocab = torch.mul(p_vocab, p_gen) # [batch, tlen, vocab_size]
        p_copy = torch.mul(attention_map, 1 - p_gen) # [batch, tlen, extend_size]
        
        
        return p_vocab, p_copy, p_gen, attention_map
    
    def inference(self, decoder_input, decoder_hidden, decoder_control, encoder_outputs):
        """
        Args:
            decoder_input: [batch, 1, emb_size]
            src_lengths: [batch]
            decoder_hidden: [1, batch, hidden_size]
            decoder_control: [1, batch, hidden_size]
            encoder_outputs: [batch, slen, hidden]
            
        Returns:
            p_vocab: [batch, vocab_size]
            p_copy: [batch, slen]
            p_gen: [batch, 1]
            hidden: [1, batch, hidden]
            control: [1, batch, hidden]
            attention_map: [batch, slen]
        """
        decoder_outputs, (decoder_hidden, decoder_control) = self.rnn(decoder_input, (decoder_hidden, decoder_control)) # [1, batch, hidden_size]
        
        attention_map = self.attention(encoder_outputs, decoder_outputs) # [batch, 1, slen]
        context = torch.bmm(attention_map, encoder_outputs) # [batch, 1, hidden_size]
        
        p_vocab = F.softmax(self.out(torch.cat((decoder_outputs, context), dim=2)), dim=2) # [batch, 1, vocab_size]
        p_gen = torch.sigmoid(self.weight_h(context) + self.weight_s(decoder_outputs) + self.weight_x(decoder_input)) # [batch, 1, 1]
        
        p_vocab = torch.mul(p_vocab, p_gen) # [batch, 1, vocab_size]
        p_copy = torch.mul(attention_map, 1 - p_gen) # [batch, 1, extend_size]
               
        return p_vocab.squeeze(1), p_copy.squeeze(1), p_gen.squeeze(1), decoder_hidden, decoder_control, attention_map.squeeze(1)