import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
class Rotowire(object):
    
    def __init__(self, config):
        _dataset = torch.load('./experiments/'+config.name+'/data/dataset.pt') 
        self.vocabs = torch.load('./experiments/'+config.name+'/data/vocabs.pt')
        self.trainloader=self.build_dataloader('train', _dataset, config.batch_size, config.num_workers)
        self.validloader=self.build_dataloader('valid', _dataset, config.batch_size, config.num_workers)
        self.testloader=self.build_dataloader('test', _dataset, config.batch_size, config.num_workers)
        
        
    @property
    def vocab_size(self): return len(self.vocabs['idx2word'])
    
    def build_dataloader(self, setname, dataset, batch_size, num_workers):
        dataset = _Rotowire(setname, dataset, self.vocabs['word2idx']['<pad>'])        
        return DataLoader(dataset, batch_size=batch_size, 
                          shuffle=(setname=='train'), 
                          drop_last=(setname=='train'), 
                          pin_memory=True,
                          num_workers=num_workers,
                          collate_fn=dataset.get_collate_fn())
    
    def post_process(self, batch, name="test"):
        """
        batch:
            predictions: [batch, max_tlen], type: numpy
            attetion_maps: [batch, max_tlen, max_slen], type: numpy
            p_gens: [batch, max_tlen], type: numpy
        return:
            post_process: [batch, max_tlen], type list
            predictions: [batch, max_tlen], type: numpy
            attetion_maps: [batch, max_tlen, max_slen], type: numpy
            p_gens: [batch, max_tlen], type: numpy            
            
        """
        post_process = []
        if name == "valid":            
            for predictions, tlen in zip(batch['predictions'], batch['tgt_lengths']):
                post_process_=[]
                for idx, word_idx in enumerate(predictions):
                    if word_idx != self.vocabs['word2idx']['<eos>'] and  idx < tlen:
                        post_process_.append(self.vocabs['idx2word'][word_idx])
                    else:
                        break
                post_process.append(post_process_)
        elif name == "test":
            for predictions in batch['predictions']:
                post_process_=[]
                for idx in predictions:
                    if idx != self.vocabs['word2idx']['<eos>']:
                        post_process_.append(self.vocabs['idx2word'][idx])
                    else:
                        break
                post_process.append(post_process_)

        batch['post_process'] = post_process     
        
        return batch
    
class _Rotowire(Dataset):
    def __init__(self, setname, dataset, pad):
        self.setname = setname
        self._dataset = dataset[setname]
        self.pad = pad

    def get_collate_fn(self):
        if self.setname == 'train' or self.setname == 'valid':
            def fn(batch):
                """
                data = [('src_k', 'src_v', ...), ...]
                """
                src_k, src_v,  src_lengths, tgt, tgt_lengths, alignment, template, template_lengths = zip(*batch)

                ## padding
                src_k = pad_sequence([torch.tensor(data) for data in src_k], batch_first=True , padding_value=self.pad)
                src_v = pad_sequence([torch.tensor(data) for data in src_v], batch_first=True , padding_value=self.pad)
                tgt = pad_sequence([torch.tensor(data) for data in tgt], batch_first=True , padding_value=self.pad)
                alignment = pad_sequence([torch.tensor(data) for data in alignment], batch_first=True , padding_value=-1)
                template = pad_sequence([torch.tensor(data) for data in template], batch_first=True , padding_value=self.pad)  

                ## convert
                alignment[alignment==-1] = src_k.size(1) # replace with max slen + 1
                src_lengths = torch.tensor(src_lengths)
                tgt_lengths = torch.tensor(tgt_lengths)
                template_lengths = torch.tensor(template_lengths)

                batch = {'src_k':src_k, 
                         'src_v':src_v, 
                         'src_lengths':src_lengths, 
                         'tgt':tgt, 
                         'tgt_lengths':tgt_lengths, 
                         'alignment':alignment,
                         'template':template, 
                         'template_lengths':template_lengths,
                        }

                return batch
            return fn
        elif self.setname == 'test':
            def fn(batch):
                """
                data = [('src_k', 'src_v', ...), ...]
                """
                src_k, src_v,  src_lengths = zip(*batch)

                ## padding
                src_k = pad_sequence([torch.tensor(data) for data in src_k], batch_first=True , padding_value=self.pad)
                src_v = pad_sequence([torch.tensor(data) for data in src_v], batch_first=True , padding_value=self.pad)  

                ## convert
                src_lengths = torch.tensor(src_lengths)

                batch = {'src_k':src_k, 
                         'src_v':src_v, 
                         'src_lengths':src_lengths, 
                        }

                return batch
            return fn
    def __len__(self):
        return len(self._dataset['src_k'])

    def __getitem__(self, idx):

        if self.setname == 'train' or self.setname == 'valid':
            src_k = self._dataset['src_k'][idx]
            src_v = self._dataset['src_v'][idx]
            src_lengths = self._dataset['src_lengths'][idx]
            tgt = self._dataset['tgt'][idx]
            tgt_lengths = self._dataset['tgt_lengths'][idx]
            alignment = self._dataset['alignment'][idx]
            template = self._dataset['template'][idx]
            template_lengths = self._dataset['tgt_lengths'][idx]

            sample = (src_k, src_v,  src_lengths, tgt, tgt_lengths, alignment, template, template_lengths)

        elif self.setname == 'test':
            src_k = self._dataset['src_k'][idx]
            src_v = self._dataset['src_v'][idx]
            src_lengths = self._dataset['src_lengths'][idx]

            sample = (src_k, src_v, src_lengths)

        return sample