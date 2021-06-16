import os
import numpy as np
import torch
import json
from tqdm import tqdm
from collections import defaultdict
root = './data/'


def build_vocab(save_path):
    vocabs_k=set()
    vocabs_v=set()
    with open(root+'train_input.txt', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            pairs = np.array([pair.split(u"￨") for pair in line.split()])
            vocabs_k |= set(pairs[:, 1])
            vocabs_v |= set(pairs[:, 0])

    vocabs_tgt=set()
    with open(root+'train_output.txt', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            vocabs_tgt |= set(line)

    print(f'vocabs_k :{len(vocabs_k)}')
    print(f'vocabs_v :{len(vocabs_v)}')
    print(f'vocabs_tgt :{len(vocabs_tgt)}')
    print(f'vocabs_v + vocabs_tgt:{len(vocabs_v|vocabs_tgt)}')
    print(f'vocabs_all:{len(vocabs_v|vocabs_tgt|vocabs_k)}')

    vocabs_default=['<pad>', '<bos>', '<eos>', '<unk>', '<ent>']
    word2idx, idx2word={}, {}
    all_vocabs=vocabs_default+list(vocabs_k|vocabs_v|vocabs_tgt)
    for idx, word in enumerate(all_vocabs):
        word2idx[word]=idx
        idx2word[idx]=word

    vocabs = {'idx2word':idx2word, 'word2idx':word2idx}

    torch.save(vocabs, save_path+'/'+'vocabs.pt')
    with open(save_path+'/'+'vocabs.json', 'w') as outfile: json.dump(vocabs, outfile)
    return

def build_dataset(save_path):
    vocabs = torch.load(save_path+'/'+'vocabs.pt')
    value = vocabs['word2idx']['<unk>']
    vocabs['word2idx']  = defaultdict(lambda: value, vocabs['word2idx'])

    dataset = {'train': {'src_k':[], 'src_v':[], 'src_lengths':[], 'tgt':[], 'tgt_lengths':[], 'alignment':[], 'template':[]},
               'valid': {'src_k':[], 'src_v':[], 'src_lengths':[], 'tgt':[], 'tgt_lengths':[], 'alignment':[], 'template':[]},
               'test': {'src_k':[], 'src_v':[], 'src_lengths':[], 'tgt':[], 'tgt_lengths':[], 'alignment':[], 'template':[]},        
            }
    set_name = ['train', 'valid', 'test']
    types = ['_input.txt', '_output.txt']


    for name in set_name:
        print(name)
        input_file_path = root+name+types[0]
        print(input_file_path)
        with open(input_file_path, encoding='utf-8') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                pairs = np.array([pair.split(u"￨") for pair in line.split()])
                src_k = pairs[:, 1]
                src_v = pairs[:, 0]
                length = len(src_k)

                src_k = [vocabs['word2idx'][word] for word in src_k]
                src_v = [vocabs['word2idx'][word] for word in src_v]                
                # save
                dataset[name]['src_k'].append(src_k)
                dataset[name]['src_v'].append(src_v)
                dataset[name]['src_lengths'].append(length)

        output_file_path = root+name+types[1]
        print(output_file_path)
        with open(output_file_path, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                tgt = line.strip().split()
                tgt += ['<eos>']
                length = len(tgt)

                tgt = [vocabs['word2idx'][word] for word in tgt]               
                # save
                dataset[name]['tgt_lengths'].append(length)
                dataset[name]['tgt'].append(tgt)


        ## DO ALIGNMENT
        debug=False
        text_to_number = {  'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
                            'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90 
                         }
        with open(input_file_path, encoding='utf-8') as fi, open(output_file_path, encoding='utf-8') as fo, open(save_path+'/'+name+'_template.txt', 'w', encoding='utf-8') as ft:
            input_lines, output_lines = fi.readlines(), fo.readlines()
            for (input_line, output_line) in tqdm(zip(input_lines, output_lines)):
                ## data preparation
                tgt = output_line.strip().split()
                tgt += ['<eos>']
                template = tgt
                pairs = np.array([pair.split(u"￨") for pair in input_line.split()])
                src_k = pairs[:, 1]
                src_v = pairs[:, 0] 
                alignment = [-1]*len(tgt)
                buffers, buffer_, queue_for_second_stage = [], {'TEAM-CITY':[],'TEAM-NAME':[],'FIRST_NAME':[],'SECOND_NAME':[],'target':[]}, []

                ## first stage alignment
                for word_idx, word in enumerate(tgt):            
                    if word in text_to_number.keys():
                        word_ = str(text_to_number[word])
                    else:
                        word_ = word

                    indices = [i for i, x in enumerate(src_v) if x == word_]
                    key_name = [src_k[idx] for idx in indices]            
                    if indices:
                        # template information
                        template[word_idx] = '<ent>'
                        if debug: print(f'No.{word_idx} Word:[{word}]: indices:{indices}, key:{key_name}\n')
                        # add to queue_for_next_stage
                        if len(indices) > 1: 
                            queue_for_second_stage.append([len(buffers), word_idx, indices, key_name])
                        elif len(indices)==1:
                            alignment[word_idx]=indices[0]

                        # add target into buffer 
                        for key, index in zip(key_name, indices):
                            if key in buffer_.keys():
                                index = 24*(index//24+1)-1
                                buffer_[key].append(index)
                    else:
                        if debug: print(f'No.{word_idx} Word:[{word}]: indices:{indices}\n')

                    if word == '.' or word == '<eos>': 
                        # merge target
                        for first, second in [(set(buffer_['TEAM-CITY']), set(buffer_['TEAM-NAME']) ), (set(buffer_['FIRST_NAME']), set(buffer_['SECOND_NAME']))]:
                            if len(first)>0 and len(second)>0:
                                buffer_['target']+=list(first&second)
                            elif (len(first)>0 and len(second)==0) or (len(first)==0 and len(second)>0):
                                buffer_['target']+=list(first|second)             

                        if debug: print("="*100)
                        if debug: print(f'length of queue_for_second_stage: {len(queue_for_second_stage)}')
                        if debug: print(f'buffer: {buffer_}')
                        if debug: print("="*100)

                        # initial
                        buffers.append(buffer_)
                        buffer_={'TEAM-CITY':[],'TEAM-NAME':[],'FIRST_NAME':[],'SECOND_NAME':[],'target':[]}

                ## second stage alignment

                queue_for_third_stage =[]
                for buffer_idx, word_idx, indices, keys in queue_for_second_stage:
                    indices_target,keys_target=[],[]
                    for idx, key in zip(indices, keys):
                        if 24*(idx//24+1)-1 in buffers[buffer_idx]['target']: 
                            indices_target.append(idx)
                            keys_target.append(key)

                    ## alignment
                    if len(indices_target) == 1: alignment[word_idx]=indices_target[0]
                    elif len(indices_target) > 1: queue_for_third_stage.append([buffer_idx, word_idx, indices_target, keys_target])

                    if debug: print(f'buffer_idx: {buffer_idx}, word_idx:{word_idx}, indices:{indices}')
                    if debug: print(f'buffer_idx: {buffer_idx}, buffer:{buffers[buffer_idx]}')
                    if debug: print(f'merged target:{indices_target} key_target:{keys_target}\n')        


                ## third stage alignment
                for buffer_idx, word_idx, indices, keys in queue_for_third_stage:
                    alignment[word_idx]=indices[0]
                
                # save
                dataset[name]['alignment'].append(alignment)
                ft.write(" ".join(template[:-1])+'\n')
                dataset[name]['template'].append([vocabs['word2idx'][word] for word in template])

    torch.save(dataset, save_path+'/'+'dataset.pt')
    return 

def build(save_path):
    build_vocab(save_path)
    build_dataset(save_path)
    return


    