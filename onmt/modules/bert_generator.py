import torch
import torch.nn as nn
from onmt.modules import Embeddings, CopyGenerator
import onmt.modules
from pytorch_transformers import *
from onmt.modules.util_class import Cast
from onmt.global_model import GlobalModel
import numpy as np
import time



class BertGenerator(nn.Module):

    def __init__(self, generator_function, dec_rnn_size,base_field):
        super(BertGenerator, self).__init__()

        self.generator = nn.Sequential(nn.Linear(dec_rnn_size, len(base_field.vocab)), Cast(torch.float32),generator_function)

    def forward(self, hidden, tgts = None, lang_scores = None):

        log_probs = self.generator(hidden)
        if tgts == None: return log_probs

        increment = 1.1
        language_scores = lang_scores[:,[v for v in GlobalModel.converter.values()]]

        percentile = language_scores.topk(language_scores.shape[1]*(100-80)//100)[0][:,-1].repeat(language_scores.shape[1],1).transpose(0,1)
        mean = log_probs.mean(dim=1).repeat(log_probs.shape[1],1).transpose(0,1)

        #log_probs = log_probs - mean
        #log_probs[language_scores>percentile] = log_probs[language_scores>percentile].clone() * increment
        #log_probs = log_probs + mean

        log_probs = log_probs + language_scores


        
        #log_probs[language_scores>percentile] = log_probs[language_scores>percentile].clone() * increment # + (1-increment)*mean[language_scores>percentile]
        #log_probs[language_scores>percentile] += (1-increment)*mean[language_scores>percentile]

        '''
        lang_scores_temp = []
        #start_time = time.time()
        for tgt in tgts:
            input_ids = tgt.cpu().apply_(lambda y: GlobalModel.converter[y]).to('cuda')
            mask_token = torch.tensor(GlobalModel.tokenizer.encode("[MASK]")).repeat(input_ids.shape[0], 1).unsqueeze(1).to('cuda')
            input_ids = torch.cat([input_ids,mask_token],dim=1).squeeze(2).to('cuda')
            lang_scores_temp.append(GlobalModel.lang_model(input_ids, masked_lm_labels=input_ids)[1][:,-1,:].detach().cpu().numpy())
        #print("--- %s seconds ---" % (time.time() - start_time))

            #language_scores = GlobalModel.lang_model(input_ids, masked_lm_labels=input_ids)[1][:,-1,:]
            #top_lang_scores_temp.append(language_scores.topk(language_scores.shape[1]*(100-90)//100)[1].cpu().numpy())
        language_scores = np.concatenate(lang_scores_temp)
        #top_lang_scores = np.concatenate(top_lang_scores_temp)
        percentile = np.percentile(language_scores,90,axis=1)


        mean = log_probs.mean(dim=1)
        log_probs= (log_probs - mean.repeat(log_probs.shape[1],1).transpose(0,1))
        language_map =log_probs.clone()*0.9
        topk = log_probs.topk(10)[1].cpu().numpy()
        word_ids = set()
        word_ids.add(GlobalModel.vocab.index('[SEP]'))
        for batch_id, word_id in np.ndindex(topk.shape):
            if GlobalModel.converter[topk[batch_id,word_id]] in top_lang_scores[batch_id,:]:
            #if language_scores[batch_id,GlobalModel.converter[topk[batch_id,word_id]]] > percentile[batch_id]:
                word_ids.add(topk[batch_id,word_id].item())
        word_ids = list(word_ids)
        language_map[:,word_ids] = log_probs[:,word_ids] 
        log_probs = language_map.to('cuda')
        log_probs= (log_probs + mean.repeat(log_probs.shape[1],1).transpose(0,1))
        '''

        return log_probs