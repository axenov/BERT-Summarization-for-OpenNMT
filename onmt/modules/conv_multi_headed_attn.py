""" Multi-Head Attention module """
import math
import torch
import torch.nn as nn
import torch as torch
import torch.nn.functional as F

from onmt.utils.misc import generate_relative_positions_matrix,\
                            relative_matmul
# from onmt.utils.misc import aeq


class ConvMultiHeadedAttention(nn.Module):
    """Multi-Head Attention module from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, M, N, dropout=0.1,
                 max_relative_positions=0):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(ConvMultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.convol_softmax = self.convolSoftmax
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)
        self.M=M
        self.N=N

        self.max_relative_positions = max_relative_positions

        if max_relative_positions > 0:
            vocab_size = max_relative_positions * 2 + 1
            self.relative_positions_embeddings = nn.Embedding(
                vocab_size, self.dim_per_head)

    def forward(self, key, value, query, mask=None,
                layer_cache=None, type=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask indicating which keys have
               non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):

           * output context vectors ``(batch, query_len, dim)``
           * one of the attention vectors ``(batch, query_len, key_len)``
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)
        value_len = value.size(1)
        device = key.device
        # Size of the window
        M = self.M
        N = self.N

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)
        # 1) Project key, value, and query.
        if layer_cache is not None:
            if type == "self":
                query, key, value = self.linear_query(query),\
                                    self.linear_keys(query),\
                                    self.linear_values(query)
                key = shape(key)
                value = shape(value)
                if layer_cache["self_keys"] is not None:
                    key = torch.cat(
                        (layer_cache["self_keys"].to(device), key),
                        dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat(
                        (layer_cache["self_values"].to(device), value),
                        dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value
            elif type == "context":
                query = self.linear_query(query)
                if layer_cache["memory_keys"] is None:
                    key, value = self.linear_keys(key),\
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache["memory_keys"],\
                               layer_cache["memory_values"]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)

        # batch x num_heads * query_len x num_heads * key_len
        query = query.contiguous().view(batch_size, head_count * query_len, dim_per_head)
        key = key.contiguous().view(batch_size, head_count * key_len, dim_per_head)
        value = value.contiguous().view(batch_size, head_count *  value_len, dim_per_head)
        #print('0: {}'.format(torch.cuda.memory_allocated() / 1024**2))


        # batch x num_heads x query_len x key_len
        scores = torch.matmul(query, key.transpose(1, 2))
        #print('1: {}'.format(torch.cuda.memory_allocated() / 1024**2))
        scores = scores.float()

        if mask is not None:
          scores = scores.contiguous().view(batch_size, head_count, query_len, head_count , key_len).transpose(2,3)
          mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
          mask = mask.unsqueeze(1)  # [B, 1, 1, 1, T_values]
          scores = scores.masked_fill(mask, -1e18)
          scores = scores.transpose(2,3).contiguous().view(batch_size, head_count * query_len, head_count * key_len)
        # 3) Apply attention dropout and compute context vectors.
        #attn = self.softmax(scores).to(query.dtype)
        attn = self.convol_softmax(scores, head_count,M, N, device).to(query.dtype)
        del scores

        attn = attn.contiguous().view(batch_size, head_count, query_len, head_count , key_len)

        mask_w = torch.zeros_like(attn, dtype=torch.uint8)  # or dtype=torch.ByteTensor

        attn_masked = torch.zeros_like(attn).to(device)
        for i in range(head_count):
          #attn_temp[:,i,:,max(0,i-(N-1)//2):min(head_count,i+(N-1)//2+1),:] = attn[:,i,:,max(0,i-(N-1)//2):min(head_count,i+(N-1)//2+1),:]
          mask_w[:,i,:,max(0,i-(N-1)//2):min(head_count,i+(N-1)//2+1),:] = 1
        for j in range(query_len):
          mask_w[:,:,j,:,max(0,j-(M-1)//2):min(query_len,j+(M-1)//2+1)] = 1
          #attn_temp[:,:,j,:,max(0,j-(M-1)//2):min(query_len,j+(M-1)//2+1)] = attn[:,:,j,:,max(0,j-(M-1)//2):min(query_len,j+(M-1)//2+1)]
        torch.cuda.empty_cache()
        #print('6: {}'.format(torch.cuda.memory_allocated() / 1024**2))

        attn_masked.masked_scatter_(mask_w,attn)
        torch.cuda.empty_cache()  
        #print('7: {}'.format(torch.cuda.memory_allocated() / 1024**2))

        #attn = attn_temp
        attn_masked = attn_masked.contiguous().view(batch_size, head_count * query_len, head_count *  key_len)

        drop_attn = self.dropout(attn_masked)
        context_original = torch.matmul(drop_attn, value)

        context = unshape(context_original)
        output = self.final_linear(context)

        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # Return one attn
        top_attn = attn \
            .view(batch_size, head_count,
                  query_len, head_count, key_len)[:, 0, :,0, :] \
            .contiguous()

        return output, top_attn

    def convolSoftmax(self,scores,head_count,M,N,device):

        batch_size = scores.size(0)
        # Num of heads X Sequence leangth
        all_keys_size = scores.size(1)
        # Exponent values
        scores_exp = torch.exp(scores)
        torch.cuda.empty_cache()
        #print('2: {}'.format(torch.cuda.memory_allocated() / 1024**2))

        #Convolute over sequence
        output = scores_exp.contiguous().view(batch_size * all_keys_size * head_count, 1, all_keys_size//head_count)
        kernel_seq = torch.zeros(1,1,M).to(device)
        kernel_seq[:,:,0:M] = 1
        output = F.conv1d(output, kernel_seq, padding=(M-1)//2)
        output = output.contiguous().view(batch_size * all_keys_size , head_count, all_keys_size//head_count).transpose(1,2)
        torch.cuda.empty_cache()
        #print('3: {}'.format(torch.cuda.memory_allocated() / 1024**2))  

        #Convolute over heads
        output = output.contiguous().view(batch_size * all_keys_size * all_keys_size//head_count, 1, head_count)
        kernel_head = torch.zeros(1,1,N).to(device)
        kernel_head[:,:,0:N] = 1
        output = F.conv1d(output, kernel_head, padding=(N-1)//2)
        output = output.contiguous().view(batch_size * all_keys_size , all_keys_size//head_count, head_count).transpose(1,2)
        torch.cuda.empty_cache()
        #print('4: {}'.format(torch.cuda.memory_allocated() / 1024**2))

        #Calculate softmax
        output = output.contiguous().view(batch_size , all_keys_size , all_keys_size)
        output = scores_exp.div(output)#
        #self.softmax(scores).to(query.dtype)
        del scores_exp
        torch.cuda.empty_cache()
        #print('5: {}'.format(torch.cuda.memory_allocated() / 1024**2))

        #for i in range(all_keys_size):
        #  mask = torch.zeros(all_keys_size)
        #  #mask[1]
        #  masked = scores_exp
        #  output[:,:,i] = scores_exp[:,:,i] / torch.sum(masked,dim=2)

        return output

