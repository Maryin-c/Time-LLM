from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from transformers import AutoModel, AutoTokenizer, AutoConfig
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'Qwen':
            self.qwen_config = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B")
            # self.qwen_config.num_hidden_layers = configs.llm_layers 
            self.qwen_config.output_attentions = True 
            self.qwen_config.output_hidden_states = True  

            try:
                self.llm_model = AutoModel.from_pretrained(
                    "Qwen/Qwen2.5-7B",
                    trust_remote_code=True,  
                    local_files_only=True,  
                    config=self.qwen_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = AutoModel.from_pretrained(
                    "Qwen/Qwen2.5-7B",
                    trust_remote_code=True,  
                    local_files_only=False,  
                    config=self.qwen_config,
                )

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen2.5-7B",
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen2.5-7B",
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'DeepSeek':
            self.deepseek_config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V2-Lite", trust_remote_code=True)
            # self.deepseek_config.num_hidden_layers = configs.llm_layers 
            self.deepseek_config.output_attentions = True 
            self.deepseek_config.output_hidden_states = True  

            try:
                self.llm_model = AutoModel.from_pretrained(
                    "deepseek-ai/DeepSeek-V2-Lite",
                    trust_remote_code=True,  
                    local_files_only=True,  
                    config=self.deepseek_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = AutoModel.from_pretrained(
                    "deepseek-ai/DeepSeek-V2-Lite",
                    trust_remote_code=True,  
                    local_files_only=False,  
                    config=self.deepseek_config,
                )

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "deepseek-ai/DeepSeek-V2-Lite",
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "deepseek-ai/DeepSeek-V2-Lite",
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)


        if configs.attn_type == 'mha':
            print(f'ReprogrammingLayer using MHA......')
            self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
        elif configs.attn_type == 'mqa':
            print(f'ReprogrammingLayer using MQA......')
            self.reprogramming_layer = ReprogrammingLayer_MQA(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
        elif configs.attn_type == 'GQA':
            print(f'ReprogrammingLayer using GQA......')
            self.reprogramming_layer = ReprogrammingLayer_GQA(configs.d_model, configs.n_heads, configs.n_groups, self.d_ff, self.d_llm)
        elif configs.attn_type == 'mla':
            self.reprogramming_layer = ReprogrammingLayer_MLA(d_model=configs.d_model, n_heads=configs.n_heads,  d_llm=self.d_llm, q_lora_rank=configs.q_lora_rank, kv_lora_rank=configs.kv_lora_rank)
            print(f'ReprogrammingLayer using MLA......')
        else:
            print(f'wrong type of attention mechanism in ReprogrammingLayer')
            exit()


        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
    
"""wxq add """

class ReprogrammingLayer_MQA(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer_MQA, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)

        # share single head key and value
        self.key_projection = nn.Linear(d_llm, d_keys * 1)
        self.value_projection = nn.Linear(d_llm, d_keys * 1)


        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, 1, -1)
        value_embedding = self.value_projection(value_embedding).view(S, 1, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)


        # broadcasting
        scores = torch.einsum("blhe,ske->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        # broadcasting
        reprogramming_embedding = torch.einsum("bhls,ske->blhe", A, value_embedding)

        return reprogramming_embedding
    


class ReprogrammingLayer_GQA(nn.Module):
    def __init__(self, d_model, n_heads, n_groups, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer_GQA, self).__init__()

        assert n_heads % n_groups == 0

        d_keys = d_keys or (d_model // n_heads)
        self.n_heads = n_heads
        self.n_groups = n_groups 
        self.heads_per_group = n_heads // n_groups 

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)

        self.key_projection = nn.Linear(d_llm, d_keys * n_groups)
        self.value_projection = nn.Linear(d_llm, d_keys * n_groups)

        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H, G = self.n_heads, self.n_groups

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)

        source_embedding = self.key_projection(source_embedding).view(S, G, -1)
        value_embedding = self.value_projection(value_embedding).view(S, G, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape 
        S, G, E = source_embedding.shape 
        heads_per_group = self.heads_per_group 

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,sge->bhls", target_embedding, source_embedding)

        scores = scores.view(B, G, heads_per_group, L, S).permute(0, 2, 1, 3, 4).reshape(B, H, L, S)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        reprogramming_embedding = torch.einsum("bhls,sge->blhe", A, value_embedding)

        return reprogramming_embedding
    


class ReprogrammingLayer_MLA(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_llm: int,
        d_keys: int = None,
        q_lora_rank: int = None,
        kv_lora_rank: int = None,
        attention_dropout: float = 0.1,
    ):
        super(ReprogrammingLayer_MLA, self).__init__()
        self.d_model = d_model
        self.d_llm = d_llm
        self.n_heads = n_heads
        self.d_keys = d_keys or (d_model // n_heads)
        # self.head_dim = q_head_dim if q_head_dim is not None else d_model // n_heads
        # self.v_head_dim = v_head_dim if v_head_dim is not None else d_llm // n_heads
        self.q_lora_rank = q_lora_rank  
        self.kv_lora_rank = kv_lora_rank
        
        self.q_a_proj = nn.Linear(d_model, q_lora_rank, bias=False)
        self.q_a_layernorm = nn.LayerNorm(q_lora_rank)
        self.q_b_proj = nn.Linear(q_lora_rank, n_heads * self.d_keys, bias=False)


        self.kv_a_proj_with_mqa = nn.Linear(
            self.d_llm,
            self.kv_lora_rank,
            bias=False
        )
        self.kv_a_layernorm = nn.LayerNorm(self.kv_lora_rank)

        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.n_heads
            * (self.d_keys + self.d_keys),
            bias=False,
        )

        
        self.dropout = nn.Dropout(attention_dropout)
        self.out_proj = nn.Linear(self.n_heads * self.d_keys, d_llm)
        
    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S = source_embedding.shape[0]

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(target_embedding)))
        q = q.view(B, L, self.n_heads, self.d_keys)



        assert torch.equal(source_embedding, value_embedding)

        compressed_kv = self.kv_a_proj_with_mqa(source_embedding)

        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(1, S, self.n_heads, 2 * self.d_keys)
        )

        k, v = torch.split(
            kv, [self.d_keys, self.d_keys], dim=-1
        )
        
        # q_inter = self.q_a_proj(target_embedding)      
        # q_inter = self.q_a_layernorm(q_inter)      
        # q = self.q_b_proj(q_inter)       

        # q = q.view(B, L, self.n_heads, self.q_head_dim)
        
        # k_inter = self.key_a_proj(source_embedding)        
        # k_inter = self.key_a_layernorm(k_inter)        
        # k = self.key_b_proj(k_inter)        
        # k = k.view(S, self.n_heads, self.q_head_dim)
        
        # v_inter = self.value_a_proj(value_embedding)          
        # v_inter = self.value_a_layernorm(v_inter)         
        # v = self.value_b_proj(v_inter)                      
        # v = v.view(S, self.n_heads, self.v_head_dim)


        k = k.squeeze(0)
        v = v.squeeze(0)

        print(f'q.shape:{q.shape}')
        print(f'k.shape:{k.shape}')
        print(f'v.shape:{v.shape}')

        out = self.reprogramming(q, k, v)

        out = out.reshape(B, L, -1)

        return self.out_proj(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
        




    




