from models.TimeLLM import *
import torch



if __name__ == '__main__':

    d_model = 1024
    n_heads = 8
    d_llm = 512
    n_groups = 4

    btz = 16
    seq_len = 256
    S = 64
    target_embedding = torch.randn((btz, seq_len, d_model))
    source_embedding = torch.randn((S, d_llm))
    value_embedding = torch.randn((S, d_llm))

    print(f'target_embedding.shape:{target_embedding.shape}')
    print(f'source_embedding.shape:{source_embedding.shape}')
    print(f'value_embedding.shape:{value_embedding.shape}')

    reprogrammingLayer_mha = ReprogrammingLayer(d_model=d_model, n_heads=n_heads, d_llm=d_llm)
    reprogrammingLayer_mqa = ReprogrammingLayer_MQA(d_model=d_model, n_heads=n_heads, d_llm=d_llm)
    reprogrammingLayer_gqa = ReprogrammingLayer_GQA(d_model=d_model, n_heads=n_heads, d_llm=d_llm, n_groups=n_groups)

    q_lora_rank = 16
    kv_lora_rank = 16


    reprogrammingLayer_mla = ReprogrammingLayer_MLA(d_model=d_model, n_heads=n_heads, d_llm=d_llm, q_lora_rank=q_lora_rank, kv_lora_rank=kv_lora_rank)

    

    print(f'reprogrammingLayer_mha.shape:{reprogrammingLayer_mha(target_embedding, source_embedding, value_embedding).shape}')
    print(f'reprogrammingLayer_mqa.shape:{reprogrammingLayer_mqa(target_embedding, source_embedding, value_embedding).shape}')
    print(f'reprogrammingLayer_gqa.shape:{reprogrammingLayer_gqa(target_embedding, source_embedding, value_embedding).shape}')
    print(f'reprogrammingLayer_mla.shape:{reprogrammingLayer_mla(target_embedding, source_embedding, value_embedding).shape}')
    


