import torch
import torch.nn as nn
from .transformer_encoder import TransformerEncoder
from .transformer_decoder import TransformerDecoder

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_seq_len, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, vocab_size, max_seq_len, dropout)
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, d_ff, vocab_size, max_seq_len, dropout)
    
    def forward(self, src, tgt, tgt_mask=None, memory_mask=None):
        enc_output = self.encoder(src)
        output = self.decoder(tgt, enc_output, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return output

if __name__ == "__main__":
    num_layers = 6
    d_model = 512
    num_heads = 8
    d_ff = 2048
    vocab_size = 10000
    max_seq_len = 100
    dropout = 0.1

    batch_size = 2
    src_seq_len = 10  # 소스 시퀀스 길이 (인코더 입력)
    tgt_seq_len = 8  # 타겟 시퀀스 길이 (디코더 입력)

    model = Transformer(num_layers, d_model, num_heads, d_ff, vocab_size, max_seq_len, dropout)

    src = torch.randint(0, vocab_size, (batch_size, src_seq_len))  # (batch_size, src_seq_len)
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))  # (batch_size, tgt_seq_len)

    def generate_memory_mask(src_seq_len, tgt_seq_len, device='cpu'):

        mask = torch.zeros(tgt_seq_len, src_seq_len, device=device)
        mask = mask.masked_fill(mask == 0, float('-inf'))  # 마스크 적용
        return mask.unsqueeze(0)  # (1, tgt_seq_len, src_seq_len) 형태로 반환

    # 마스크 생성
    tgt_mask = model.decoder.self_attention.generate_square_subsequent_mask(tgt_seq_len).unsqueeze(0)  # (1, tgt_seq_len, tgt_seq_len)
    memory_mask = generate_memory_mask(src_seq_len, tgt_seq_len, device=src.device)


    output = model(src, tgt, tgt_mask=tgt_mask, memory_mask=memory_mask)

    # 결과 출력
    print("Output shape:", output.shape)  # 예상 출력: (batch_size, tgt_seq_len, vocab_size)