from dm08_output import *
from trace_utils import trace, param_summary, Wrap
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, output_layer):
        """
        :param encoder: 编码器实例
        :param decoder: 解码器实例
        :param src_embed: 源序列嵌入层实例
        :param tgt_embed: 目标序列嵌入层实例
        :param output_layer: 输出层实例
        """
        super(Transformer, self).__init__()
        self.encoder = encoder  # 编码器
        self.decoder = decoder  # 解码器
        self.src_embed = src_embed  # 源序列嵌入层
        self.tgt_embed = tgt_embed  # 目标序列嵌入层
        self.output_layer = output_layer  # 输出层

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        前向传播
        :param src: 源序列输入 (B, L_src)
        :param tgt: 目标序列输入 (B, L_tgt)
        :param src_mask: 源序列掩码 (B, L_src, L_src)
        :param tgt_mask: 目标序列掩码 (B, L_tgt, L_tgt)
        :return: 输出张量 (B, L_tgt, vocab_size)
        """
        # 对源序列进行嵌入和位置编码，然后传入编码器
        memory = self.encoder(self.src_embed(src), src_mask)  # 编码器输出 (B, L_src, d_model)
        
        # 对目标序列进行嵌入和位置编码，然后传入解码器，使用编码器的输出作为memory
        decoder_output = self.decoder(self.tgt_embed(tgt), memory, src_mask=src_mask, tgt_mask=tgt_mask)  # 解码器输出 (B, L_tgt, d_model)
        
        # 将解码器的输出传入输出层，得到最终的预测结果
        output = self.output_layer(decoder_output)  # 输出张量 (B, L_tgt, vocab_size)
        
        return output

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, heads=8, dropout=0.1):
    """
    创建一个完整的Transformer模型
    :param src_vocab: 源序列词汇表大小
    :param tgt_vocab: 目标序列词汇表大小
    :param N: 编码器和解码器层数
    :param d_model: 嵌入维度
    :param d_ff: 前馈网络中间层的维度
    :param heads: 注意力头数
    :param dropout: dropout率
    :return: Transformer模型实例
    """
    # 创建编码器层和解码器层实例
    encoder_layer = EncoderLayer(d_model, heads, d_ff, dropout)
    decoder_layer = DecoderLayer(d_model, heads, d_ff, dropout)
    
    # 创建编码器和解码器实例
    encoder = Encoder(encoder_layer, N)
    decoder = Decoder(decoder_layer, N)
    
    # 创建源序列和目标序列嵌入层实例
    src_embed = nn.Sequential(Embedding(src_vocab, d_model), PositionalEncoding(d_model))
    tgt_embed = nn.Sequential(Embedding(tgt_vocab, d_model), PositionalEncoding(d_model))
    
    # 创建输出层实例
    output_layer = Output(tgt_vocab, d_model)
    
    # 创建并返回Transformer模型实例
    model = Transformer(encoder, decoder, src_embed, tgt_embed, output_layer)
    
    return model


def use_transformer():
    """
    使用Transformer模型进行测试
    """
    src_vocab = 1000  # 源序列词汇表大小
    tgt_vocab = 1000  # 目标序列词汇表大小
    model = make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, heads=8, dropout=0.1)
    
    print('Transformer模型:', model)
    
    # 创建一个示例输入
    src = torch.randint(0, src_vocab, (2, 10))  # (B, L_src)
    tgt = torch.randint(0, tgt_vocab, (2, 10))  # (B, L_tgt)
    
    # 创建掩码
    src_mask = torch.ones(src.size(0), src.size(1), src.size(1))  # (B, L_src, L_src)
    tgt_mask = torch.ones(tgt.size(0), tgt.size(1), tgt.size(1))  # (B, L_tgt, L_tgt)
    
    print('源序列输入:', src)
    print('目标序列输入:', tgt)
    
    # 前向传播
    output = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)  # 输出张量 (B, L_tgt, vocab_size)
    
    print('Transformer输出形状:', output.shape)  # (2, 10, vocab_size)
    print('Transformer输出:', output)  # 输出张量
    print('----'*20)
    trace(model, src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
    return output    

if __name__ == "__main__":
    use_transformer()