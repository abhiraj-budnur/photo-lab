import math

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.moe import MoEFeedForward

@dataclass
class TransformerConfig:
    vocab_size:int = 8192
    hidden_size:int  = 384
    num_layers:int  = 6
    num_heads:int  = 6
    num_kv_heads:int  = 2
    intermediate_size:int = 1024
    max_seq_len :int = 512
    rope_theta:float = 10_000.0
    norm_eps:float = 1e-5
    dropout:float = 0.0
    use_moe:bool = False
    num_experts:int = 4
    moe_top_k:int = 2
    moe_intermediate_size:int = 256
    moe_aux_loss_coef:float = 0.01

class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self,x:torch.Tensor)->torch.Tensor:
        # Accumulate the normalization in FP32 for stability
        x_fp32 = x.float()
        rms = torch.sqrt(
            x_fp32.pow(2).mean( dim= -1, keepdim = True) + self.eps
        )
        return (x_fp32 * rms).to(x.dtype) * self.weight

def rotate_half(x:torch.Tensor)->torch.Tensor:
    # Adjacent pair rotation
    x = x.reshape(*x.shape[:-1],-1,2)
    x0,x1 = x.unbind(dim=-1)
    return torch.stack((-x1,x0),dim=-1).flatten(-2)

class RotaryEmbedding(nn.Module):

    inv_buffer :torch.Tensor
    def __init__(self,head_dim:int,theta:float = 10_000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        
        inv_freq = 1.0 / (
            theta**(
                torch.arange(0,head_dim,2,dtype=torch.float32)
                / head_dim
            )
        )
        self.register_buffer("inv_buffer",inv_freq,persistent=False)

    def forward(self,x:torch.Tensor, offset:int = 0):
        # x.shape: [batch,head,sequence,head_dim]
        seq_len = x.shape[-2]
        positions = torch.arange(
            offset,offset+seq_len,
            device=x.device,
            dtype=torch.float32,
        )
        freqs = torch.outer(positions,self.inv_buffer.float())
        angles = torch.repeat_interleave(freqs,2,dim=-1)

        cos = angles.cos().to(x.dtype)[None,None,:,:]
        sin = angles.sin().to(x.dtype)[None,None,:,:]

        return x * cos + rotate_half(x) * sin

class GroupedQueryAttention(nn.Module):

    def __init__(self,config:TransformerConfig):
        super().__init__()
        assert config.hidden_size % config.num_heads == 0
        assert config.num_heads % config.num_kv_heads == 0

        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads

        if self.head_dim % 2:
            raise ValueError("Attention head must be even.")

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias = False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.rope = RotaryEmbedding(self.head_dim,config.rope_theta)
        self.dropout  = config.dropout

    def forward(self,x:torch.Tensor)->torch.Tensor:

        batch,seq_len,hidden = x.shape

        q = self.q_proj(x).view(
            batch, seq_len, self.num_heads, self.head_dim
        ).transpose(1,2)

        k = self.k_proj(x).view(
            batch, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1,2)

        v = self.v_proj(x).view(
            batch, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1,2)

        q = self.rope(q)
        k = self.rope(k)

        # Expand KV heads logically for grouped-query attention.
        # repeat intereleave is simple 
        groups = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(groups,dim=1)
        v = v.repeat_interleave(groups,dim=1)

        out = F.scaled_dot_product_attention(
            q,k,v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        out = out.transpose(1,2).contiguous().view(
            batch,seq_len,hidden
        )

        return self.o_proj(out)

class SwiGLU(nn.Module):

    def __init__(self,config:TransformerConfig):
        super().__init__()

        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )

        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )

        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias= False,
        )

    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(x)) * self.up_proj(x)
        )

class TransformerBlock(nn.Module):

    def __init__(self,config:TransformerConfig):

        super().__init__()

        self.attn_norm = RMSNorm(
            config.hidden_size,
            config.norm_eps
            )
        self.attn = GroupedQueryAttention(config)

        self.mlp_norm = RMSNorm(
            config.hidden_size,
            config.norm_eps
            )
        self.use_moe = config.use_moe

        if self.use_moe:
            self.mlp = MoEFeedForward(
                hidden_size = config.hidden_size,
                intermediate_size = config.moe_intermediate_size,
                num_experts = config.num_experts,
                top_k = config.moe_top_k,
                aux_loss_coef = config.moe_aux_loss_coef,
            )
        else:
            self.mlp = SwiGLU(config)

    def forward(self,x:torch.Tensor):
        x = x + self.attn(self.attn_norm(x))

        if self.use_moe:
            mlp_output,aux_loss = self.mlp(self.mlp_norm(x))
            x = x + mlp_output
        else:
            x = x + self.mlp(self.mlp_norm(x))
            aux_loss = x.new_zeros(())

        return x, aux_loss

class SmallTransformer(nn.Module):

    def __init__(self,config:TransformerConfig):
        super().__init__()
        self.config = config

        self.embd_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.num_layers)
        ])

        self.final_norm = RMSNorm(config.hidden_size,config.norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module,nn.Linear):
            nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
        elif isinstance(module,nn.Embedding):
            nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(
            self,
            input_ids:torch.Tensor,
            labels:torch.Tensor | None = None,
            ):

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have [batch,sequence]")

        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("Sequence exceeds max_seq_len")

        x = self.embd_tokens(input_ids)
        aux_losses = []

        for layer in self.layers:
            x, aux_loss = layer(x)
            aux_losses.append(aux_loss)

        logits = self.lm_head(self.final_norm(x))

        if aux_losses:
            aux_loss = torch.stack(aux_losses).sum()
        else:
            aux_loss = logits.new_zeros(())

        lm_loss = None
        loss = None

        if labels is not None:
            # predict token from t+1 form positions up through t
            shift_logits = logits[:,:-1,:].contiguous()
            shift_labels = labels[:,1:].contiguous()

            lm_loss = F.cross_entropy(
                shift_logits.reshape(-1,shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
            loss = lm_loss + aux_loss

        return {
            "logits":logits,
            "loss":loss,
            "lm_loss":lm_loss,
            "aux_loss":aux_loss
            }

    @torch.no_grad()
    def generate(
        self,
        input_ids:torch.Tensor,
        max_new_tokens:int = 64,
        temperature:float = 1.0,
        top_k: int | None = 50,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.eval()

        for _ in range(max_new_tokens):
            context = input_ids[:,-self.config.max_seq_len:]
            logits = self(context)["logits"][:,-1,:]
            logits = logits / temperature

            if top_k is not None:
                k = min(top_k,logits.shape[-1])
                threshold = torch.topk(logits,k,dim=-1).values[:,-1:]
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            probs = F.softmax(logits.float(),dim=-1)
            next_token = torch.multinomial(probs,num_samples=1)
            input_ids = torch.cat([input_ids,next_token],dim=-1)

        return input_ids