import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEFeedForward(nn.Module):

    def __init__(
            self,
            hidden_size:int,
            intermediate_size:int,
            num_experts:int = 4,
            top_k:int = 2,
            aux_loss_coef:float = 0.01,
    ):
        super().__init__()
        self.last_expert_counts = None

        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if not 1<= top_k <=num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef


        self.router = nn.Linear(
            hidden_size, 
            num_experts,
            bias=False,
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    hidden_size,
                    intermediate_size,
                    bias=False
                    ),
                nn.SiLU(),
                nn.Linear(
                    intermediate_size,
                    hidden_size,
                    bias=False,
                ),
            )
            for _ in range(num_experts)
        ])

    def forward(self,x:torch.Tensor):
        # x: [batch, sequence, hidden]
        batch, seq_len,hidden = x.shape
        flat = x.reshape(-1,hidden)
        num_tokens = flat.shape[0]

        router_logits = self.router(flat)
        router_probs = F.softmax(router_logits.float(),dim = -1)

        top_probs, top_indices = torch.topk(
            router_probs,
            k=self.top_k,
            dim=-1,
        )

        # Renormalize selected routing weights to sum to one
        top_probs = top_probs / top_probs.sum(
            dim=-1,
            keepdim=True
            ).clamp_min(1e-9)

        output = torch.zeros_like(flat)

        # Expert dispatch: each token contributes to its selected experts
        for expert_idx, expert in enumerate(self.experts):
            token_idx,slot_idx = torch.where(
                top_indices == expert_idx
            )

            if token_idx.numel() == 0:
                continue

            expert_input = flat.index_select(0,token_idx)
            expert_output = expert(expert_input)

            weights = top_probs[token_idx,slot_idx].to(
                expert_output.dtype
            ).unsqueeze(-1)

            weighted_output = expert_output * weights

            output.index_add(
                0,token_idx,weighted_output
            )

        # Switch-style balancing objective:
        # encourage routing probability and actual assignment
        # frequency to be distributed across experts

        assignment_counts = F.one_hot(
            top_indices,
            num_classes=self.num_experts,
        ).float().sum(dim=(0,1))
        self.last_expert_counts = assignment_counts.detach()

        assignment_fraction = (
            assignment_counts / ( num_tokens * self.top_k)
        )

        mean_router_probability = router_probs.mean(dim=0)

        aux_loss = self.num_experts * torch.sum(
            assignment_fraction * mean_router_probability
        )

        return (
            output.reshape(batch,seq_len,hidden),
            aux_loss * self.aux_loss_coef,
        )

    