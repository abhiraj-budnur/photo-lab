import torch

from src.models.transformer import TransformerConfig, SmallTransformer
from src.models.moe import MoEFeedForward

def test_moe_shapes_and_finite_aux_loss():

    moe = MoEFeedForward(
        hidden_size=32,
        intermediate_size=64,
        num_experts=4,
        top_k=2,
    )

    x = torch.randn(2,8,32)
    output, aux_loss = moe(x)

    assert output.shape == x.shape
    assert aux_loss.ndim == 0
    assert torch.isfinite(output).all()
    assert torch.isfinite(aux_loss)

def test_moe_backward():
    moe = MoEFeedForward(
        hidden_size=32,
        intermediate_size=64,
        num_experts=4,
        top_k=2,
    )

    x = torch.randn(2,8,32,requires_grad=True)
    output, aux_loss = moe(x)

    loss = output.square().mean() + aux_loss
    loss.backward()

    assert x.grad is not None
    assert moe.router.weight.grad is not None
    assert torch.isfinite(moe.router.weight.grad).all()

def test_transformer_with_moe():
    config = TransformerConfig(
        vocab_size = 128,
        hidden_size=48,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=96,
        max_seq_len=32,
        use_moe=True,
        num_experts=4,
        moe_top_k=2,
        moe_intermediate_size=64,
    )

    model = SmallTransformer(config)
    tokens = torch.randint(0,128,(2,12))

    result = model(tokens,labels=tokens)

    assert torch.isfinite(result["loss"])
    assert torch.isfinite(result["lm_loss"])
    assert torch.isfinite(result["aux_loss"])

    result["loss"].backward()

    assert model.layers[0].mlp.router.weight.grad is not None
    