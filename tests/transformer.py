import torch

from src.models.transformer import (
    TransformerConfig,
    SmallTransformer,
    RotaryEmbedding,
)

def tiny_config():
    return TransformerConfig(
        vocab_size=128,
        hidden_size=48,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=96,
        max_seq_len=32
    )

def test_forward_shapes():
    model = SmallTransformer(tiny_config())
    tokens = torch.randint(0,128,(2,12))

    output = model(tokens)

    assert output["loss"] is not None
    assert output["loss"].ndim == 0
    assert torch.isfinite(output["loss"])

    output["loss"].backward()

    assert model.embd_tokens.weight.grad is not None
    assert torch.isfinite(model.embd_tokens.weight.grad).all()

def test_rope_preserves_shape():
    rope = RotaryEmbedding(head_dim=12)
    x = torch.randn(2,4,16,12)

    result = rope(x)

    assert result.shape == x.shape
    assert torch.isfinite(result).all()

def test_casual_attention_does_not_leak_future():
    torch.manual_seed(123)
    model = SmallTransformer(tiny_config()).eval()

    a = torch.randint(0,128,(1,12))
    b = a.clone()
    b[:,8:] = torch.randint(0,128,(1,4))

    with torch.no_grad():
        out_a = model(a)["logits"]
        out_b = model(b)["logits"]

    # Changing future tokens must not change earlier predictions
    torch.testing.assert_close(
        out_a[:,:8],
        out_b[:,:8],
        rtol=1e-5,
        atol=1e-5,
    )