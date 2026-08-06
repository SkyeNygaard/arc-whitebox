import torch

from src.edge_dws import EdgeStateDWS, permute_hidden_layers


def test_hidden_permutation_invariance():
    torch.manual_seed(7)
    b, depth, width = 2, 4, 9
    model = EdgeStateDWS(depth=depth, label_dim=3, node_obs_dim=2, layer_obs_dim=3,
                         edge_channels=4, node_channels=6, token_channels=24,
                         passes=2, transformer_heads=4).eval()
    w = torch.randn(b, depth, width, width)
    no = torch.randn(b, depth + 1, width, 2)
    lo = torch.randn(b, depth, 3)
    perms = [torch.arange(width)]
    perms += [torch.randperm(width) for _ in range(depth - 1)]
    perms += [torch.arange(width)]
    wp, nop = permute_hidden_layers(w, perms, no)
    a = model(w, no, lo)
    c = model(wp, nop, lo)
    torch.testing.assert_close(a.correction, c.correction, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(a.scale, c.scale, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(a.confidence, c.confidence, atol=3e-5, rtol=3e-5)
