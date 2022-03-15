from typing import Optional
from numpy import identity
import torch

from allennlp.modules import Seq2SeqEncoder
from allennlp_models.lm.modules.seq2seq_encoders import BidirectionalLanguageModelTransformer
from allennlp_models.lm.modules.seq2seq_encoders.bidirectional_lm_transformer import MultiHeadedAttention


def fix_to_identity(linear: torch.nn.Linear):
    """Fix a `Linear` layer to compute the identity transformation."""
    linear.weight.requires_grad_(False)
    linear.bias.requires_grad_(False)
    linear.weight.set_(torch.eye(linear.in_features, linear.out_features))
    linear.bias.mul_(0.)


def fix_allennlp_transformer(transformer: torch.nn.Module):
    """Set all the heads in the AllenNLP transformer to use identity attention."""
    for module in transformer.modules():
        if isinstance(module, MultiHeadedAttention):
            query, key, _, _ = module.linears
            fix_to_identity(query)
            fix_to_identity(key)


@Seq2SeqEncoder.register("identity_transformer")
class IdentityTransformer(BidirectionalLanguageModelTransformer):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.1,
        input_dropout: float = None,
        return_all_layers: bool = False,
        identity_attention: bool = True,
    ) -> None:
        super().__init__(input_dim, hidden_dim, num_layers, dropout, input_dropout, return_all_layers)
        if identity_attention:
            fix_allennlp_transformer(self)


if __name__ == "__main__":
    # transformer = IdentityPytorchTransformer(80, 80, 80)
    transformer = IdentityTransformer(80, 80, 8)
    for mod in transformer.modules():
        if isinstance(mod, MultiHeadedAttention):
            query, key, _, _ = mod.linears
            torch.testing.assert_allclose(query.weight, torch.eye(query.in_features, query.out_features))
            torch.testing.assert_allclose(key.weight, torch.eye(key.in_features, key.out_features))
            torch.testing.assert_allclose(query.bias, torch.zeros_like(query.bias))
            torch.testing.assert_allclose(key.bias, torch.zeros_like(key.bias))
    print("Passed identity tests")
    
    transformer = IdentityTransformer(80, 80, 8, identity_attention=False)
    for mod in transformer.modules():
        if isinstance(mod, MultiHeadedAttention):
            query, key, _, _ = mod.linears
            assert not (query.weight == torch.eye(query.in_features, query.out_features)).all()
            assert not (key.weight == torch.eye(key.in_features, key.out_features)).all()
    print("Passed non identity tests")
