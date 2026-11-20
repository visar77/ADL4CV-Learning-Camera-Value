"""Check the adapter against an independent reference of official LVSM equations.

Reference: Haian-Jin/LVSM model/transformer.py and
model/LVSM_scene_encoder_decoder.py (accessed 2026-11-12).
Uses explicit softmax attention and literal patch slicing on a tiny input, so it
runs on CPU without xformers. This verifies checkpoint computation on identical
prepared inputs; it does not test the original CUDA/xformers kernel.
"""
from __future__ import annotations

from pathlib import Path
import argparse

from torch.nn import functional as F
import torch

from gain_predictor.geometry import apply_frame, context_frame, rays
from gain_predictor.models.lvsm import LVSMFeatureBackbone


def reference_block(block, x, valid=None):
    dim = x.shape[-1]
    normalized = F.layer_norm(x, (dim,), block.norm1.weight, eps=1e-5)
    q, k, v = F.linear(normalized, block.attn.to_qkv.weight).chunk(3, -1)
    shape = (x.shape[0], x.shape[1], block.heads, block.head_dimension)
    def rms(values, weight):
        values = values.reshape(shape).float()
        return (values / torch.sqrt(values.square().mean(-1, keepdim=True) + 1e-5)) * weight
    q = rms(q, block.attn.q_norm.weight).transpose(1, 2)
    k = rms(k, block.attn.k_norm.weight).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    scores = q @ k.transpose(-1, -2) / block.head_dimension ** .5
    if valid is not None:
        scores = scores.masked_fill(~valid[:, None, None, :], -torch.inf)
    attention = (scores.softmax(-1) @ v).transpose(1, 2).reshape_as(x)
    x = x + F.linear(attention, block.attn.fc.weight)
    normalized = F.layer_norm(x, (dim,), block.norm2.weight, eps=1e-5)
    return x + F.linear(F.gelu(F.linear(normalized, block.mlp.mlp[0].weight)), block.mlp.mlp[2].weight)


def reference_tokenize(linear, values):
    b, v, c, h, w = values.shape
    patches = [values[..., y:y+8, x:x+8].permute(0, 1, 3, 4, 2).reshape(b*v, -1)
               for y in range(0, h, 8) for x in range(0, w, 8)]
    return F.linear(torch.stack(patches, dim=1), linear.weight)


def reference_forward(model, images, poses, query, mask, intrinsics, query_intrinsics):
    b, n, _, h, w = images.shape
    poses = poses.reshape(b, n, 4, 4)
    frame, scale = context_frame(poses, mask)
    poses = apply_frame(poses, frame, scale)
    origin, direction = rays(poses, intrinsics, h, w)
    posed = torch.cat((images * 2 - 1, torch.cross(origin, direction, dim=2), direction), dim=2)
    tokens = reference_tokenize(model.image_tokenizer[1], posed)
    patches = tokens.shape[1]
    latent = model.n_light_field_latent[:model.latent_tokens].unsqueeze(0).expand(b, -1, -1)
    x = torch.cat((latent, tokens.reshape(b, n*patches, -1)), dim=1)
    valid = torch.cat((torch.ones(b, model.latent_tokens, dtype=torch.bool, device=mask.device),
                       mask.repeat_interleave(patches, 1)), 1)
    for layer in model.transformer_encoder:
        x = reference_block(layer, x, valid)
    origin, direction = rays(apply_frame(query.reshape(b, 1, 4, 4), frame, scale),
                             query_intrinsics[:, None], h, w)
    target = reference_tokenize(model.target_pose_tokenizer[1],
                                 torch.cat((torch.cross(origin, direction, dim=2), direction), dim=2))
    x = F.layer_norm(torch.cat((target, x[:, :model.latent_tokens]), 1), (x.shape[-1],),
                     model.transformer_input_layernorm_decoder.weight, eps=1e-5)
    for layer in model.transformer_decoder:
        x = reference_block(layer, x)
    return x[:, :target.shape[1]].mean(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.manual_seed(7)
    backbone = LVSMFeatureBackbone(args.checkpoint, image_size=16, latent_tokens=4)
    images = torch.rand(1, 2, 3, 16, 16)
    poses = torch.eye(4).repeat(1, 2, 1, 1)
    poses[0, 1, 0, 3] = 1
    query = poses[:, 0].clone().flatten(1)
    intrinsics = torch.tensor([1.1, 1.1, .5, .5]).repeat(1, 2, 1)
    mask = torch.ones(1, 2, dtype=torch.bool)
    with torch.no_grad():
        actual = backbone.encoder(images, poses.flatten(2), query, mask, intrinsics, intrinsics[:, 0])
        expected = reference_forward(backbone.encoder, images, poses, query, mask, intrinsics, intrinsics[:, 0])
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    print(f"Reference parity passed; max absolute difference {(actual-expected).abs().max().item():.8g}")


if __name__ == "__main__":
    main()
