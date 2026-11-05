"""Shared per-finger current-frame tactile MLP, no attention or contact heuristics."""
from flax import nnx
import jax
import jax.numpy as jnp


class SimplePerFingerTactileMLP(nnx.Module):
    def __init__(self, *, rngs):
        self.linear1 = nnx.Linear(720, 512, rngs=rngs)
        self.linear2 = nnx.Linear(512, 256, rngs=rngs)
        self.linear3 = nnx.Linear(256, 128, rngs=rngs)
        self.norm = nnx.LayerNorm(128, rngs=rngs)

    def __call__(self, tactile, transforms):
        if tactile.xyz_m.shape[1:] != (600, 3):
            raise ValueError("simple tactile MLP requires 600 canonically ordered taxels")
        xyz = tactile.xyz_m.astype(jnp.float32)
        force = tactile.force.astype(jnp.float32)
        valid = tactile.point_mask.astype(bool) & jnp.isfinite(xyz).all(-1) & jnp.isfinite(force).all(-1)
        xyz = transforms.tactile_xyz(jnp.where(valid[..., None], xyz, 0))
        # The shared force transform also returns a norm, which is deliberately not an input.
        force = transforms.force(jnp.where(valid[..., None], force, 0), jnp.zeros_like(tactile.force_norm)).force
        features = jnp.where(valid[..., None], jnp.concatenate([xyz, force], axis=-1), 0).reshape(-1, 5, 720)
        tokens = self.norm(self.linear3(jax.nn.swish(self.linear2(jax.nn.swish(self.linear1(features))))))
        mask = valid.reshape(-1, 5, 120).any(axis=-1)
        raw = jnp.where(valid[..., None], tactile.xyz_m, 0).reshape(-1, 5, 120, 3)
        center = raw.sum(axis=2) / jnp.maximum(valid.reshape(-1, 5, 120).sum(axis=2, keepdims=True), 1)
        return jnp.where(mask[..., None], tokens, 0), mask, center, {"tactile_token_count": mask.sum(axis=1)}
