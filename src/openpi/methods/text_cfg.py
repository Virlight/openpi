"""Text classifier-free guidance support for Pi0/Pi05 configs.

This module contains the reusable model wrapper and model-input transforms for
text CFG. Config modules should import this file and register their own derived
``TrainConfig`` entries.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import einops
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_config
from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
import openpi.transforms as _transforms


@dataclasses.dataclass(frozen=True)
class TokenizePromptWithUncond(_transforms.DataTransformFn):
    """Tokenize the normal prompt and an empty-text prompt with the same state."""

    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        uncond_tokens, uncond_token_masks = self.tokenizer.tokenize("", state)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_masks,
            # Pi0/Pi05 do not otherwise consume these optional FAST fields.
            "token_ar_mask": uncond_tokens.astype(np.int32),
            "token_loss_mask": uncond_token_masks.astype(bool),
        }


@dataclasses.dataclass(frozen=True)
class TextCFGModelTransformFactory:
    """Model transforms that emit conditional and unconditional prompt tokens."""

    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        TokenizePromptWithUncond(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _:
                raise ValueError(f"Text CFG only supports Pi0/Pi05, got {model_config.model_type}")


class Pi0TextCFG(_pi0.Pi0):
    """Pi0/Pi05 wrapper that supports text dropout and text CFG sampling."""

    def __init__(self, config: TextCFGPi0Config, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.text_dropout_prob = config.text_dropout_prob

    def _uncond_observation(self, observation: _model.Observation) -> _model.Observation:
        if observation.token_ar_mask is None or observation.token_loss_mask is None:
            raise ValueError(
                "Text CFG requires unconditional tokens. Use TextCFGModelTransformFactory "
                "so token_ar_mask/token_loss_mask carry the empty-text prompt tokens."
            )
        return dataclasses.replace(
            observation,
            tokenized_prompt=observation.token_ar_mask.astype(observation.tokenized_prompt.dtype),
            tokenized_prompt_mask=observation.token_loss_mask.astype(jnp.bool_),
        )

    def _apply_text_dropout(self, rng: at.KeyArrayLike, observation: _model.Observation) -> _model.Observation:
        if self.text_dropout_prob <= 0:
            return observation

        uncond_observation = self._uncond_observation(observation)
        drop = jax.random.bernoulli(
            rng,
            p=self.text_dropout_prob,
            shape=observation.tokenized_prompt_mask.shape[:-1],
        )[..., None]
        return dataclasses.replace(
            observation,
            tokenized_prompt=jnp.where(drop, uncond_observation.tokenized_prompt, observation.tokenized_prompt),
            tokenized_prompt_mask=jnp.where(
                drop,
                uncond_observation.tokenized_prompt_mask,
                observation.tokenized_prompt_mask,
            ),
        )

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, text_dropout_rng, noise_rng, time_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        if train:
            observation = self._apply_text_dropout(text_dropout_rng, observation)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    def _build_prefix_cache(self, observation: _model.Observation):
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        return prefix_tokens, prefix_mask, kv_cache

    def _predict_velocity(
        self,
        observation: _model.Observation,
        prefix_tokens,
        prefix_mask,
        kv_cache,
        x_t,
        time,
    ):
        batch_size = observation.state.shape[0]
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation,
            x_t,
            jnp.broadcast_to(time, batch_size),
        )
        suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        assert full_attn_mask.shape == (
            batch_size,
            suffix_tokens.shape[1],
            prefix_tokens.shape[1] + suffix_tokens.shape[1],
        )
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        text_cfg_scale: float | at.Float[at.Array, ""] = 1.0,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        uncond_observation = self._uncond_observation(observation)

        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        cond_prefix_tokens, cond_prefix_mask, cond_kv_cache = self._build_prefix_cache(observation)
        uncond_prefix_tokens, uncond_prefix_mask, uncond_kv_cache = self._build_prefix_cache(uncond_observation)

        def step(carry):
            x_t, time = carry
            v_cond = self._predict_velocity(
                observation,
                cond_prefix_tokens,
                cond_prefix_mask,
                cond_kv_cache,
                x_t,
                time,
            )
            v_uncond = self._predict_velocity(
                uncond_observation,
                uncond_prefix_tokens,
                uncond_prefix_mask,
                uncond_kv_cache,
                x_t,
                time,
            )
            v_t = v_uncond + text_cfg_scale * (v_cond - v_uncond)
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0


@dataclasses.dataclass(frozen=True)
class TextCFGPi0Config(pi0_config.Pi0Config):
    """Pi0Config that creates the text-CFG model wrapper."""

    text_dropout_prob: float = 0.1

    @override
    def create(self, rng: at.KeyArrayLike) -> Pi0TextCFG:
        return Pi0TextCFG(self, rngs=nnx.Rngs(rng))


def copy_model_config(base_model: pi0_config.Pi0Config, *, text_dropout_prob: float = 0.1) -> TextCFGPi0Config:
    return TextCFGPi0Config(
        dtype=base_model.dtype,
        paligemma_variant=base_model.paligemma_variant,
        action_expert_variant=base_model.action_expert_variant,
        action_dim=base_model.action_dim,
        action_horizon=base_model.action_horizon,
        max_token_len=base_model.max_token_len,
        pi05=base_model.pi05,
        discrete_state_input=base_model.discrete_state_input,
        pytorch_compile_mode=base_model.pytorch_compile_mode,
        text_dropout_prob=text_dropout_prob,
    )


def with_text_cfg(
    base: Any,
    name: str,
    *,
    model_transforms: Any | None = None,
    text_dropout_prob: float = 0.1,
) -> Any:
    """Return a separate TrainConfig derived from an existing Pi0/Pi05 config."""
    if not isinstance(base.model, pi0_config.Pi0Config):
        raise ValueError(f"Text-CFG config {name!r} requires a Pi0Config base model.")

    model = copy_model_config(base.model, text_dropout_prob=text_dropout_prob)
    try:
        data = dataclasses.replace(
            base.data,
            model_transforms=model_transforms or TextCFGModelTransformFactory(),
        )
    except TypeError as exc:
        raise ValueError(f"Text-CFG config {name!r} requires data config with model_transforms.") from exc

    return dataclasses.replace(
        base,
        name=name,
        model=model,
        data=data,
        freeze_filter=model.get_freeze_filter(),
    )
