import argparse
import dataclasses
import functools
import logging
import os
import platform

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
from openpi.training.subtask_data_loader import create_lerobot_subtask_data_loader
from openpi.training.subtask_data_loader import create_subtask_jsonl_data_loader


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    real_action_dim: int,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, real_action_dim=real_action_dim, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pi05 subtask-generation on LeRobot or flattened JSONL data.")
    parser.add_argument("--config-name", required=True, help="Existing training config name, e.g. right_pi05_20")
    parser.add_argument(
        "--data-source",
        required=True,
        help="LeRobot repo id/local dataset path, or JSONL manifest path if --data-format=jsonl",
    )
    parser.add_argument(
        "--data-format",
        choices=("lerobot", "jsonl"),
        default="lerobot",
        help="Use the segment-level LeRobot loader by default; JSONL remains available for debugging.",
    )
    parser.add_argument("--exp-name", required=True, help="Experiment name for checkpoints and wandb")
    parser.add_argument("--real-action-dim", type=int, default=14, help="Real action dimension before padding")
    parser.add_argument("--state-key", default="state", help="Observation state key in the LeRobot frame dict")
    parser.add_argument("--action-key", default="actions", help="Action key in the LeRobot frame dict")
    parser.add_argument("--tasks-index", type=int, default=0, help="Which entry in episode.tasks to use as high-level prompt")
    parser.add_argument("--base-image-key", default="face_view", help="LeRobot image feature key for base_0_rgb")
    parser.add_argument("--left-image-key", default="left_wrist_view", help="LeRobot image feature key for left_wrist_0_rgb")
    parser.add_argument("--right-image-key", default="right_wrist_view", help="LeRobot image feature key for right_wrist_0_rgb")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch size")
    parser.add_argument("--num-train-steps", type=int, default=None, help="Override config num_train_steps")
    parser.add_argument("--num-workers", type=int, default=None, help="Override config num_workers")
    parser.add_argument("--save-interval", type=int, default=None, help="Override config save_interval")
    parser.add_argument("--log-interval", type=int, default=None, help="Override config log_interval")
    parser.add_argument("--wandb-enabled", action="store_true", help="Enable wandb logging")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing checkpoint dir")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    return parser.parse_args()


def main():
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

    args = parse_args()
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    config = _config.get_config(args.config_name)
    config = dataclasses.replace(
        config,
        exp_name=args.exp_name,
        batch_size=args.batch_size or config.batch_size,
        num_train_steps=args.num_train_steps or config.num_train_steps,
        num_workers=args.num_workers if args.num_workers is not None else config.num_workers,
        save_interval=args.save_interval or config.save_interval,
        log_interval=args.log_interval or config.log_interval,
        wandb_enabled=args.wandb_enabled,
        overwrite=args.overwrite,
        resume=args.resume,
    )

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    local_batch_size = config.batch_size // jax.process_count()
    if args.data_format == "lerobot":
        image_key_mapping = {
            "base_0_rgb": args.base_image_key,
            "left_wrist_0_rgb": args.left_image_key,
            "right_wrist_0_rgb": args.right_image_key,
        }
        data_loader = create_lerobot_subtask_data_loader(
            args.data_source,
            batch_size=local_batch_size,
            action_horizon=config.model.action_horizon,
            action_dim=config.model.action_dim,
            max_token_len=config.model.max_token_len,
            shuffle=True,
            num_workers=config.num_workers,
            seed=config.seed,
            sharding=data_sharding,
            image_key_mapping=image_key_mapping,
            state_key=args.state_key,
            action_key=args.action_key,
            tasks_index=args.tasks_index,
        )
    else:
        data_loader = create_subtask_jsonl_data_loader(
            args.data_source,
            batch_size=local_batch_size,
            action_horizon=config.model.action_horizon,
            action_dim=config.model.action_dim,
            max_token_len=config.model.max_token_len,
            shuffle=True,
            num_workers=config.num_workers,
            seed=config.seed,
            sharding=data_sharding,
        )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized subtask data loader ({args.data_format}):\n{training_utils.array_tree_to_info(batch)}")

    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, mesh=mesh)

    ptrain_step = jax.jit(
        functools.partial(train_step, config, real_action_dim=args.real_action_dim),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)

        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main()
