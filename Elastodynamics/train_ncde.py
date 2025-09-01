#!/usr/bin/env python
# ====================================================================
#  Elastodynamics NCDE-DeepONet
# ====================================================================

import time
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pathlib
import datetime
import numpy as np
import math
import jax
import jax.numpy as jnp
import jax.random as jr
import diffrax as dfx
import equinox as eqx
import jax.nn as jnn
import optax
import matplotlib.pyplot as plt
import imageio
import yaml
import matplotlib.tri as tri
from typing import Callable, Optional

from lookahead_jax import lookahead, lookahead_adamw, get_lookahead_step

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.85'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']  = 'false'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']      = 'true'
os.environ['TF_GPU_ALLOCATOR']              = 'cuda_malloc_async'

ENABLE_SHAPE_DEBUG = True
def debug_shape(label, obj, enabled=ENABLE_SHAPE_DEBUG):
    if enabled:
        if isinstance(obj, (jnp.ndarray, np.ndarray)):
            print(f"[SHAPE] {label}: {obj.shape} – {obj.dtype}")
        elif isinstance(obj, list) and len(obj) and hasattr(obj[0], "shape"):
            print(f"[SHAPE] {label}: {obj[0].shape} ×{len(obj)}")
        else:
            print(f"[SHAPE] {label}: {type(obj)}")

@eqx.filter_jit
def precompute_all_trunk_inputs(coords, time_steps):
    n_coords = coords.shape[0]
    t_coords = jnp.arange(time_steps) / time_steps
    
    coords_expanded = coords[:, None, :]
    t_expanded = t_coords[None, :, None]
    
    xy_grid = jnp.broadcast_to(coords_expanded, (n_coords, time_steps, 2))
    t_grid = jnp.broadcast_to(t_expanded, (n_coords, time_steps, 1))
    
    full_grid = jnp.concatenate([xy_grid, t_grid], axis=-1)
    return full_grid.reshape(-1, 3), full_grid

@eqx.filter_jit
def evaluate_full_field(model, x_branch, trunk_inputs_flat):
    return jax.vmap(lambda xt: model(x_branch, xt))(trunk_inputs_flat)

@eqx.filter_jit
def batch_evaluate_full_fields(model, x_branch_batch, trunk_inputs_flat):
    return jax.vmap(evaluate_full_field, in_axes=(None, 0, None))(
        model, x_branch_batch, trunk_inputs_flat
    )
            
@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    print("DEBUG: Loaded configuration from Hydra:")
    print(OmegaConf.to_yaml(cfg))
    
    print("DEBUG: Testing hyperparameter combination:")
    print(OmegaConf.to_yaml(cfg.solver))

    jax.config.update("jax_enable_x64", False)
    jax.config.update('jax_platform_name', 'gpu')
    jax.config.update('jax_default_matmul_precision', 'tensorfloat32')  

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.85'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

    print(f'jax devices = {jax.devices()}')
    print("\nA100 Optimization Status:")
    print(f"TF32 Enabled: {jax.config.jax_default_matmul_precision == 'tensorfloat32'}")
    print(f"GPU Type: {jax.devices()[0].device_kind}")
    print(f"Available VRAM: {jax.devices()[0].memory_stats()['bytes_limit']/1e9:.2f}GB")

    jaxf64 = False
    jax.config.update("jax_enable_x64", jaxf64)
    def get_float_dtype(jaxf64):
        return jnp.float64 if jaxf64 else jnp.float32
    float_dtype = get_float_dtype(jaxf64)
    print(f"float 64 = {jaxf64}")
    print(f'float type = {float_dtype}')


    seed = cfg.training.seed
    batch_size = cfg.training.batch_size
    epochs = cfg.training.epochs
    save_model_every = cfg.training.save_model_every
    initial_lr = cfg.training.learning_rate.initial
    final_lr   = cfg.training.learning_rate.final
    scheduler_method = cfg.training.learning_rate.scheduler

    data_dir = cfg.data.data_dir
    
    use_lookahead = cfg.training.get("use_lookahead", True)
    lookahead_alpha = cfg.training.get("lookahead_alpha", 0.5)
    lookahead_k = cfg.training.get("lookahead_k", 6)

    solver_map = {
        "Tsit5": dfx.Tsit5,
        "Dopri8": dfx.Dopri8,
        "Euler": dfx.Euler,
        "Heun": dfx.Heun,
    }

    solver_type = cfg.solver.type
    rtol        = float(cfg.solver.rtol)
    atol        = float(cfg.solver.atol)
    max_steps   = int(cfg.solver.max_steps)
    interpolation = cfg.solver.interpolation
    solver_class = solver_map[solver_type]
    solver = solver_class()

    print("DEBUG: Training parameters loaded from Hydra:")
    print(f"  Seed = {seed}, Batch Size = {batch_size}, Epochs = {epochs}, Save Model Every = {save_model_every}")
    print(f"  Learning Rate = {initial_lr} -> {final_lr}, Scheduler = {scheduler_method}")
    print("DEBUG: Solver parameters loaded from Hydra:")
    print(f"  Solver = {solver_type}, rtol = {rtol}, atol = {atol}, max_steps = {max_steps}, interpolation = {interpolation}")
    print("DEBUG: Lookahead optimizer parameters:")
    print(f"  Enabled = {use_lookahead}, Alpha = {lookahead_alpha}, K = {lookahead_k}")

    parent_results_folder = os.path.join("resultsV2", cfg.paths.results) if not cfg.paths.results.startswith("resultsV2") else cfg.paths.results
    model_checkpoints_subdir = cfg.paths.model_checkpoints
    training_metrics_image   = cfg.paths.training_metrics_image


    def create_training_round_folder(parent_folder):
        slurm_id = os.getenv("SLURM_JOB_ID", "local")
        os.makedirs(parent_folder, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        lookahead_suffix = "_lookahead" if use_lookahead else ""
        round_folder = os.path.join(parent_folder, f"training_round_{slurm_id}_fullfield_mse{lookahead_suffix}")
        os.makedirs(round_folder, exist_ok=True)
        print(f"DEBUG: Created training round folder: {round_folder}")
        return round_folder

    training_round_folder = create_training_round_folder(parent_results_folder)
    model_checkpoints_folder = os.path.join(training_round_folder, model_checkpoints_subdir)
    os.makedirs(model_checkpoints_folder, exist_ok=True)
    print(f"DEBUG: Model checkpoints will be saved in: {model_checkpoints_folder}")

    def build_schedule(total_steps: int) -> optax.Schedule:
        warmup = int(0.05 * total_steps)
        decay  = total_steps - warmup
        if scheduler_method == "warmup_cosine_decay":
            return optax.warmup_cosine_decay_schedule(
                init_value   = initial_lr,
                peak_value   = initial_lr,
                warmup_steps = warmup,
                decay_steps  = decay,
                end_value    = final_lr,
            )
        elif scheduler_method == "warmup_linear_decay":
            lin_warm = optax.linear_schedule(initial_lr / 10, initial_lr, warmup)
            def lin_tail(step):
                step = jnp.asarray(step, jnp.float32)
                return jnp.clip(
                    initial_lr - (initial_lr - final_lr) * (step - warmup) / max(1, decay),
                    final_lr,
                    initial_lr,
                )
            return optax.join_schedules([lin_warm, lin_tail], [warmup])
        else:                                      
            return optax.constant_schedule(initial_lr)



    x_train = jnp.array(np.load(os.path.join(data_dir, 'x_train.npy')), dtype=float_dtype)
    y_train = jnp.array(np.load(os.path.join(data_dir, 'y_train.npy')), dtype=float_dtype)
    x_test  = jnp.array(np.load(os.path.join(data_dir, 'x_test.npy')),  dtype=float_dtype)
    y_test  = jnp.array(np.load(os.path.join(data_dir, 'y_test.npy')),  dtype=float_dtype)
    coords = jnp.array(np.load(os.path.join(data_dir, 'coordinates.npy')), dtype=float_dtype)

    print('\nData shapes:')
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test  shape: {x_test.shape}")
    print(f"y_test  shape: {y_test.shape}")
    print(f"coords shape: {coords.shape}")

    print('\nData ranges:')
    print(f"x_train range: [{x_train.min():.2e}, {x_train.max():.2e}]")
    print(f"y_train range: [{y_train.min():.2e}, {y_train.max():.2e}]")
    print(f"x_test range:  [{x_test.min():.2e},  {x_test.max():.2e}]")
    print(f"y_test range:  [{y_test.min():.2e},  {y_test.max():.2e}]")
    print(f"coords range: [{coords.min():.2e}, {coords.max():.2e}]")

    print('\nDetailed statistics:')
    print('x_train statistics:')
    for i in range(x_train.shape[-1]):
        print(f"  Component {i}: Mean={x_train[..., i].mean():.2e}, Std={x_train[..., i].std():.2e}, "
              f"Min={x_train[..., i].min():.2e}, Max={x_train[..., i].max():.2e}")

    print('\ny_train statistics:')
    for i in range(y_train.shape[-1]):
        print(f"  Component {i}: Mean={y_train[..., i].mean():.2e}, Std={y_train[..., i].std():.2e}, "
              f"Min={y_train[..., i].min():.2e}, Max={y_train[..., i].max():.2e}")

    print('\ncoords statistics:')
    for i in range(coords.shape[-1]):
        print(f"  Component {i}: Mean={coords[..., i].mean():.2e}, Std={coords[..., i].std():.2e}, "
              f"Min={coords[..., i].min():.2e}, Max={coords[..., i].max():.2e}")


    def scale_data_inplace(x_train, x_test, y_train, y_test):
        x_train_np = np.asarray(x_train, dtype=np.float32)
        x_test_np  = np.asarray(x_test,  dtype=np.float32)
        y_train_np = np.asarray(y_train, dtype=np.float32)
        y_test_np  = np.asarray(y_test,  dtype=np.float32)

        normalization_method = cfg.data.get("normalization_method", "maxabs")
        if normalization_method == "standard":
            x_means = [np.mean(x_train_np[:, :, i]) for i in range(x_train_np.shape[-1])]
            x_stds  = [np.std(x_train_np[:, :, i]) for i in range(x_train_np.shape[-1])]
            y_means = [np.mean(y_train_np[..., i]) for i in range(y_train_np.shape[-1])]
            y_stds  = [np.std(y_train_np[..., i]) for i in range(y_train_np.shape[-1])]
            scale_factors = {"x": {"mean": x_means, "std": x_stds},
                             "y": {"mean": y_means, "std": y_stds}}

            def standardize_array(arr, means, stds, axis):
                arr = np.array(arr, copy=True)
                means = np.array(means, dtype=np.float32)
                stds  = np.array(stds, dtype=np.float32)
                view = np.moveaxis(arr, axis, -1)
                view[:] = (view - means) / stds
                return arr

            x_train_np = standardize_array(x_train_np, x_means, x_stds, axis=-1)
            x_test_np  = standardize_array(x_test_np,  x_means, x_stds, axis=-1)
            y_train_np = standardize_array(y_train_np, y_means, y_stds, axis=-1)
            y_test_np  = standardize_array(y_test_np,  y_means, y_stds, axis=-1)
        else:
            scale_factors = {
                'x': [
                    np.max(np.abs(x_train_np[:, :, 0])),
                    np.max(np.abs(x_train_np[:, :, 1])),
                    np.max(np.abs(x_train_np[:, :, 2]))
                ],
                'y': [
                    np.max(np.abs(y_train_np[:, :, :, 0])),
                    np.max(np.abs(y_train_np[:, :, :, 1]))
                ]
            }
            def scale_array(arr, factors, axis):
                arr = np.array(arr, copy=True)
                factors = np.array(factors, dtype=np.float32)
                view = np.moveaxis(arr, axis, -1)
                view[:] = view / factors
                return arr

            x_train_np = scale_array(x_train_np, scale_factors['x'], axis=-1)
            x_test_np  = scale_array(x_test_np,  scale_factors['x'], axis=-1)
            y_train_np = scale_array(y_train_np, scale_factors['y'], axis=-1)
            y_test_np  = scale_array(y_test_np, scale_factors['y'], axis=-1)

        print("\nNormalization method:", normalization_method)
        print("Scale factors / normalization statistics:")
        print(scale_factors)

        x_train = jax.device_put(x_train_np)
        x_test  = jax.device_put(x_test_np)
        y_train = jax.device_put(y_train_np)
        y_test  = jax.device_put(y_test_np)

        os.makedirs(data_dir, exist_ok=True)
        np.save(os.path.join(data_dir, 'scale_factors.npy'), scale_factors)
        
        return x_train, x_test, y_train, y_test, scale_factors

    x_train, x_test, y_train, y_test, scale_factors = scale_data_inplace(x_train, x_test, y_train, y_test)
    print(f"Post-scaling y_train range: [{y_train.min():.2e}, {y_train.max():.2e}]")
    print("Scale factors for y:", scale_factors['y'])

    print('\nDetailed statistics: Post scaling')
    print('x_train statistics:')
    for i in range(x_train.shape[-1]):
        print(f"  Component {i}: Mean={x_train[..., i].mean():.2e}, Std={x_train[..., i].std():.2e}, "
              f"Min={x_train[..., i].min():.2e}, Max={x_train[..., i].max():.2e}")

    print('\ny_train statistics:')
    for i in range(y_train.shape[-1]):
        print(f"  Component {i}: Mean={y_train[..., i].mean():.2e}, Std={y_train[..., i].std():.2e}, "
              f"Min={y_train[..., i].min():.2e}, Max={y_train[..., i].max():.2e}")

    HD_interact = x_train.shape[1]
    n_coords = coords.shape[0]
    print(f'\nHD_interact (number of time steps) = {HD_interact}')
    print(f'n_coords (number of spatial points) = {n_coords}')
    
    field_size = n_coords * HD_interact
    memory_per_sample = field_size * 2 * 4
    memory_per_batch = memory_per_sample * batch_size / (1024**3) 
    print(f'Field size per sample: {field_size} points')
    print(f'Estimated memory per batch: {memory_per_batch:.2f} GB')
    
    if memory_per_batch > 8.0:
        print(f"WARNING: Batch size {batch_size} may cause OOM. Consider reducing it.")
        recommended_batch = int(8.0 * (1024**3) / memory_per_sample)
        print(f"Recommended batch size: {recommended_batch}")

    steps_per_epoch = len(x_train) // batch_size
    total_steps = epochs * steps_per_epoch
    schedule = build_schedule(total_steps)
    
    print(f"Training setup: {steps_per_epoch} steps/epoch, {total_steps} total steps")

    branch_hidden_size = cfg.network.branch_hidden_size
    branch_num_layers  = cfg.network.branch_num_layers
    branch_feature_in  = 3 
    num_output_components = 2 
    
    trunk_feature_in   = 3 
    trunk_num_layers   = cfg.network.trunk_num_layers
    trunk_hidden_size  = cfg.network.trunk_hidden_size

    branch_output_dim = branch_hidden_size * num_output_components

    print('Branch Parameters:')
    print(f'branch feature size = {branch_feature_in}')
    print(f'branch layers size  = {branch_num_layers}')
    print(f'branch hidden size  = {branch_hidden_size}')
    print(f'branch output dim   = {branch_output_dim}')
    print(f"solver = {solver_type}, rtol = {rtol}, atol = {atol}, max_steps = {max_steps}")

    print('Trunk Parameters:')
    print(f'trunk feature size = {trunk_feature_in}')
    print(f'trunk layers size  = {trunk_num_layers}')
    print(f'trunk hidden size  = {trunk_hidden_size}')

    print("Pre-computing trunk input grid for full field evaluation...")
    trunk_inputs_flat, trunk_inputs_grid = precompute_all_trunk_inputs(coords, HD_interact)
    print(f"Trunk inputs shape: {trunk_inputs_flat.shape} (flattened), {trunk_inputs_grid.shape} (grid)")
    print(f"Total field size per sample: {n_coords * HD_interact}")

    class MLP(eqx.Module):
        layers: list
        norms : list
        act   : Callable
        use_ln: bool
        final_act: Optional[Callable]

        def __init__(self, sizes, *, key,
                    use_ln=True,
                    act=jnn.silu,
                    final_act: Optional[Callable] = None):
            n_layers = len(sizes) - 1
            ks = jr.split(key, n_layers)
            
            self.layers = [eqx.nn.Linear(a, b, key=ks[i])
                        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]))]
            self.norms = [eqx.nn.LayerNorm(b) for b in sizes[1:-1]] if use_ln else []
                    
            self.act = act
            self.use_ln = use_ln
            self.final_act = final_act

        def __call__(self, x):
            for i, lin in enumerate(self.layers):
                x = lin(x)
                is_last = i == len(self.layers) - 1
                
                if not is_last:
                    x = self.act(x)
                    if self.use_ln:
                        x = self.norms[i](x)
                elif self.final_act is not None:
                    x = self.final_act(x)
                    
            return x

    class CDEFunc(eqx.Module):
        mlp: MLP
        hidden: int
        inp: int
        
        def __init__(self, hidden_layers, branch_num_layers, *, key):
            self.inp, self.hidden = hidden_layers[0], hidden_layers[1]
            self.mlp = MLP(
                [self.hidden] + [self.hidden]*branch_num_layers + [self.hidden*self.inp],
                key=key,
                act=jnn.gelu, 
                final_act=None
            )
        
        def __call__(self, t, y, args):
            output = self.mlp(y)
            
            output = output * 0.1
            
            output = jnn.tanh(output / jnp.sqrt(self.hidden * self.inp))
            
            return output.reshape(self.hidden, self.inp)

    class NeuralCDE(eqx.Module):
        init:MLP; func:CDEFunc; read:MLP; evolving_out:bool
        def __init__(self, hidden_layers, evolving_out, branch_num_layers, *, key):
            k1,k2,k3 = jr.split(key,3)
            self.evolving_out = evolving_out
            self.init  = MLP([hidden_layers[0], hidden_layers[1]], key=k1)
            self.func  = CDEFunc(hidden_layers, branch_num_layers, key=k2)
            self.read  = MLP([hidden_layers[1],
                              branch_hidden_size*num_output_components], key=k3)
        def __call__(self, xs):
            ts = xs[...,0]
            ctrl = (dfx.LinearInterpolation if interpolation=="Linear"
                    else lambda t,x: dfx.CubicInterpolation(t, dfx.backward_hermite_coefficients(t,x)))(ts, xs)
            term = dfx.ControlTerm(self.func, ctrl).to_ode()
            y0   = self.init(ctrl.evaluate(ts[...,0]))
            sol  = dfx.diffeqsolve(term, solver, ts[0], ts[-1], None, y0,
                                stepsize_controller=dfx.PIDController(rtol, atol),
                                saveat=dfx.SaveAt(ts=ts) if self.evolving_out else dfx.SaveAt(t1=True),
                                max_steps=max_steps)
            z = sol.ys if self.evolving_out else sol.ys[-1][None]
            z = jax.vmap(self.read)(z)
            z = z[-1] if self.evolving_out else z[0]  
            return z.reshape(branch_hidden_size, num_output_components)

    class StackedNeuralCDE(eqx.Module):
        model1: NeuralCDE
        def __init__(self, model1):
            self.model1 = model1
        def __call__(self, xs):
            return self.model1(xs)

    class CDE_DeepONet(eqx.Module):
        branch_net: StackedNeuralCDE
        trunk_net: MLP

        def __init__(self, *, key):
            keys = jax.random.split(key, 2)
            ncde_mlp_layers = [branch_feature_in, branch_hidden_size, branch_hidden_size]
            o1 = NeuralCDE(hidden_layers=ncde_mlp_layers, evolving_out=False, 
                          branch_num_layers=branch_num_layers, key=keys[0])
            self.branch_net = StackedNeuralCDE(model1=o1)
            
            self.trunk_net = MLP(
                [trunk_feature_in] + [trunk_hidden_size]*trunk_num_layers + [branch_hidden_size],
                key=keys[1],
                use_ln=True,
                final_act=jnn.silu
            )
            
            print(f"Debug - Branch network input dim: {branch_feature_in}, hidden dim: {branch_hidden_size}")
            print(f"Debug - Trunk network input dim: {trunk_feature_in}, hidden dim: {trunk_hidden_size}")

        def __call__(self, x_branch, x_trunk):
            branch_out = self.branch_net(x_branch)
            trunk_out = self.trunk_net(x_trunk)
            output_model = jnp.einsum('hc,h->c', branch_out, trunk_out)
            return output_model

    key = jr.PRNGKey(seed)
    model_key, loader_key, coord_key = jr.split(key, 3)
    model = CDE_DeepONet(key=model_key)
    print(model)

    def count_parameters(model):
        """Count the total number of trainable parameters."""
        params = eqx.filter(model, eqx.is_inexact_array)
        
        def count_leaves(tree):
            return sum(x.size for x in jax.tree_util.tree_leaves(tree))
        
        return count_leaves(params)

    total_params = count_parameters(model)
    print(f"\n{'='*80}")
    print(f"Total Trainable Parameters: {total_params:,}")
    print(f"{'='*80}\n")

    if use_lookahead:
        print(f"Using Lookahead optimizer with α={lookahead_alpha}, k={lookahead_k}")
        optimizer = optax.chain(
            optax.clip_by_global_norm(0.5), 
            lookahead(
                optax.adamw(learning_rate=schedule, weight_decay=1e-4),
                alpha=lookahead_alpha,
                k=lookahead_k
            )
        )
    else:
        print("Using standard AdamW optimizer (no Lookahead)")
        optimizer = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adamw(learning_rate=schedule, weight_decay=1e-4),
        )

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))


    def create_epoch_dataloader(x_data, y_data, batch_size, key):
        """Create a dataloader that goes through all data once per epoch."""
        n_samples = x_data.shape[0]
        
        def epoch_iterator():
            nonlocal key
            key, shuffle_key = jr.split(key)
            indices = jr.permutation(shuffle_key, jnp.arange(n_samples))
            
            for i in range(0, n_samples - batch_size + 1, batch_size):
                batch_indices = indices[i:i + batch_size]
                yield x_data[batch_indices], y_data[batch_indices]
                
        return epoch_iterator, (n_samples - 1) // batch_size + 1

    lw0 = cfg.training.get("loss_weight_component_0", 1.0)
    lw1 = cfg.training.get("loss_weight_component_1", 1.0)
    loss_weights = jnp.array([lw0, lw1], dtype=float_dtype)
    print(f"Using loss weights: Component 0 (ux) = {lw0}, Component 1 (uy) = {lw1}")

    @eqx.filter_jit
    def compute_full_field_loss(model, x_branch_batch, y_true_batch, trunk_inputs_flat):

        batch_size = x_branch_batch.shape[0]
        n_coords = y_true_batch.shape[1]
        time_steps = y_true_batch.shape[2]
        
        predictions = batch_evaluate_full_fields(model, x_branch_batch, trunk_inputs_flat)
        
        predictions = predictions.reshape(batch_size, n_coords, time_steps, 2)
        
        sq_err = (y_true_batch - predictions) ** 2
        weighted_sq_err = sq_err * loss_weights[None, None, None, :] 
        
        return jnp.mean(weighted_sq_err)

    grad_loss = eqx.filter_value_and_grad(compute_full_field_loss)

    @eqx.filter_jit
    def make_step(model, xb, yb, trunk_inputs, opt_state):
        loss, grads = grad_loss(model, xb, yb, trunk_inputs)
        params = eqx.filter(model, eqx.is_inexact_array)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    def compute_metrics_full_field(m, x_te, y_te, trunk_inputs):
        tb = min(8, x_te.shape[0])  
        nb = math.ceil(x_te.shape[0] / tb)
        mse = mae = 0.0
        
        for i in range(nb):
            s, e = i*tb, min((i+1)*tb, x_te.shape[0])
            x_b, y_b = x_te[s:e], y_te[s:e]
            
            preds = batch_evaluate_full_fields(m, x_b, trunk_inputs)
            preds = preds.reshape(y_b.shape)
            
            mse_b = jnp.mean((y_b - preds) ** 2)
            mae_b = jnp.mean(jnp.abs(y_b - preds))
            
            mse += mse_b * (e - s)
            mae += mae_b * (e - s)
            
        mse /= x_te.shape[0]
        mae /= x_te.shape[0]
        return float(mse), float(mae)

    def ensure_directory_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

    def save_model(model, epoch, subdirectory):
        ensure_directory_exists(subdirectory)
        model_filename = pathlib.Path(subdirectory) / f'deeponet_trained_model_epoch_{epoch}.eqx'
        eqx.tree_serialise_leaves(model_filename, model)
        print(f"Model parameters saved to {model_filename}")

    print('DEBUG: Starting training loop with full field evaluation (no trunk sampling)')
    print('Using MSE loss for training, reporting RMSE for interpretability')
    if use_lookahead:
        print(f'Lookahead optimizer enabled: α={lookahead_alpha}, k={lookahead_k}')

    NNN = save_model_every
    loss_history = []
    test_metrics_history = []
    lr_history = []

    coord_key = jr.PRNGKey(seed)
    loader_key = jr.PRNGKey(seed + 1)

    print(f'Final shapes used for training and testing')
    debug_shape("x_train", x_train); debug_shape("y_train", y_train)
    debug_shape("x_test",  x_test);  debug_shape("y_test",  y_test)
    print("Using full field evaluation: no trunk sampling")

    overall_train_start = time.time()
    global_step = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_losses = []
        
        dataloader, n_batches = create_epoch_dataloader(
            x_train, y_train, batch_size, loader_key
        )
        loader_key, _ = jr.split(loader_key)
        
        for batch_idx, (xb, yb) in enumerate(dataloader()):
            if batch_idx == 0 and epoch == 0:
                print(f"Debug – xb {xb.shape}, yb {yb.shape}, trunk_inputs {trunk_inputs_flat.shape}")
            
            loss_mse, model, opt_state = make_step(model, xb, yb, trunk_inputs_flat, opt_state)
            loss_rmse = jnp.sqrt(loss_mse)
            
            epoch_losses.append(float(loss_rmse))
            
            global_step += 1
            current_lr = float(schedule(global_step))
            lr_history.append(current_lr)
            loss_history.append(float(loss_rmse))
            
            if use_lookahead:
                lookahead_step = get_lookahead_step(opt_state)
                slow_update_occurred = lookahead_step > 0 and lookahead_step % lookahead_k == 0
            else:
                slow_update_occurred = False
            
            if batch_idx % 10 == 0:
                slow_update_info = " [Slow update]" if slow_update_occurred else ""
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{n_batches} | "
                      f"Step {global_step} | LR {current_lr:.1e} | "
                      f"Loss RMSE {loss_rmse:.3e}{slow_update_info}")
        
        avg_epoch_loss = np.mean(epoch_losses)
        std_epoch_loss = np.std(epoch_losses)
        epoch_time = time.time() - epoch_start
        
        if epoch % 10 == 0:
            val_start = time.time()
            val_mse, val_mae = compute_metrics_full_field(
                model, x_test, y_test, trunk_inputs_flat
            )
            val_rmse = math.sqrt(val_mse)
            val_time = time.time() - val_start
            
            test_metrics_history.append({
                "epoch": epoch,
                "rmse": val_rmse,
                "mse_total": val_mse,
                "mae_total": val_mae
            })
            
            print(f"\nEpoch {epoch+1}/{epochs} Summary:")
            print(f"  Train RMSE: {avg_epoch_loss:.3e} ± {std_epoch_loss:.3e}")
            print(f"  Val MSE: {val_mse:.3e} (RMSE: {val_rmse:.3e}), MAE: {val_mae:.3e}")
            print(f"  Time: {epoch_time:.1f}s (Val: {val_time:.1f}s)")
            print(f"  Learning rate: {current_lr:.1e}")
            print("-" * 50)
        
        if (epoch + 1) % save_model_every == 0:
            save_model(model, epoch + 1, model_checkpoints_folder)

    overall_train_end = time.time()
    total_training_time = overall_train_end - overall_train_start
    print(f"\nDEBUG: Training complete. Total training time: {total_training_time:.2f} s ({total_training_time/60:.1f} min)")
    print("All results stored in:", training_round_folder)
    print('#' * 120)


    loss_arr = np.asarray(loss_history, dtype=np.float32)
    lr_arr   = np.asarray(lr_history,   dtype=np.float32)

    steps_all = np.arange(len(loss_arr))
    epochs_val = np.array([m["epoch"] for m in test_metrics_history])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    ax1.plot(steps_all, loss_arr, label='Training RMSE', alpha=0.7)
    if test_metrics_history:
        val_rmse = np.array([m['rmse'] for m in test_metrics_history], np.float32)
        val_steps = epochs_val * steps_per_epoch
        ax1.plot(val_steps, val_rmse, label='Validation RMSE', color='orange', marker='o')
    ax1.set(title='Loss curves (RMSE)', xlabel='Optimizer step', ylabel='Loss', yscale='log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if test_metrics_history:
        val_mse = np.array([m['mse_total'] for m in test_metrics_history], np.float32)
        ax2.plot(epochs_val, val_mse, label='Validation MSE', marker='o')
        ax2.legend()
    ax2.set(title='Validation MSE', xlabel='Epoch',
            ylabel='Error', yscale='log')
    ax2.grid(True, alpha=0.3)

    if test_metrics_history:
        mae_vals = np.array([m['mae_total'] for m in test_metrics_history], np.float32)
        ax3.plot(epochs_val, mae_vals, label='MAE', marker='s'); ax3.legend()
    ax3.set(title='Validation MAE', xlabel='Epoch',
            ylabel='Error', yscale='log')
    ax3.grid(True, alpha=0.3)

    ax4.plot(steps_all, lr_arr, 'g-')
    ax4.set(title='Learning-rate schedule', xlabel='Optimizer step',
            ylabel='Learning rate', yscale='log')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(training_round_folder, training_metrics_image)
    plt.savefig(plot_path, dpi=150); plt.close()
    print(f"Training metrics plot saved to {plot_path}")

    print('DEBUG: Starting evaluation of the model')

    def compute_sample_error(u_true, u_pred):
        denom = np.linalg.norm(u_true)
        return np.linalg.norm(u_true - u_pred) / denom if denom > 0 else np.inf

    def evaluate_model_full_field(model, x_test, y_test, trunk_inputs, n_samples=10):
        n_samples = min(n_samples, x_test.shape[0])
        n_coords = y_test.shape[1]
        n_t = y_test.shape[2]
        num_components = y_test.shape[-1]
        
        all_errors = []
        all_predictions = np.zeros((n_samples, n_coords, n_t, num_components))

        for i in range(n_samples):
            xb = x_test[i:i+1]
            yb = y_test[i]
            
            pred = evaluate_full_field(model, xb[0], trunk_inputs)
            all_predictions[i] = np.asarray(pred.reshape(n_coords, n_t, num_components))
            
            sample_errors = [
                compute_sample_error(yb[..., c], all_predictions[i, ..., c])
                for c in range(num_components)
            ]
            all_errors.append(sample_errors)

        all_errors = np.asarray(all_errors)

        comp_names = ["ux", "uy"]
        for c in range(num_components):
            comp_errs = all_errors[:, c]
            print(
                f"Component {c} ({comp_names[c]}) Error – "
                f"Min {comp_errs.min():.3f} | Mean {comp_errs.mean():.3f} | "
                f"Median {np.median(comp_errs):.3f} | 90-pct {np.percentile(comp_errs, 90):.3f} | "
                f"Max {comp_errs.max():.3f}"
            )

        return all_errors, all_predictions, y_test[:n_samples]

    errors, predictions, true_values = evaluate_model_full_field(
        model, x_test, y_test, trunk_inputs_flat
    )

    num_components  = true_values.shape[-1]
    combined_errors = np.mean(errors, axis=1)
    sorted_indices  = np.argsort(combined_errors)
    n               = len(combined_errors)

    if n >= 6:
        lowest_idx  = sorted_indices[0]
        p25_idx     = sorted_indices[int(0.25 * (n - 1))]
        median_idx  = sorted_indices[int(0.5  * (n - 1))]
        p75_idx     = sorted_indices[int(0.75 * (n - 1))]
        p90_idx     = sorted_indices[int(0.9  * (n - 1))]
        highest_idx = sorted_indices[-1]
        selected_indices = [lowest_idx, p25_idx, median_idx,
                            p75_idx, p90_idx, highest_idx]
        labels = ["Lowest", "25th pct", "Median",
                "75th pct", "90th pct", "Highest"]
    else:
        selected_indices = list(range(n))
        labels = [f"Sample {i}" for i in range(n)]

    print("Selected sample indices:", selected_indices)

    comp_names = ["ux", "uy"]
    for idx, label in zip(selected_indices, labels):
        pred_sample = predictions[idx]
        true_sample = np.array(true_values[idx])
        time_steps  = ([0, HD_interact // 2, HD_interact - 1]
                    if HD_interact >= 3 else list(range(HD_interact)))

        for t in time_steps:
            plt.figure(figsize=(12, 6))
            for c in range(num_components):
                plt.subplot(1, num_components, c + 1)
                coord_idx = coords.shape[0] // 2
                plt.plot(true_sample[coord_idx, :, c],
                        label="True", color='green')
                plt.plot(pred_sample[coord_idx, :, c],
                        '--', label="Pred", color='red')
                plt.axvline(x=t, color='black', linestyle=':')
                plt.title(f"Sample {idx} | {comp_names[c]} | Time {t}\n({label})")
                plt.xlabel("Time step"); plt.ylabel("Value")
                plt.legend()
            fname = f"sample_{idx}_{label}_time_{t}_v2.png"
            plt.tight_layout()
            plt.savefig(os.path.join(training_round_folder, fname))
            plt.close()

    median_sample_idx = median_idx if 'median_idx' in locals() else 0
    true_med = np.array(true_values[median_sample_idx])
    pred_med = predictions[median_sample_idx]

    num_timesteps = true_med.shape[1]
    sel_ts = ([0, num_timesteps // 2, num_timesteps - 1]
            if num_timesteps >= 3 else list(range(num_timesteps)))

    triang = tri.Triangulation(coords[:, 0], coords[:, 1])

    for t in sel_ts:
        for c in range(num_components):
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            true_vals = true_med[:, t, c]
            pred_vals = pred_med[:, t, c]
            diff_vals = np.abs(true_vals - pred_vals)

            cf0 = axs[0].tricontourf(triang, true_vals, 100, cmap='Blues')
            axs[0].set_title(f"True {comp_names[c]} field  t={t}")
            cb0 = fig.colorbar(cf0, ax=axs[0]); cb0.set_label("Value")

            cf1 = axs[1].tricontourf(triang, pred_vals, 100, cmap='Blues')
            axs[1].set_title(f"Predicted {comp_names[c]} field  t={t}")
            cb1 = fig.colorbar(cf1, ax=axs[1]); cb1.set_label("Value")

            cf2 = axs[2].tricontourf(triang, diff_vals, 100, cmap='Reds')
            axs[2].set_title(f"Abs. error {comp_names[c]}  t={t}")
            cb2 = fig.colorbar(cf2, ax=axs[2]); cb2.set_label("Error")

            plt.tight_layout()
            fname = f"heatmap_{comp_names[c]}_sample{median_sample_idx}_t{t}_v2.png"
            plt.savefig(os.path.join(training_round_folder, fname))
            plt.close()
            print(f"Saved contour {fname}")

    median_errors = {
        f"component_{c}_median_error": float(np.median(errors[:, c]))
        for c in range(num_components)
    }
    median_errors["combined_median_error"] = float(np.median(combined_errors))

    evaluation_info = {
        "sampling_strategy": "full_field_no_trunk_sampling",
        "field_size_per_sample": n_coords * HD_interact,
        "loss_function": "MSE",
        "loss_weights": {"ux": lw0, "uy": lw1},
        "training_time_minutes": total_training_time / 60,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": global_step,
        "batch_size": batch_size
    }

    lookahead_optimizer_info = {
        "enabled": use_lookahead,
        "alpha": lookahead_alpha if use_lookahead else None,
        "k": lookahead_k if use_lookahead else None
    }

    median_errors["evaluation_info"] = evaluation_info
    median_errors["lookahead_optimizer"] = lookahead_optimizer_info

    print("\nMedian errors per component:")
    comp_names = ["ux", "uy"]
    for c in range(num_components):
        key = f"component_{c}_median_error"
        print(f"  {comp_names[c]} ({key}): {median_errors[key]:.6f}")
    print(f"  Combined: {median_errors['combined_median_error']:.6f}")

    sorted_medians = sorted([(f"{comp_names[c]}", median_errors[f"component_{c}_median_error"]) 
                           for c in range(num_components)], key=lambda x: x[1])
    print("\nSorted median errors (best → worst):")
    for comp, err in sorted_medians:
        print(f"  {comp}: {err:.6f}")
    
    print("\nEvaluation summary:")
    print(f"  Full field evaluation: {n_coords} coords × {HD_interact} time steps = {n_coords * HD_interact} points")
    print(f"  Loss function: MSE (reporting RMSE for interpretability)")
    print(f"  Loss weights: ux = {lw0}, uy = {lw1}")
    print(f"  Batch size used: {batch_size}")
    
    if use_lookahead:
        print(f"\nLookahead optimizer: α={lookahead_alpha}, k={lookahead_k}")

    metrics_filename = "evaluation_metrics_fullfield_mse_lookahead.yaml" if use_lookahead else "evaluation_metrics_fullfield_mse.yaml"
    metrics_file = os.path.join(training_round_folder, metrics_filename)
    with open(metrics_file, "w") as f:
        yaml.dump(median_errors, f)
    print(f"Saved evaluation metrics to {metrics_file}")

if __name__ == "__main__":
    main()