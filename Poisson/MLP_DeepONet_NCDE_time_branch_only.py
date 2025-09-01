#!/usr/bin/env python
# ====================================================================
#  NCDE-DeepONet – transient Poisson problem
#  WITH SPATIAL-ONLY TRUNK (NO TIME IN TRUNK)
# ====================================================================
import time, os, math, pathlib, datetime, yaml
import functools
import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import diffrax as dfx
import equinox as eqx
import optax
import matplotlib.pyplot as plt
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
def precompute_spatial_trunk_inputs(coords):
    return coords

@eqx.filter_jit
def evaluate_full_field(model, x_branch, trunk_inputs):
    return model(x_branch, trunk_inputs)

@eqx.filter_jit
def batch_evaluate_full_fields(model, x_branch_batch, trunk_inputs):
    return jax.vmap(evaluate_full_field, in_axes=(None, 0, None))(
        model, x_branch_batch, trunk_inputs
    )

@hydra.main(config_path="configs", config_name="config_ncde", version_base=None)
def main(cfg: DictConfig):

    print("Loaded YAML configuration:")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("-" * 80)

    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_platform_name", "gpu")
    jax.config.update("jax_default_matmul_precision", "tensorfloat32")

    dev = jax.devices()[0]
    print(f"Device : {dev.device_kind}  –  VRAM {dev.memory_stats()['bytes_limit']/1e9:.1f} GB")
    print("-" * 80)

    seed            = cfg.training.seed
    batch_size      = cfg.training.batch_size
    epochs          = cfg.training.epochs
    save_every      = cfg.training.save_model_every
    lr_init         = cfg.training.learning_rate.initial
    lr_final        = cfg.training.learning_rate.final
    scheduler_name  = cfg.training.learning_rate.scheduler
    data_dir        = cfg.data.data_dir
    
    use_lookahead = cfg.training.get("use_lookahead", True)
    lookahead_alpha = cfg.training.get("lookahead_alpha", 0.5)
    lookahead_k = cfg.training.get("lookahead_k", 6)
    
    solver_cls = {"Tsit5": dfx.Tsit5, "Dopri8": dfx.Dopri8,
                  "Euler": dfx.Euler, "Heun":   dfx.Heun}[cfg.solver.type]
    rtol, atol = float(cfg.solver.rtol), float(cfg.solver.atol)
    max_steps  = int(cfg.solver.max_steps)
    interpolation = cfg.solver.interpolation

    parent_res = (os.path.join("resultsV2", cfg.paths.results)
                  if not cfg.paths.results.startswith("resultsV2")
                  else cfg.paths.results)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    slurm_id = os.getenv("SLURM_JOB_ID", "local")
    
    lookahead_suffix = "_lookahead" if use_lookahead else ""
    round_dir = pathlib.Path(parent_res) / f"training_round_{slurm_id}_{ts}_spatial_trunk_only{lookahead_suffix}"
    ckpt_dir  = round_dir / cfg.paths.model_checkpoints
    round_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results → {round_dir}")

    def build_schedule(total_steps:int)->optax.Schedule:
        warmup = int(0.01 * total_steps)
        decay  = total_steps - warmup
        if scheduler_name == "warmup_cosine_decay":
            return optax.warmup_cosine_decay_schedule(
                init_value   = lr_init,
                peak_value   = lr_init,
                warmup_steps = warmup,
                decay_steps  = decay,
                end_value    = lr_final)
        elif scheduler_name == "warmup_linear_decay":
            lin = optax.linear_schedule(lr_init/10, lr_init, warmup)
            def tail(step):
                step = jnp.asarray(step, jnp.float32)
                return jnp.clip(lr_init - (lr_init-lr_final)*(step-warmup)/max(1, decay),
                                lr_final, lr_init)
            return optax.join_schedules([lin, tail], [warmup])
        else:
            return optax.constant_schedule(lr_init)

    x_train = jnp.asarray(np.load(f"{data_dir}/x_train.npy"), dtype=jnp.float32)
    y_train = jnp.asarray(np.load(f"{data_dir}/y_train.npy"), dtype=jnp.float32)
    x_test  = jnp.asarray(np.load(f"{data_dir}/x_test.npy"),  dtype=jnp.float32)
    y_test  = jnp.asarray(np.load(f"{data_dir}/y_test.npy"),  dtype=jnp.float32)
    coords  = jnp.asarray(np.load(f"{data_dir}/coordinates.npy"), dtype=jnp.float32)

    debug_shape("x_train", x_train); debug_shape("y_train", y_train)

    x_train = x_train[:, 1:, :]
    x_test  = x_test[:, 1:, :]

    if y_train.shape[1] == x_train.shape[1] + 1:
        y_train = jnp.moveaxis(y_train[:, 1:, :, :], 1, 2)
        y_test  = jnp.moveaxis(y_test[:, 1:, :, :], 1, 2)
    elif y_train.shape[2] == x_train.shape[1] + 1:
        y_train = y_train[:, :, 1:, :]
        y_test  = y_test[:, :, 1:, :]
    else:
        raise ValueError(f"Unrecognised y layout {y_train.shape}")

    assert y_train.shape[1] == coords.shape[0], \
        f"Expected y[:, Ncoord, T, 1] but got {y_train.shape}"

    HD_interact = x_train.shape[1]
    n_coords = coords.shape[0]
    print(f'HD_interact = {HD_interact}, n_coords = {n_coords}')

    def scale_data(x_tr, x_te, y_tr, y_te):
        norm = cfg.data.get("normalization_method", "maxabs")
        if norm == "standard":
            xm = jnp.mean(x_tr, axis=(0,1)); xs = jnp.std(x_tr, axis=(0,1))
            ym = jnp.mean(y_tr);              ys = jnp.std(y_tr)
            x_tr = (x_tr - xm)/xs; x_te = (x_te - xm)/xs
            y_tr = (y_tr - ym)/ys; y_te = (y_te - ym)/ys
            stats = {"x":{"mean":xm.tolist(),"std":xs.tolist()},
                     "y":{"mean":float(ym),"std":float(ys)}}
        else:
            xmax = jnp.max(jnp.abs(x_tr), axis=(0,1))
            ymax = jnp.max(jnp.abs(y_tr))
            x_tr = x_tr/xmax; x_te = x_te/xmax
            y_tr = y_tr/ymax; y_te = y_te/ymax
            stats = {"x":xmax.tolist(),"y":float(ymax)}
        np.save(round_dir/"scale_factors.npy", stats)
        return x_tr, x_te, y_tr, y_te

    x_train, x_test, y_train, y_test = scale_data(x_train, x_test, y_train, y_test)

    branch_feature_in      = 2
    trunk_feature_in       = 2
    num_output_components  = 1
    branch_hidden_size     = cfg.network.branch_hidden_size
    branch_layers          = cfg.network.branch_num_layers
    trunk_hidden_size      = cfg.network.trunk_hidden_size
    trunk_layers           = cfg.network.trunk_num_layers

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
        
        def __init__(self, hidden_layers, *, key):
            self.inp, self.hidden = hidden_layers[0], hidden_layers[1]
            self.mlp = MLP(
                [self.hidden] + [self.hidden]*branch_layers + [self.hidden*self.inp],
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
        def __init__(self, hidden_layers, evolving_out, *, key):
            k1,k2,k3 = jr.split(key,3)
            self.evolving_out = evolving_out
            self.init  = MLP([hidden_layers[0], hidden_layers[1]], key=k1)
            self.func  = CDEFunc(hidden_layers, key=k2)
            self.read  = MLP([hidden_layers[1],
                              branch_hidden_size], key=k3)
        def __call__(self, xs):
            ts = xs[...,0]
            ctrl = (dfx.LinearInterpolation if interpolation=="Linear"
                    else lambda t,x: dfx.CubicInterpolation(t, dfx.backward_hermite_coefficients(t,x)))(ts, xs)
            term = dfx.ControlTerm(self.func, ctrl).to_ode()
            y0   = self.init(ctrl.evaluate(ts[...,0]))
            sol  = dfx.diffeqsolve(term, solver_cls(), ts[0], ts[-1], None, y0,
                                   stepsize_controller=dfx.PIDController(rtol, atol),
                                   saveat=dfx.SaveAt(ts=ts) if self.evolving_out else dfx.SaveAt(t1=True),
                                   max_steps=max_steps)
            z = sol.ys if self.evolving_out else sol.ys[-1][None]
            z = jax.vmap(self.read)(z)
            
            if self.evolving_out:
                return z
            else:
                return z[0].reshape(branch_hidden_size, num_output_components)

    class DeepONet(eqx.Module):
        branch:NeuralCDE; trunk:MLP
        def __init__(self, *, key):
            kb,kt = jr.split(key,2)
            self.branch = NeuralCDE([branch_feature_in,
                                     branch_hidden_size,
                                     branch_hidden_size],
                                     evolving_out=True, key=kb)
            self.trunk  = MLP([trunk_feature_in] + 
                              [trunk_hidden_size]*trunk_layers +
                              [branch_hidden_size],
                              key=kt, final_act=jnn.silu)
        def __call__(self, x_branch, x_trunk):
            
            b = self.branch(x_branch) 
            
            t = jax.vmap(self.trunk)(x_trunk)
            
            result = jnp.einsum("th,nh->nt", b, t)[..., None]
            return result

    steps_per_epoch = len(x_train) // batch_size
    total_steps = epochs * steps_per_epoch
    schedule = build_schedule(total_steps)
    
    print(f"Training setup: {steps_per_epoch} steps/epoch, {total_steps} total steps")

    model_key = jr.PRNGKey(seed)
    model     = DeepONet(key=model_key)

    if use_lookahead:
        print(f"Using Lookahead optimizer with α={lookahead_alpha}, k={lookahead_k}")
        opt = optax.chain(
            optax.clip_by_global_norm(0.5),
            lookahead(
                optax.adamw(learning_rate=schedule, weight_decay=1e-4),
                alpha=lookahead_alpha,
                k=lookahead_k
            )
        )
    else:
        print("Using standard AdamW optimizer (no Lookahead)")
        opt = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adamw(learning_rate=schedule, weight_decay=1e-4)
        )

    opt_state   = opt.init(eqx.filter(model, eqx.is_inexact_array))
    global_step = 0

    print("Pre-computing spatial trunk inputs...")
    trunk_inputs = precompute_spatial_trunk_inputs(coords)
    print(f"Trunk inputs shape: {trunk_inputs.shape} (spatial coordinates only)")

    def create_epoch_dataloader(x_data, y_data, batch_size, key):
        n_samples = x_data.shape[0]
        
        def epoch_iterator():
            nonlocal key
            key, shuffle_key = jr.split(key)
            indices = jr.permutation(shuffle_key, jnp.arange(n_samples))
            
            for i in range(0, n_samples - batch_size + 1, batch_size):
                batch_indices = indices[i:i + batch_size]
                yield x_data[batch_indices], y_data[batch_indices]
                
        return epoch_iterator, (n_samples - 1) // batch_size + 1

    lw = jnp.array([cfg.training.get("loss_weight_component_0", 1.0)], dtype=jnp.float32)

    @eqx.filter_jit
    def compute_full_field_loss(model, x_branch_batch, y_true_batch, trunk_inputs):
        batch_size = x_branch_batch.shape[0]
        
        predictions = batch_evaluate_full_fields(model, x_branch_batch, trunk_inputs)
        
        sq_err = (y_true_batch - predictions) ** 2
        weighted_sq_err = sq_err * lw
        
        return jnp.mean(weighted_sq_err)

    grad_loss = eqx.filter_value_and_grad(compute_full_field_loss)

    @eqx.filter_jit
    def step(m, xb, yb, trunk_inputs, opt_state):
        l, gr = grad_loss(m, xb, yb, trunk_inputs)
        up, opt_state = opt.update(gr, opt_state, eqx.filter(m, eqx.is_inexact_array))
        m = eqx.apply_updates(m, up)
        return l, m, opt_state

    def compute_metrics_full_field(m, x_te, y_te, trunk_inputs):
        tb = min(16, x_te.shape[0])
        nb = math.ceil(x_te.shape[0] / tb)
        mse = mae = 0.0
        
        for i in range(nb):
            s, e = i*tb, min((i+1)*tb, x_te.shape[0])
            x_b, y_b = x_te[s:e], y_te[s:e]
            
            preds = batch_evaluate_full_fields(m, x_b, trunk_inputs)
            
            mse_b = jnp.mean((y_b - preds) ** 2)
            mae_b = jnp.mean(jnp.abs(y_b - preds))
            
            mse += mse_b * (e - s)
            mae += mae_b * (e - s)
            
        mse /= x_te.shape[0]
        mae /= x_te.shape[0]
        return float(mse), float(mae)

    print("\nFinal shapes used for training and testing:")
    debug_shape("x_train", x_train); debug_shape("y_train", y_train)
    debug_shape("x_test",  x_test);  debug_shape("y_test",  y_test)
    print("Architecture: Spatial-only trunk (x,y), time-evolving branch features")
    print("Using MSE loss for training, reporting RMSE for interpretability")
    if use_lookahead:
        print(f"Lookahead optimizer enabled: α={lookahead_alpha}, k={lookahead_k}")

    loader_key = jr.PRNGKey(seed + 1)
    
    loss_history, lr_history = [], []
    test_metrics_history = []
    overall_train_start = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_losses = []
        
        dataloader, n_batches = create_epoch_dataloader(
            x_train, y_train, batch_size, loader_key
        )
        loader_key, _ = jr.split(loader_key)
        
        for batch_idx, (xb, yb) in enumerate(dataloader()):
            if batch_idx == 0 and epoch == 0:
                print(f"Debug – xb {xb.shape}, yb {yb.shape}, trunk_inputs {trunk_inputs.shape}")
            
            loss_mse, model, opt_state = step(model, xb, yb, trunk_inputs, opt_state)
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
                      f"RMSE {loss_rmse:.3e}{slow_update_info}")
        
        avg_epoch_loss = np.mean(epoch_losses)
        std_epoch_loss = np.std(epoch_losses)
        epoch_time = time.time() - epoch_start
        
        if epoch % 10 == 0:
            val_start = time.time()
            val_mse, val_mae = compute_metrics_full_field(
                model, x_test, y_test, trunk_inputs
            )
            val_rmse = math.sqrt(val_mse)
            val_time = time.time() - val_start
            
            test_metrics_history.append({
                "epoch": epoch,
                "rmse": val_rmse,
                "mse": val_mse,
                "mae": val_mae
            })
            
            print(f"\nEpoch {epoch+1}/{epochs} Summary:")
            print(f"  Train RMSE: {avg_epoch_loss:.3e} ± {std_epoch_loss:.3e}")
            print(f"  Val MSE: {val_mse:.3e} (RMSE: {val_rmse:.3e}), MAE: {val_mae:.3e}")
            print(f"  Time: {epoch_time:.1f}s (Val: {val_time:.1f}s)")
            print(f"  Learning rate: {current_lr:.1e}")
            print("-" * 50)
        
        if (epoch + 1) % save_every == 0:
            ckpt_path = ckpt_dir / f"deeponet_epoch_{epoch+1}.eqx"
            eqx.tree_serialise_leaves(ckpt_path, model)
            print(f"Saved model → {ckpt_path}")

    total_time = time.time() - overall_train_start
    print(f"\nTraining finished in {total_time/60:.1f} min")

    steps_all = np.arange(len(loss_history))
    epochs_val = np.array([m["epoch"] for m in test_metrics_history])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    ax1.plot(steps_all, loss_history, label="Train RMSE", alpha=0.7)
    if test_metrics_history:
        val_rmse = np.array([m["rmse"] for m in test_metrics_history], np.float32)
        val_steps = epochs_val * steps_per_epoch
        ax1.plot(val_steps, val_rmse, label="Val RMSE", color='orange', marker='o')
    ax1.set(title="Loss curves (RMSE)", xlabel="Optimizer step",
            ylabel="Error", yscale="log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if test_metrics_history:
        val_mse = np.array([m["mse"] for m in test_metrics_history], np.float32)
        val_mae = np.array([m["mae"] for m in test_metrics_history], np.float32)
        ax2.plot(epochs_val, val_mse, label="MSE", marker='o')
        ax2.plot(epochs_val, val_mae, label="MAE", marker='s')
        ax2.legend()
    ax2.set(title="Validation metrics", xlabel="Epoch",
            ylabel="Error", yscale="log")
    ax2.grid(True, alpha=0.3)

    zoom_start = int(0.75 * len(loss_history))
    ax3.plot(steps_all[zoom_start:], loss_history[zoom_start:])
    ax3.set(title="Train loss (RMSE, last 25%)", xlabel="Optimizer step",
            ylabel="Error")
    ax3.grid(True, alpha=0.3)

    ax4.plot(steps_all, lr_history)
    ax4.set(title="Learning-rate schedule", xlabel="Optimizer step",
            ylabel="LR", yscale="log")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = round_dir / cfg.paths.training_metrics_image
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Training metrics plot saved to {plot_file}")

    def rel_error(u_true, u_pred):
        return np.linalg.norm(u_true - u_pred) / np.linalg.norm(u_true)

    def evaluate_model_full_field(model, x_test, y_test, trunk_inputs, n_samples=10):
        n_samples = min(n_samples, x_test.shape[0])
        n_coords = y_test.shape[1]
        n_t = y_test.shape[2]
        
        preds = np.zeros((n_samples, n_coords, n_t, 1), dtype=np.float32)
        errors = np.zeros((n_samples, 1), dtype=np.float32)

        for i in range(n_samples):
            xb = x_test[i:i+1]
            yb = y_test[i]
            
            pred = evaluate_full_field(model, xb[0], trunk_inputs)
            preds[i] = np.asarray(pred)
            
            errors[i, 0] = rel_error(np.asarray(yb[:, :, 0]), preds[i, :, :, 0])
            
        return preds, errors

    predictions, errors = evaluate_model_full_field(
        model, x_test, y_test, trunk_inputs
    )
    np.save(round_dir / "predictions.npy", predictions)
    np.save(round_dir / "errors.npy", errors)

    combined = errors.mean(axis=1)
    idx_sorted = np.argsort(combined)
    sel_idx = [idx_sorted[0],
               idx_sorted[int(0.25*(len(combined)-1))],
               idx_sorted[int(0.50*(len(combined)-1))],
               idx_sorted[int(0.75*(len(combined)-1))],
               idx_sorted[-1]]
    labels = ["best", "25-pct", "median", "75-pct", "worst"]

    triang = tri.Triangulation(coords[:, 0], coords[:, 1])
    for idx, label in zip(sel_idx, labels):
        true_sample = np.asarray(y_test[idx, :, :, 0])
        pred_sample = predictions[idx, :, :, 0]
        for t in [0, HD_interact // 2, HD_interact - 1]:
            fig, axs = plt.subplots(1, 3, figsize=(18, 5))
            for k, (field, ttl) in enumerate(
                zip([true_sample[:, t],
                    pred_sample[:, t],
                    np.abs(true_sample[:, t] - pred_sample[:, t])],
                    ["True", "Predicted", "|Error|"])):
                cf = axs[k].tricontourf(triang, field, levels=100, cmap="Blues")
                axs[k].set_title(f"{ttl}  t={t}  ({label})")
                fig.colorbar(cf, ax=axs[k])
            plt.tight_layout()
            fname = round_dir / f"heatmap_{label}_t{t}.png"
            plt.savefig(fname, dpi=150)
            plt.close()

    median_err = float(np.median(errors))
    
    eval_metrics = {
        "median_relative_error": median_err,
        "loss_function": "MSE", 
        "architecture": {
            "type": "spatial_only_trunk",
            "description": "Trunk processes only (x,y), branch outputs time-evolving features",
            "trunk_inputs": "spatial_coordinates_only", 
            "branch_output": "time_series_of_features"
        },
        "lookahead_optimizer": {
            "enabled": use_lookahead,
            "alpha": lookahead_alpha if use_lookahead else None,
            "k": lookahead_k if use_lookahead else None
        },
        "training_time_minutes": total_time / 60,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": global_step,
        "n_spatial_points": n_coords,
        "n_time_steps": HD_interact
    }
    
    yaml.dump(eval_metrics, open(round_dir / "evaluation_metrics.yaml", "w"))
    print(f"\nFinal evaluation:")
    print(f"  Median relative error: {median_err:.3e}")
    print(f"  Architecture: Spatial-only trunk (x,y), time-evolving branch")
    print(f"  Spatial points: {n_coords}, Time steps: {HD_interact}")
    print(f"  Total field size per sample: {n_coords * HD_interact}")
    if use_lookahead:
        print(f"  Lookahead optimizer: α={lookahead_alpha}, k={lookahead_k}")

if __name__ == "__main__":
    main()