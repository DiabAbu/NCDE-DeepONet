from typing import NamedTuple, Any, Tuple
import jax
import jax.numpy as jnp
import optax
from optax._src import base


class LookaheadState(NamedTuple):
    slow_params: base.Params
    fast_state: base.OptState 
    step: jnp.ndarray
    
    
def lookahead(
    base_optimizer: base.GradientTransformation,
    alpha: float = 0.5,
    k: int = 6
) -> base.GradientTransformation:

    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    
    def init_fn(params: base.Params) -> LookaheadState:
        slow_params = jax.tree_map(lambda p: jnp.array(p), params)
        fast_state = base_optimizer.init(params)
        step = jnp.array(0, dtype=jnp.int32)
        
        return LookaheadState(
            slow_params=slow_params,
            fast_state=fast_state,
            step=step
        )
    
    def update_fn(
        updates: base.Updates,
        state: LookaheadState,
        params: base.Params
    ) -> Tuple[base.Updates, LookaheadState]:
        fast_updates, new_fast_state = base_optimizer.update(
            updates, state.fast_state, params
        )
        
        new_step = state.step + 1
        
        should_update_slow = (new_step % k) == 0
        
        new_fast_params = optax.apply_updates(params, fast_updates)
        
        def update_slow_params(slow, fast):
            return jax.lax.cond(
                should_update_slow,
                lambda: slow + alpha * (fast - slow),
                lambda: slow
            )
        
        new_slow_params = jax.tree_map(
            update_slow_params,
            state.slow_params,
            new_fast_params
        )
        
        def compute_final_updates(param, slow, fast_update):
            slow_update = slow - param
            return jax.lax.cond(
                should_update_slow,
                lambda: slow_update,
                lambda: fast_update
            )
        
        final_updates = jax.tree_map(
            compute_final_updates,
            params,
            new_slow_params,
            fast_updates
        )
        
        new_state = LookaheadState(
            slow_params=new_slow_params,
            fast_state=new_fast_state,
            step=new_step
        )
        
        return final_updates, new_state
    
    return base.GradientTransformation(init_fn, update_fn)


def lookahead_adam(
    learning_rate: float,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    alpha: float = 0.5,
    k: int = 6
) -> base.GradientTransformation:
    """Adam with Lookahead."""
    return lookahead(
        optax.adam(learning_rate=learning_rate, b1=b1, b2=b2, eps=eps),
        alpha=alpha,
        k=k
    )


def lookahead_adamw(
    learning_rate: float,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 1e-4,
    alpha: float = 0.5,
    k: int = 6
) -> base.GradientTransformation:
    """AdamW with Lookahead."""
    return lookahead(
        optax.adamw(
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            eps=eps,
            weight_decay=weight_decay
        ),
        alpha=alpha,
        k=k
    )


def get_lookahead_step(opt_state: Any) -> int:
    if isinstance(opt_state, LookaheadState):
        return int(opt_state.step)
    elif hasattr(opt_state, '__iter__'):
        for state in opt_state:
            if isinstance(state, LookaheadState):
                return int(state.step)
    return 0


if __name__ == "__main__":
    import numpy as np
    
    params = {
        'layer1': {'weight': jnp.ones((3, 4)), 'bias': jnp.zeros(4)},
        'layer2': {'weight': jnp.ones((4, 2)), 'bias': jnp.zeros(2)}
    }
    
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1e-3,
        warmup_steps=100,
        decay_steps=900,
        end_value=1e-5
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        lookahead(
            optax.adamw(learning_rate=lr_schedule, weight_decay=1e-4),
            alpha=0.5,
            k=6
        )
    )
    
    opt_state = optimizer.init(params)
    
    key = jax.random.PRNGKey(0)
    for step in range(20):
        grads = jax.tree_map(
            lambda p: 0.1 * jax.random.normal(key, p.shape),
            params
        )
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        lookahead_step = get_lookahead_step(opt_state)
        if step > 0 and lookahead_step % 6 == 0:
            print(f"Step {step}: Slow update performed (lookahead step {lookahead_step})")
    
    print("\nLookahead optimizer test completed successfully!")