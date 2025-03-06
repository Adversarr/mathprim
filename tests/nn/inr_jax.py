import jax
import jax.numpy as jnp
import optax
import time
from flax import linen as nn
from flax.core.frozen_dict import freeze

class MLP(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(self.output_dim)(x)
        return x

def init_weights(key, shape):
    return jax.random.uniform(key, shape, minval=-1, maxval=1)

def main():
    key = jax.random.PRNGKey(3407)
    input_dim = 2
    hidden_dim = 64
    output_dim = 1
    batch_size = 1 << 18
    max_iterations = 1000
    learning_rate = 1e-3
    beta1 = 0.9
    beta2 = 0.95
    weight_decay = 1e-2

    model = MLP(input_dim, hidden_dim, output_dim)
    x_init = jnp.zeros((batch_size, input_dim))
    params = model.init(key, x_init)

    optimizer = optax.adamw(learning_rate, b1=beta1, b2=beta2, weight_decay=weight_decay)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, x_batch, y_target):
        def loss_fn(params):
            y_pred = model.apply(params, x_batch)
            return 0.5 * jnp.mean((y_pred.squeeze() - y_target) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    start_time = time.time()
    x_batch = jax.random.uniform(key, (batch_size, input_dim), minval=-2, maxval=2)
    for iteration in range(max_iterations):
        y_target = jnp.sin(x_batch[:, 0]) * jnp.sin(x_batch[:, 1])
        params, opt_state, loss = train_step(params, opt_state, x_batch, y_target)

        if iteration % 100 == 0:
            print(f"Iteration: {iteration}, Loss: {loss}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time: {elapsed_time * 1000:.2f}ms")
    print(f"{batch_size * max_iterations} samples in {elapsed_time * 1000:.2f}ms")
    print(f"=> {(batch_size * max_iterations) / elapsed_time:.2e} samples per second")
    
    # Test inference speed
    x_batch = jax.random.uniform(key, (batch_size, input_dim), minval=-2, maxval=2)
    start_time = time.time()
    for _ in range(max_iterations):
      y_pred = model.apply(params, x_batch)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Inference time: {elapsed_time * 1000:.2f}ms")
    print(f"{batch_size * max_iterations} samples in {elapsed_time * 1000:.2f}ms")
    print(f"=> {(batch_size * max_iterations) / elapsed_time:.2e} samples per second")

if __name__ == "__main__":
    main()