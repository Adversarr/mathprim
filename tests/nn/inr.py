import math
import torch
import torch.nn as nn
import torch.optim as optim
import time

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.weight, -1, 1)
        if m.bias is not None:
            torch.nn.init.uniform_(m.bias, -1, 1)

def main():
    torch.manual_seed(3407)
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    input_dim = 2
    hidden_dim = 32
    output_dim = 1
    batch_size = 1 << 10
    max_iterations = 10000
    learning_rate = 1e-3
    beta1 = 0.9
    beta2 = 0.95
    weight_decay = 1e-2
    
    def mg(width):
        x = torch.linspace(-.5, .5, width)
        x, y = torch.meshgrid(x, x)
        return torch.stack([x, y], dim=-1).view(-1, 2)

    model = MLP(input_dim, hidden_dim, output_dim)
    model.apply(init_weights)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)

    start_time = time.time()
    x = mg(int(math.sqrt(batch_size)))

    for iteration in range(max_iterations):
        optimizer.zero_grad()

        # Generate random input data
        # x = torch.rand(batch_size, input_dim) * 4 - 2  # Uniform distribution between -2 and 2
        # Compute target output
        y_target = torch.sin(x[:, 0]) * torch.sin(x[:, 1])

        # Forward pass
        y_pred = model(x)
        # Compute loss
        loss = 0.5 * torch.nn.functional.mse_loss(y_pred.squeeze(), y_target)
        # Backward pass
        loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print(f"Iteration: {iteration}, Loss: {loss.item()}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Time: {elapsed_time * 1000:.2f}ms")
    print(f"{batch_size * max_iterations} samples in {elapsed_time * 1000:.2f}ms")
    print(f"=> {(batch_size * max_iterations) / elapsed_time:.2f} samples per second")
    width = 40
    # Create input tensor
    i_indices = torch.arange(width).float()
    j_indices = torch.arange(width).float()
    i_grid, j_grid = torch.meshgrid(i_indices, j_indices)
    in_tensor = torch.stack([i_grid * 0.5 / width - 0.5, j_grid * 0.5 / width - 0.5], dim=-1).view(-1, 2)

    # Forward pass
    out_tensor = model(in_tensor).view(width, width)

    # Ground truth calculation
    gt_tensor = torch.sin(i_grid * 0.5 / width - 0.5) * torch.sin(j_grid * 0.5 / width - 0.5)

    # Error calculation
    err_tensor = out_tensor - gt_tensor

    # RMSE calculation
    rmse = torch.norm(err_tensor) / width
    print(f"RMSE: {rmse.item()}")

if __name__ == "__main__":
    main()