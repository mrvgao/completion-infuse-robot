import torch
import torch.nn as nn

class CompletionTaskEmbeddingModel(nn.Module):
    def __init__(self, input_dim_s, d, V):
        super(CompletionTaskEmbeddingModel, self).__init__()
        self.map_s_to_scalar = nn.Linear(input_dim_s, 1)  # Maps sentence embedding 's' to scalar
        self.map_to_d = nn.Linear(2, d)  # Concatenates 'p' and scalar s, then maps to dimension d
        self.nonlinear = nn.ReLU()  # Nonlinear activation function
        self.map_to_V = nn.Linear(d, V)  # Maps to dimension V

    def forward(self, p, s):
        # p: [batch, time_steps]
        # s: [batch, time_steps, input_dim_s]

        batch_size, time_steps, input_dim_s = s.shape

        # Flatten batch and time_steps for efficient processing: [batch * time_steps, input_dim_s]
        s_flat = s.view(-1, input_dim_s)  # Flattening the sentence embeddings
        p_flat = p.view(-1, 1)  # Flattening the scalars

        # Map sentence embedding s to a scalar for each time step
        s_scalar = self.map_s_to_scalar(s_flat)  # Resulting shape: [batch * time_steps, 1]

        # Concatenate scalar s and scalar p: [batch * time_steps, 2]
        concatenated = torch.cat((p_flat, s_scalar), dim=1)

        # Map the concatenated result to dimension d
        mapped_to_d = self.map_to_d(concatenated)

        # Apply nonlinear function
        nonlinear_output = self.nonlinear(mapped_to_d)

        # Map to dimension V
        output = self.map_to_V(nonlinear_output)  # Shape: [batch * time_steps, V]

        # Reshape back to [batch, time_steps, V]
        output = output.view(batch_size, time_steps, -1)

        return output


if __name__ == '__main__':

    # Example usage:
    input_dim_s = 768  # Dimension of the sentence embedding 's'
    d = 128  # Dimension for the intermediate mapping
    V = 512  # Final output dimension

    model = CompletionTaskEmbeddingModel(input_dim_s, d, V)

    # Example input data
    batch_size = 32
    time_steps = 20
    p = torch.randn(batch_size, time_steps)  # Example scalar 'p' for each time step
    s = torch.randn(batch_size, time_steps, input_dim_s)  # Example sentence embeddings 's'

    # Forward pass
    output = model(p, s)
    print(output.shape)  # Output shape: [batch_size, time_steps, V]
