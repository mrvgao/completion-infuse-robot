import torch
import torch.nn as nn

class CompletionTaskEmbeddingModel(nn.Module):
    def __init__(self, input_dim_s, hidden_mapping_size, V):
        super(CompletionTaskEmbeddingModel, self).__init__()
        self.hidden_mapping_size = hidden_mapping_size
        self.map_s_to_scalar = nn.Linear(input_dim_s, 1)  # Maps sentence embedding 's' to scalar
        self.map_to_d = nn.Linear(2, hidden_mapping_size)  # Concatenates 'p' and scalar s, then maps to dimension d
        self.nonlinear = nn.ReLU()  # Nonlinear activation function
        self.map_to_V = nn.Linear(hidden_mapping_size, V)  # Maps to dimension V

    def forward(self, completion_rate, task_str_emb):
        # Map sentence embedding s to a scalar
        s_scalar = self.map_s_to_scalar(task_str_emb)
        # Concatenate scalar s and scalar p
        concatenated = torch.cat((completion_rate, s_scalar), dim=1)
        # Map the concatenated result to dimension d
        mapped_to_d = self.map_to_d(concatenated)
        # Apply nonlinear function
        nonlinear_output = self.nonlinear(mapped_to_d)
        # Map to dimension V
        output = self.map_to_V(nonlinear_output)
        return output


if __name__ == '__main__':

    # Example usage:
    input_dim_s = 768  # Dimension of the sentence embedding 's'
    d = 128  # Dimension for the intermediate mapping
    V = 512  # Final output dimension

    model = CompletionTaskEmbeddingModel(input_dim_s, d, V)

    # Example input data
    batch_size = 16
    time_steps = 10
    p = torch.randn(batch_size, 1)  # Example scalar 'p' for each time step
    s = torch.randn(batch_size, input_dim_s)  # Example sentence embeddings 's'

    print(p.size())
    print(s.size())
    # Forward pass
    output = model(p, s)
    print(output.shape)  # Output shape: [batch_size, time_steps, V]
