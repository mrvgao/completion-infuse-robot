import torch
import torch.nn as nn

class CompletionTaskEmbeddingModel(nn.Module):
    def __init__(self, task_str_emb_size, hidden_mapping_size, transformer_encoding_size):
        super(CompletionTaskEmbeddingModel, self).__init__()
        self.hidden_mapping_size = hidden_mapping_size
        self.map_completion_to_vector = nn.Linear(1, hidden_mapping_size)  # Maps scalar 'p' to dimension d
        # self.map_s_to_scalar = nn.Linear(input_dim_s, 1)  # Maps sentence embedding 's' to scalar
        self.merge_info = nn.Linear(hidden_mapping_size + task_str_emb_size, hidden_mapping_size)  # Concatenates 'p' and scalar s, then maps to dimension d
        self.nonlinear = nn.ReLU()  # Nonlinear activation function
        self.map_to_encoding_input_size = nn.Linear(hidden_mapping_size, transformer_encoding_size)  # Maps to dimension V

    def forward(self, completion_rate, task_str_emb):
        # Map sentence embedding s to a scalar
        completion_emb = self.map_completion_to_vector(completion_rate)

        # s_scalar = self.map_s_to_scalar(task_str_emb)
        # Concatenate scalar s and scalar p
        concatenated = torch.cat((completion_emb, task_str_emb), dim=1)
        # Map the concatenated result to dimension d
        mapped_to_d = self.merge_info(concatenated)
        # Apply nonlinear function
        nonlinear_output = self.nonlinear(mapped_to_d)
        # Map to dimension V
        output = self.map_to_encoding_input_size(nonlinear_output)
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
