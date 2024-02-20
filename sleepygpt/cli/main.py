
from torch import Tensor
import torch
import torch.nn as nn
import torch.optim as optim
import random

class SimpleChatModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(SimpleChatModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

# Simple tokenizer and detokenizer functions
def tokenize(text) -> Tensor:
    # This function converts text to a tensor of "token IDs"
    # For simplicity, let's just convert characters to their ASCII values
    return torch.tensor([ord(c) for c in text], dtype=torch.float)

def detokenize(token_ids) -> str:
    # Convert token IDs back to a string
    return ''.join([chr(int(c)) for c in token_ids])

# Function to generate a response (dummy for demonstration)
def generate_response(input_tensor):
    # For demonstration, let's just add a small value to each input token ID
    # In a real application, this function would involve model inference
    return input_tensor + random.randint(1, 3)

# Main chat function with targeted SGD
def chat() -> None:
    input_size = 10  # Assuming max 10 characters in input
    hidden_size = 8
    output_size = 10  # Output size matches input for simplicity
    model  = SimpleChatModel(input_size, hidden_size, output_size)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    print("Chatbot ready! Type 'quit' to exit.")
    
    while True:
        user_input: str = input("You: ")
        if user_input.lower() == 'quit':
            break

        # Tokenize and pad/truncate input to fixed size
        input_tensor: Tensor = tokenize(user_input)
        input_tensor = torch.nn.functional.pad(input_tensor, (0, input_size - input_tensor.size(0)), "constant", 0)
        input_tensor = input_tensor.view(1, -1)  # Add batch dimension
        
        # Generate a response (dummy logic here)
        response_tensor = generate_response(input_tensor)
        
        # "Learn" from this interaction by performing a targeted SGD update
        optimizer.zero_grad()
        predicted_response = model(input_tensor)
        loss = nn.MSELoss()(predicted_response, response_tensor)
        loss.backward()
        optimizer.step()

        # Detokenize response for display
        response_text: str = detokenize(predicted_response.detach().squeeze())
        print(f"Chatbot: {response_text}")

if __name__ == "__main__":
    chat()

