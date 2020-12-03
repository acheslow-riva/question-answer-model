import torch
import torch_neuron
import numpy as np
import os

model_neuron = torch.neuron.trace(model, example_inputs=[pt_batch])
## Export to saved model
model_neuron.save("model.pt")

tokenizer_neuron = torch.neuron.trace(tokenizer)
tokenizer_neuron.save('tokenizer.pt')

# Now try with pipeline