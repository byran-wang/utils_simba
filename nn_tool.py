def get_learnable_parameters(nn_class):
    for name, param in nn_class.named_parameters():
        print(f"Name: {name}")
        print(f" - Shape: {param.shape}")
        print(f" - Requires Grad: {param.requires_grad}")
        print(f" - Number of Elements: {param.numel()}")
        print()