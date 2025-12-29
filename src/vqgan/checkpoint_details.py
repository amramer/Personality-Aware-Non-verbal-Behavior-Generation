import torch

def load_checkpoint(checkpoint_path):
    """
    Load the checkpoint from the given path.

    Parameters:
    checkpoint_path (str): Path to the checkpoint file.

    Returns:
    dict: Loaded checkpoint.
    """
    checkpoint = torch.load(checkpoint_path)
    return checkpoint

def main():
    checkpoint_path = '/home/amr/Clusters@DFKI/Thesis/pipeline/vqgan/experiments/experiment_2023-08-30_17-19-35/models/l2_32_smoothSS_er2er_best.pth'
    checkpoint = load_checkpoint(checkpoint_path)
    
    # Print the details
    print("Last epoch: ", checkpoint['epoch'])
    # print("Last optimizer state: ", checkpoint['optimizer'])
    # print("Model state dict: ", checkpoint['state_dict'])

if __name__ == "__main__":
    main()
