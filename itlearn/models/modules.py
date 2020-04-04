import torch


class ArgsModule(torch.nn.Module):
    def __init__(self, args):
        super(ArgsModule, self).__init__()
        args_dict = vars(args) # Save argparse arguments into member variables
        for key in args_dict:
            setattr(self, key, args_dict[key])

    def forward(self):
        return
