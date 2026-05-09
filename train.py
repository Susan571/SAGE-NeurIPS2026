"""
This file trains SAGE on AC Environment.
It sets up the training environment, initializes the SAGEPolicy, and runs the SAGE training loop.

Run this script directly to start training SAGE:

```
python -m sage.train --use-sage
```

To see the entire list of command line arguments you may pass, check args.py
"""

import numpy as np
import torch
import random
from torch.optim import Adam
from sage.policy import SAGEPolicy
from sage.config import parse_args
from sage.env_setup import get_env
from sage.training import sage_training_loop


def train_sage():
    """
    Main training function for SAGE algorithm.
    """
    args = parse_args()
    
    # Ensure SAGE is enabled
    if not args.use_sage:
        print("Warning: --use-sage is False. SAGE features will be limited.")
        print("Consider running with --use-sage flag for full SAGE functionality.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    (
        envs,
        initial_states,
        curr_states,
        success_record,
        ACMoves_hist,
        states_processed,
    ) = get_env(args)

    policy = SAGEPolicy(envs, args.nodes_counts).to(device)
    optimizer = Adam(policy.parameters(), lr=args.learning_rate, eps=args.epsilon)

    sage_training_loop(
        envs,
        args,
        device,
        optimizer,
        policy,
        curr_states,
        success_record,
        ACMoves_hist,
        states_processed,
        initial_states,
    )

    envs.close()


if __name__ == "__main__":
    train_sage()
