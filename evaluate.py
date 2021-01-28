import os
from typing import Optional

import pandas as pd
import typer
from sklearn.metrics import mean_squared_error

from generate_data import generate_data
from model import get_predictions


def evaluate(test_horizon: int,
             wandb_proj: Optional[str] = None) -> None:
    if wandb_proj is not None:
        import wandb
        github_sha = os.getenv('WANDB_SHA')
        wandb.init(project=wandb_proj)
        wandb.config.github_sha = github_sha
    data = generate_data()
    train_data = data.head(-test_horizon)
    valid_data = data.tail(test_horizon)
    predictions = get_predictions(train_data, test_horizon)
    mse = mean_squared_error(valid_data.y, predictions)
    if wandb_proj is not None:
        wandb.log({'mse': mse})
    print(f'MSE: {mse:.2f}')


if __name__ == '__main__':
    typer.run(evaluate)
