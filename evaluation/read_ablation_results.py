import json
from pathlib import Path
import re
from typing import Mapping, Union
import click

import pandas as pd
from tqdm import tqdm
from main.evaluate_separation import evaluate_data


@click.command()
@click.argument("sep_dir")
@click.argument("output_file")
def read_ablation_results(sep_dir: Union[str, Path], output_file: Union[str, Path]):
    sep_dir = Path(sep_dir)
    output_file = Path(output_file)

    hparams_files = list(sep_dir.glob("*.json"))
    records = []
    for hparam_path in tqdm(hparams_files):
        with open(hparam_path, "r") as f:
            hparams = json.load(f)
            assert isinstance(hparams, Mapping), type(hparams)

            match = re.fullmatch(
                "experiment-(?P<exp_num>[0-9]*)-hparams.json", hparam_path.name
            )
            assert match is not None
            experiment_number = match.groupdict()["exp_num"]
            experiment_dir = sep_dir / f"experiment-{experiment_number}"
            # print(experiment_dir)
            hparams_results = evaluate_data(experiment_dir)
            records.append({**hparams, **hparams_results})

    df = pd.DataFrame.from_records(records)
    df.to_csv(output_file)


if __name__ == "__main__":
    read_ablation_results()
