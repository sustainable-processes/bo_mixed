# Multi-objective reaction optimisation for the Schotten-Baumann reaction (mixed variables)

This project aims to optimize a reaction with a mixed input domain and multiple objectives.

Code associated with the paper [Multi-objective Bayesian optimisation using q-noisy expected hypervolume improvement (q NEHVI) for the Schotten–Baumann reaction](https://pubs.rsc.org/en/content/articlehtml/2023/re/d3re00502j).

## Setup

1. Clone the repo

    ```bash
    git clone https://github.com/sustainable-processes/bo_mixed.git
    ```

2. Create a virtual environment and install the dependencies

    ```bash
    python3 - m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3. Run the code

    ```
    python main.py
    ```

You can see a full list of commands by running `python main.py --help`. Some examples:

```
# Use TSEMO as a strategy
python main.py --strategy MOBO

# Use TSEMO as a strategy
python main.py --strategy TSEMO

# Pass your own initialization data and don't run LHS initialization
python main.py --initialization-data-path data.csv --num-initial-experiments=0
```
