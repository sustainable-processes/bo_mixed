# BO in a mixed, multiobjective domain

This project aims to optimize a reaction with a mixed input domain and multiple objectives.

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
```