# Collecting Rules

```sh
simplipy find-rules -e "path/to/my_config.yaml" -c "path/to/create_my_config.yaml" -o "path/to/my_rules.json" -v --reset-rules
```

- `-e` is the path to the engine configuration file to use as a backend
- `-c` is the path to the configuration file containing parameters for finding rules
- `-o` is the output path for the collected rules
- `-v` enables verbose output
- `--reset-rules` will start with an empty rule set, otherwise it will append to the existing rules loaded with the engine

A configuration file for collecting rules could look like this:

```yaml
# This includes special symbols for intermediate simplification steps
extra_internal_terms: [
  '<constant>',
  '0',
  '1',
  '(-1)',
  'np.pi',
  'np.e',
  'float("inf")',
  'float("-inf")',
  'float("nan")'
]

# Number of dummy variables used to create equations
# null means the number of dummy variables is determined automatically based on max source pattern length
dummy_variables: null  

# Maximum number of tokens in the source equation
max_source_pattern_length: 3

# Maximum number of tokens in the target equation
max_target_pattern_length: 2

# Number of data points to compute the image of the equation
n_samples: 1024

# Number of combinations of constants to sample for each equation
# This prevents false positives due to unlucky constant combinations
constants_fit_challenges: 16

# Number of retries for each challenge to find a valid constant combination
# This accounts for optimization problems that may not converge
constants_fit_retries: 16
```

# Post-processing Rules

Two commands refine an existing rule set loaded from an engine and write the
result to a JSON file (they do not modify the installed asset in place):

```sh
# Remove explicit rules that are already subsumed by wildcard-pattern rules
simplipy prune-rules -e "dev_7-3" -o "path/to/pruned_rules.json" -v

# Replace <constant> placeholders with concrete numeric values in all-numeric rules
simplipy resolve-rules -e "dev_7-3" -o "path/to/resolved_rules.json" -v
```

- `-e` is the engine name (e.g. `dev_7-3`) or a path to an engine configuration file
- `-o` is the output path for the post-processed rules
- `-v` enables verbose progress output

# Managing Engine Assets

List the engines (and test-data assets) available on Hugging Face and which are
already installed locally:

```sh
simplipy list --type engine
# --- Available Assets ---
# - dev_7-3         [installed]  Development engine 7-3 for mathematical expression simplification.
# - dev_7-2                      Development engine 7-2 for mathematical expression simplification.

simplipy list --installed        # only assets already downloaded
```

To install or remove assets, use the Python API (the `simplipy` asset-manager functions
`install` / `uninstall` / `get_path`), which is also what the engine loader uses under the hood:

```python
import simplipy as sp

sp.install("dev_7-3")     # download an asset from Hugging Face
sp.uninstall("dev_7-3")   # remove a locally installed asset
sp.get_path("dev_7-3", install=True)  # resolve a local path, installing if needed
```

`sp.SimpliPyEngine.load("dev_7-3", install=True)` installs the engine
on demand as part of loading.