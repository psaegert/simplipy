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