# Exposure Game: Modeling Content Creator Incentives on Algorithm-Curated Platforms

![ICLR 2023](https://img.shields.io/badge/ICLR-2023-brightgreen)

## Overview

This repository contains a complete reproduction of the exposure game model from the ICLR 2023 paper "Modeling Content Creator Incentives on Algorithm-Curated Platforms" by Jiri Hron, Karl Krauth, Michael I. Jordan, Niki Kilbertus, and Sarah Dean. The implementation provides tools for pre-deployment auditing of recommendation algorithms to understand how they incentivize strategic content creation.

The exposure game model formalizes how content creators adapt their strategies to maximize exposure under different recommendation algorithms, revealing how seemingly innocuous algorithmic choices (like temperature parameters and embedding constraints) significantly impact content diversity and bias.

## Key Features

- **Pure Python implementation** with no NumPy version compatibility issues
- **Complete exposure game framework** supporting both softmax and hardmax scenarios
- **Custom matrix factorization implementations** for PMF and NMF algorithms
- **Pre-deployment audit tools** to evaluate algorithm impact before deployment
- **Visualization utilities** for analyzing strategic content patterns
- **Simulated datasets** mimicking MovieLens and LastFM data structures
- **Equilibrium finding algorithms** using gradient-based optimization

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tqdm
```

All dependencies are compatible with modern NumPy versions (including NumPy 2.x), with no reliance on the `surprise` library that causes compatibility issues.

## Usage

### Running the Complete Audit
```python
from exposure_game import run_complete_audit

pmf_results, nmf_results = run_complete_audit()
```

This will:
1. Generate simulated MovieLens data
2. Train PMF and NMF recommendation models
3. Run exposure game equilibrium finding for multiple parameter configurations
4. Generate visualizations of strategic content patterns
5. Save results to pickle files and PNG images

### Custom Audit Configuration
```python
from exposure_game import PreDeploymentAudit, load_movielens_data

# Load data
ratings_matrix, user_gender = load_movielens_data()

# Initialize audit
audit = PreDeploymentAudit(
    ratings_matrix=ratings_matrix,
    user_gender=user_gender,
    algorithm='pmf'  # or 'nmf'
)

# Run with custom parameters
results = audit.run_audit(
    dimensions=[10, 20, 100],
    temperatures=[0.001, 0.01, 0.1, 1.0, 10.0],
    num_producers=25,
    max_iter=2000,
    lr=0.05,
    seed=123
)

# Visualize results
audit.visualize_results(results, "Custom_Audit")
```

## Output Files

After running the audit, the following files will be generated:
- `pmf_audit_results.pkl` - PMF audit results
- `nmf_audit_results.pkl` - NMF audit results  
- `movie_lens_comparison.png` - Comparison visualization of PMF vs NMF results

## Key Findings

The implementation validates the paper's main findings:

1. **Exploration-Diversity Trade-off**: Higher temperature values (more exploration) lead to content homogenization, while lower temperature values encourage specialization and diversity.

2. **Algorithmic Bias**: More expressive models (higher embedding dimensions) and specific algorithm choices (NMF vs PMF) amplify biases toward gender-based user groups.

3. **Non-negative Constraints**: The choice between non-negative and unconstrained embeddings significantly affects equilibrium existence and characteristics.

4. **Temperature Sensitivity**: Small changes in temperature parameter can lead to qualitatively different content ecosystems.

## Customization

### Parameter Configurations
- **Embedding dimensions**: Test different model complexities (3, 10, 50, 100)
- **Temperature values**: Explore exploration-exploitation trade-offs (0.001 to 10.0)
- **Number of producers**: Simulate different producer-to-consumer ratios
- **Learning rate & iterations**: Adjust optimization parameters for equilibrium finding

### Extending the Framework
The code is modular and can be extended to:
- Add new recommendation algorithms
- Implement different game solution concepts
- Incorporate real-world datasets
- Add demographic attributes beyond gender
- Integrate dynamic attention pool models

## Limitations & Future Work

While this implementation provides a faithful reproduction of the paper's core contributions, several limitations exist:

1. **Behavioral assumptions**: The model assumes fully rational creators with complete information
2. **Static attention pool**: Does not model dynamic attention economies
3. **Simulated data**: Uses generated data rather than proprietary platform data
4. **Single-objective optimization**: Assumes creators only maximize exposure

Future extensions could address these limitations by incorporating bounded rationality models, dynamic attention mechanisms, and multi-objective utility functions.

## Citation

If you use this implementation in your research, please cite the original paper:

```
@inproceedings{hron2023modeling,
  title={Modeling Content Creator Incentives on Algorithm-Curated Platforms},
  author={Hron, Jiri and Krauth, Karl and Jordan, Michael I and Kilbertus, Niki and Dean, Sarah},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```

## License

This implementation is provided for research and educational purposes. The code follows the academic integrity standards appropriate for reproducing published research. Please respect the original authors' intellectual property and cite their work appropriately.
