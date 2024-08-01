# Relational Cognitive Dissonances
How do ANNs resolve relational cognitive dissonances where two adaptation pathways are possible? This repository reconstructs the results and figures of the paper "Is it me, or is A larger than B: Relational cognitive dissonances with two adaptation pathways modeled by artificial neural networks"

## Figure reconstruction based on existing results
Run the functions in the main of plots.py.
The Figures of the simplified models can be reconstructed by the code files in the folder "/linear".

## Generating results

### Figure 2 left

python main.py rules="(0,0,2,0,0)" just_training=True measure_optimization=True total_alphas=[0.5] nets_per_total_alpha=100

### Figures 2 right and 6

python main.py rules="(0,0,2,0,0)" nets_per_total_alpha=100

### Figures 4 and 5

python main.py rules="(0,0,2,0,0)" total_alphas=[0.2,0.8] measure_optimization=True nets_per_total_alpha=50

### Figure S5

python main.py rules="(0,0,2,0,0)" only_for_RT=True measure_optimization=True nets_per_total_alpha=50

### Figure 8

python main.py rules="(0,0,2,0,0)" gammas=1_np.linspace(0.25,4,9) nets_per_total_alpha=50

### Figure 9

python main.py rules="(0,0,2,0,0)" eta_ratios=np.logspace(-2,2,9,base=2) nets_per_total_alpha=50

### Figure 12 right

python main.py rules="(0,0,2,0,0)" betas=np.linspace(0.1,1,9) with_neg_betas=True nets_per_total_alpha=50
