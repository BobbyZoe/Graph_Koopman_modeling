# Graph_Koopman_modeling
We propose a Koopman-inspired dynamic graph (KDG) framework for SOZ localization from iEEG, which models the temporal evolution of FC graphs using an operator-based dynamical representation. 

## Repository Structure

- **`hup_koopman_baselines/`** contains implementations of the baseline methods used for comparison in the manuscript. These methods were reproduced according to the methodological descriptions provided in the corresponding publications.
- **`compute_operator.ipynb`** implements the core KDG pipeline. It constructs dynamic functional connectivity (FC) graphs from iEEG recordings, models the temporal evolution of these graphs using a Koopman-inspired operator, and applies singular value decomposition (SVD) to the learned operators to extract their singular vectors.
- **`graph_features_calculation.py`** derives four graph-theoretic measures from the singular-vector representations: node strength, degree standard deviation, eigenvector centrality, and global efficiency.
- **`metric_loso_soz_classification.py`** evaluates the ability of these graph-theoretic features to localize the seizure onset zone (SOZ) using leave-one-subject-out (LOSO) cross-validation and records the corresponding localization performance.
