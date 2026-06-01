## C.3 GENBO SETTINGS

| Acronym    | Meaning                                                                 |
|------------|-------------------------------------------------------------------------|
| EI         | Expected Improvement                                                    |
| PI         | Probability of Improvement                                             |
| sEI        | Soft Expected Improvement, i.e., softplus$(y - \tau)$                   |
| SR         | Simple Regret (utility function)                                        |
| fKL        | Forward KL loss                                                         |
| bfKL       | Balanced forward KL loss                                                |
| rPL        | Robust preference loss                                                  |
| MF         | Mean-field categorical proposal model                                   |
| Tfm        | Transformer proposal model                                              |
| fr         | More frequent regularization (change in $\lambda_n$ schedule rate)      |
| r0p10      | Base regularization factor set to $\lambda_0 := 0.1$                    |
| exp        | Exponential regularizer, i.e., $R_n(\theta) := \lambda_n \exp\|\theta - \theta_0\|_2^2$ |
| np         | No (informative) prior, i.e., $p_0(\mathbf{x}) \propto 1$               |
| p          | Pre-trained prior, learned from initial (randomly initialized) data $\mathcal{D}_0$ |
| lg         | Importance weights                                                      |
| lr0p10     | Learning rate setting for training the generative model (e.g., 0.1 in this case) |
| pcmin0p50  | Minimum percentile for threshold $\tau_t$ annealing schedule (e.g., 50% in this case) |
| pcmax0p90  | Maximum percentile for threshold $\tau_t$ annealing schedule (e.g., 90% in this case) |

*Table 3: GenBO experiment settings acronyms*