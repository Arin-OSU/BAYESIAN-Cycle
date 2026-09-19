"""
TabMGP style forward sampling with frozen TabICLv2

Reference:
Ng et al. (2026), "TabMGP: Martingale Posterior with TabPFN"
https://arxiv.org/abs/2510.25154

This is an initial implementation of the forward-sampling procedure,
using TabICLv2 instead of TabPFN.

At each forward step:

    X_{i+1} ~ Empirical(X_1, ..., X_i)

    Y_{i+1} ~ TabICLv2(. | X_{i+1}, D_i)

Then the generated pair (X_{i+1}, Y_{i+1}) is appended to the context.

IMPORTANT:
TabICLv2's pretrained neural network weights remain frozen.
Calling model.fit() only supplies/updates the in-context dataset.

Current settings T=20 and L=5 are debug settings only.
They are not enough to claim convergence to a martingale posterior.
"""

import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange
from sklearn.linear_model import LogisticRegression
from tabicl import TabICLClassifier



# 0. EXPERIMENT CONFIGURATION

SEED = 42

N_OBS = 300
N_FEATURES = 3

# Number of forward-sampled observations per rollout
T_FORWARD = 20

# Number of independent rollouts
N_ROLLOUTS = 5

# Reduced for debugging/speed.
# Final experiments should also test standard TabICLv2 settings.
N_ESTIMATORS = 1


# 1. CREATE SYNTHETIC OBSERVED DATASET

rng = np.random.default_rng(SEED)

n = N_OBS
d = N_FEATURES

# Covariates:
# X_i ~ N(0, I)
X_obs = rng.normal(
    size=(n, d)
)

# True logistic-regression coefficients
beta_true = np.array([
    1.5,
    -1.0,
    0.7
])

# True intercept = 0
theta_true = np.concatenate([
    np.array([0.0]),
    beta_true
])

# Generate Bernoulli probabilities:
#
# p_i = sigmoid(X_i beta)

logits = X_obs @ beta_true

probs = 1.0 / (
    1.0 + np.exp(-logits)
)

# Generate binary outcomes
y_obs = rng.binomial(
    1,
    probs
)


print("=" * 60)
print("OBSERVED DATASET")
print("=" * 60)

print("X shape:", X_obs.shape)
print("y shape:", y_obs.shape)
print("Class balance:", y_obs.mean())

print("\nTrue theta:")
print(theta_true)



# 2. CREATE FROZEN PRETRAINED TabICLv2 MODEL

# DEBUG configuration.
#
# We deliberately use one estimator here for speed.
# The neural-network weights are NOT trained during this.

model = TabICLClassifier(
    n_estimators=N_ESTIMATORS,
    softmax_temperature=1.0,
    average_logits=False,
    kv_cache=False,
    random_state=SEED,
)


# 3. SAMPLE X FROM EMPIRICAL COVARIATE DISTRIBUTION

def sample_x_empirical(
    X_context,
    rng
):
    """
    Sample the next covariate from the empirical distribution
    of the covariates currently in the context.

    Mathematically:

        X_{i+1}
            ~ Empirical(X_1, ..., X_i)

    Previously forward-sampled X values are included in the
    empirical distribution.
    """

    idx = rng.integers(
        0,
        len(X_context)
    )

    x_new = X_context[
        idx:idx + 1
    ].copy()

    return x_new


# 4. SAMPLE Y FROM TabICLv2 PREDICTIVE DISTRIBUTION


def sample_y_tabicl(
    model,
    X_context,
    y_context,
    x_new,
    rng
):
    """
    Sample:

        Y_{i+1}
            ~ p_TabICL(
                y | x_{i+1}, D_i
              )

    model.fit() here does NOT train the neural-network weights.

    It updates the dataset supplied to the frozen TabICLv2
    model for in-context inference.
    """

    # Supply the current context
    model.fit(
        X_context,
        y_context
    )

    # Predict categorical distribution
    predictive_probs = model.predict_proba(
        x_new
    )[0]

    classes = model.classes_

    # Sample outcome from TabICLv2's predictive distribution
    y_new = rng.choice(
        classes,
        p=predictive_probs
    )

    return y_new, predictive_probs


# 5. RUN ONE FORWARD-SAMPLING ROLLOUT

def forward_rollout(
    model,
    X_obs,
    y_obs,
    T=T_FORWARD,
    seed=0,
    save_history=False
):
    """
    Generate one possible future dataset.

    Starting from:

        D_n = {(X_1,Y_1), ..., (X_n,Y_n)}

    repeat:

        1. Sample:

           X_{i+1}
               ~ Empirical(X_1, ..., X_i)

        2. Sample:

           Y_{i+1}
               ~ TabICLv2(. | X_{i+1}, D_i)

        3. Append:

           D_{i+1}
               = D_i U {(X_{i+1},Y_{i+1})}

    After T steps we obtain:

        D_N

    where:

        N = n + T
    """

    rollout_rng = np.random.default_rng(
        seed
    )

    X = np.asarray(
        X_obs
    ).copy()

    y = np.asarray(
        y_obs
    ).copy()

    history = []

    for t in range(T):

        # STEP A:
        # Sample next X from empirical covariate distribution
        

        x_new = sample_x_empirical(
            X_context=X,
            rng=rollout_rng
        )

        
        # STEP B:
        # Sample next Y from TabICLv2 predictive distribution
        

        y_new, predictive_probs = sample_y_tabicl(
            model=model,
            X_context=X,
            y_context=y,
            x_new=x_new,
            rng=rollout_rng
        )

        # STEP C:
        # Add simulated observation to context
    

        X = np.vstack([
            X,
            x_new
        ])

        y = np.append(
            y,
            y_new
        )

        # Save trajectory if requested
        if save_history:

            history.append({

                "step":
                    t + 1,

                "x_new":
                    x_new.copy(),

                "y_new":
                    y_new,

                "predictive_probs":
                    predictive_probs.copy()

            })

    return X, y, history


# 6. RUN ONE DEBUG ROLLOUT

print()
print("=" * 60)
print("SINGLE FORWARD ROLLOUT")
print("=" * 60)

X_full, y_full, history = forward_rollout(
    model=model,
    X_obs=X_obs,
    y_obs=y_obs,
    T=T_FORWARD,
    seed=123,
    save_history=True
)

print(
    "Original dataset size:",
    len(y_obs)
)

print(
    "Final dataset size:",
    len(y_full)
)

print(
    "Number forward sampled:",
    len(y_full) - len(y_obs)
)


print("\nFirst five generated observations:")

for item in history[:5]:

    print(
        f"step={item['step']}, "
        f"y={item['y_new']}, "
        f"probs="
        f"{np.round(item['predictive_probs'], 3)}"
    )


# 7. DEFINE FUNCTIONAL theta(F)

def compute_theta(
    X,
    y
):
    """
    Compute a scientific/statistical functional theta(F).

    Here theta(F) is defined as the coefficient vector of
    an unpenalized logistic regression fitted to the dataset.

    The returned vector is:

        theta = [
            intercept,
            beta_1,
            beta_2,
            beta_3
        ]

    In the TabMGP framework, after a forward rollout we compute:

        theta(F_N)

    Different rollout paths give different values of theta(F_N).
    """

    functional = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=2000
    )

    functional.fit(
        X,
        y
    )

    theta = np.concatenate([

        functional.intercept_.ravel(),

        functional.coef_.ravel()

    ])

    return theta


# 8. COMPARE ORIGINAL DATASET VS ONE FORWARD ROLLOUT

theta_original = compute_theta(
    X_obs,
    y_obs
)

theta_rollout = compute_theta(
    X_full,
    y_full
)


print()
print("=" * 60)
print("THETA COMPARISON")
print("=" * 60)

print("\nTrue theta:")
print(theta_true)

print("\ntheta(D_n):")
print(theta_original)

print("\ntheta(F_N) from one rollout:")
print(theta_rollout)

print("\ntheta(D_n) - true theta:")
print(
    theta_original
    -
    theta_true
)

print("\ntheta(F_N) - theta(D_n):")
print(
    theta_rollout
    -
    theta_original
)

# 9. GENERATE MULTIPLE INDEPENDENT FORWARD ROLLOUTS

def tabmgp_samples(
    model,
    X_obs,
    y_obs,
    T=T_FORWARD,
    L=N_ROLLOUTS,
    base_seed=1000
):
    """
    Generate L independent forward-sampling paths.

    For rollout l:

        D_n
            ->
        D_N^(l)
            ->
        theta(F_N^(l))

    Each rollout gives one finite-N forward-sampling sample:

        theta^(l)
            =
        theta(F_N^(l))

    If theta(F_N) stabilizes as N increases, these finite-N
    samples can be used as approximations to samples from:

        theta(F_infinity) | D_n

    which is the martingale-posterior object of interest.
    """

    theta_samples = []

    for l in trange(
        L,
        desc="Forward rollouts"
    ):

        X_rollout, y_rollout, _ = forward_rollout(

            model=model,

            X_obs=X_obs,

            y_obs=y_obs,

            T=T,

            seed=base_seed + l,

            save_history=False

        )

        theta_l = compute_theta(
            X_rollout,
            y_rollout
        )

        theta_samples.append(
            theta_l
        )

    return np.asarray(
        theta_samples
    )


# 10. RUN DEBUG POSTERIOR SAMPLING

print()
print("=" * 60)
print("MULTIPLE FORWARD ROLLOUTS")
print("=" * 60)

print(
    f"T = {T_FORWARD}"
)

print(
    f"L = {N_ROLLOUTS}"
)

print(
    "\nNOTE: These are DEBUG settings only."
)

print(
    "Do not interpret this as a converged posterior yet."
)


theta_samples = tabmgp_samples(

    model=model,

    X_obs=X_obs,

    y_obs=y_obs,

    T=T_FORWARD,

    L=N_ROLLOUTS

)


print(
    "\nPosterior sample matrix shape:"
)

print(
    theta_samples.shape
)


print(
    "\nFinite-N theta samples:"
)

print(
    theta_samples
)


# 11. COMPUTE SAMPLE SUMMARIES

posterior_mean = theta_samples.mean(
    axis=0
)

posterior_std = theta_samples.std(
    axis=0,
    ddof=1
)

lower = np.quantile(
    theta_samples,
    0.025,
    axis=0
)

upper = np.quantile(
    theta_samples,
    0.975,
    axis=0
)


print()
print("=" * 60)
print("FORWARD-SAMPLING SUMMARY")
print("=" * 60)


print(
    "\nTrue theta:"
)

print(
    theta_true
)


print(
    "\ntheta(D_n):"
)

print(
    theta_original
)


print(
    "\nMean theta(F_N):"
)

print(
    posterior_mean
)


print(
    "\nStandard deviation across rollouts:"
)

print(
    posterior_std
)


print(
    "\nEmpirical 2.5%-97.5% rollout quantiles"
)

print(
    "(DEBUG ONLY — L is too small for inference):"
)


for j in range(
    theta_samples.shape[1]
):

    print(

        f"theta[{j}]: "

        f"mean={posterior_mean[j]:.4f}, "

        f"SD={posterior_std[j]:.4f}, "

        f"quantiles="
        f"[{lower[j]:.4f}, {upper[j]:.4f}]"

    )


# 12. PLOT ONE COEFFICIENT

# theta[0] = intercept
# theta[1] = coefficient corresponding to beta_true[0]
# theta[2] = coefficient corresponding to beta_true[1]
# theta[3] = coefficient corresponding to beta_true[2]

coefficient_index = 1


plt.figure(
    figsize=(9, 6)
)


plt.hist(

    theta_samples[:, coefficient_index],

    bins=min(
        10,
        len(theta_samples)
    ),

    density=False,

    alpha=0.7
)


# Original observed-data estimate
plt.axvline(

    theta_original[coefficient_index],

    color="black",

    linestyle="--",

    linewidth=2,

    label=r"$\theta_1(D_n)$"

)


# Ground-truth coefficient
plt.axvline(

    theta_true[coefficient_index],

    color="black",

    linestyle=":",

    linewidth=2,

    label=r"True $\beta_1 = 1.5$"

)


plt.xlabel(
    r"$\theta_1$"
)

plt.ylabel(
    "Count"
)

plt.title(
    "TabICLv2 Forward-Sampling Samples\n"
    f"T={T_FORWARD}, L={N_ROLLOUTS}"
)

plt.legend()

plt.tight_layout()

plt.show()