"""
The code for this file is adapted from
https://github.com/gchrupala/ursa/
and
https://github.com/gchrupala/correlating-neural-and-symbolic-representations-of-language/
"""

import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from cross_lingual_subnets.utils import to_tensor


def pearson(x, y, dim=0, eps=1e-8):
    "Returns Pearson's correlation coefficient."
    x, y = to_tensor(x), to_tensor(y)
    x1 = x - torch.mean(x, dim)
    x2 = y - torch.mean(y, dim)
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return w12 / (w1 * w2).clamp(min=eps)


def pearson_r_score(Y_true, Y_pred):
    r = pearson(Y_true, Y_pred, dim=0).mean()
    return r


class Regress:
    default_alphas = [10**n for n in range(-3, 2)]
    metrics = dict(
        mse=make_scorer(mean_squared_error, greater_is_better=False),
        r2=make_scorer(r2_score, greater_is_better=True),
        pearson=make_scorer(pearson_r_score, greater_is_better=True),
    )

    def __init__(self, cv=10, alphas=default_alphas):
        self.cv = cv
        self.grid = {"alpha": alphas}
        self._model = GridSearchCV(
            Ridge(),
            self.grid,
            scoring=self.metrics,
            cv=self.cv,
            return_train_score=False,
            refit=False,
        )

    def fit(self, X, Y):
        self._model.fit(X, Y)
        result = {name: {} for name in self.metrics.keys()}
        for name, scorer in self.metrics.items():
            mean = self._model.cv_results_["mean_test_{}".format(name)]
            std = self._model.cv_results_["std_test_{}".format(name)]
            best = mean.argmax()
            result[name]["mean"] = mean[best] * scorer._sign
            result[name]["std"] = std[best]
            result[name]["alpha"] = self.grid["alpha"][best]
        self._report = result

    def fit_report(self, X, Y):
        self.fit(X, Y)
        return self.report()

    def report(self):
        return self._report


def rsa(A, B):
    "Returns the correlation between the similarity matrices for A and B."
    M_A = 1 - cosine_matrix(A, A)
    M_B = 1 - cosine_matrix(B, B)
    return pearson(triu(M_A), triu(M_B), dim=0)


# TODO: remove after RSA has been fully implemented
# def rsa(enc1: torch.tensor, enc2: torch.tensor) -> float:
#     D_rep = 1 - cosine_matrix(data_enc1["test"], data_enc1["test"])
#     D_rep2 = 1 - cosine_matrix(data_enc2["test"], data_enc2["test"])
#     return cor(D_rep, D_rep2)


def rsa_regress(
    enc1_test: torch.tensor,
    enc1_ref: torch.tensor,
    enc2_test: torch.tensor,
    enc2_ref: torch.tensor,
    cv: int = 10,
) -> dict:
    D_rep = 1 - cosine_matrix(enc1_test, enc1_ref).detach().cpu().numpy()
    D_rep2 = 1 - cosine_matrix(enc2_test, enc2_ref).detach().cpu().numpy()
    r = Regress(cv=cv)
    rsa_reg = r.fit_report(D_rep, D_rep2)
    return rsa_reg


def cosine_matrix(U, V):
    "Returns the matrix of cosine similarity between each row of U and each row of V."
    U_norm = U / U.norm(2, dim=1, keepdim=True)
    V_norm = V / V.norm(2, dim=1, keepdim=True)
    return U_norm @ V_norm.t()


def triu(x):
    "Extracts upper triangular part of a matrix, excluding the diagonal."
    ones = torch.ones_like(x)
    return x[torch.triu(ones, diagonal=1) == 1]


def cor(a, b):
    return pearson(triu(a), triu(b), dim=0).item()


def rsa_report(data_enc1: dict, data_enc2: dict, cv=10):
    """Compute RSA and RSA_regress scores for two different encoders."""
    rsa_score = rsa(data_enc1["test"], data_enc2["test"])

    rsa_reg = rsa_regress(
        data_enc1["test"], data_enc1["ref"], data_enc2["test"], data_enc2["ref"], cv=cv
    )
    return dict(rsa=rsa_score, rsa_regress=rsa_reg)
