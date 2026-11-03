"""Model wrappers for the traps case study.

Every model answers the same two questions on a (train, test) pair of feature frames:
``point`` (one prediction per row) and ``dist`` (a predictive distribution per row, as 99
quantiles). The distribution comes from wherever the model can give one:

* gp: its Gaussian posterior (Matern 5/2 with one length scale per input, i.e. ARD).
* gbdt: quantile gradient boosting at 7 levels, interpolated in normal-score space.
* exaone-tabular: its native 999-quantile regression head ("native").
* limix-16m: its native bucket histogram when the checkpoint has one ("native").
* tabfm, mitra: their public APIs return one number per row, so the spread is the
  yield engine's bootstrap-of-the-context plus out-of-fold noise ("bootstrap").
If a native read-out fails, the bootstrap is used and ``dist_source`` says so.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

LEVELS = np.round(np.arange(1, 100) / 100, 2)  # 0.01 .. 0.99
Z = norm.ppf(LEVELS)


@dataclass
class Dist:
    mean: np.ndarray  # (n,)
    q: np.ndarray  # (n, 99) quantiles at LEVELS
    source: str = ""

    @classmethod
    def gaussian(cls, mean, sd, source="gaussian"):
        mean = np.asarray(mean, float)
        sd = np.maximum(np.asarray(sd, float), 1e-6)
        return cls(mean, mean[:, None] + sd[:, None] * Z[None, :], source)

    @classmethod
    def from_quantiles(cls, levels, qs, mean=None, source="quantiles"):
        """Interpolate given quantile levels onto LEVELS, linear in normal score."""
        levels = np.asarray(levels, float)
        qs = np.sort(np.asarray(qs, float), axis=1)
        zl = norm.ppf(levels)
        out = np.empty((qs.shape[0], len(LEVELS)))
        for i in range(qs.shape[0]):
            out[i] = np.interp(Z, zl, qs[i])
            lo_slope = (qs[i, 1] - qs[i, 0]) / (zl[1] - zl[0])
            hi_slope = (qs[i, -1] - qs[i, -2]) / (zl[-1] - zl[-2])
            out[i] = np.where(Z < zl[0], qs[i, 0] + (Z - zl[0]) * lo_slope, out[i])
            out[i] = np.where(Z > zl[-1], qs[i, -1] + (Z - zl[-1]) * hi_slope, out[i])
        m = out.mean(axis=1) if mean is None else np.asarray(mean, float)
        return cls(m, out, source)

    def interval(self, coverage):
        a = (1 - coverage) / 2
        lo = np.interp(a, LEVELS, np.arange(len(LEVELS)))
        hi = np.interp(1 - a, LEVELS, np.arange(len(LEVELS)))
        col = lambda k: self.q[:, int(np.floor(k))] + (k - np.floor(k)) * (
            self.q[:, min(int(np.floor(k)) + 1, len(LEVELS) - 1)] - self.q[:, int(np.floor(k))])
        return col(lo), col(hi)

    @property
    def sd(self):
        lo, hi = self.interval(0.8)
        return (hi - lo) / (2 * 1.2816)

    def take(self, idx):
        return Dist(self.mean[idx], self.q[idx], self.source)

    def map(self, f):
        """Push the distribution through a monotone increasing transform f."""
        q = f(self.q)
        return Dist(q.mean(axis=1), q, self.source)


def one_hot(train: pd.DataFrame, test: pd.DataFrame, cats: list[str]):
    """One-hot categoricals, standardise numerics (train stats), fill NaN with the train mean."""
    both = pd.concat([train, test], keys=["tr", "te"])
    num = [c for c in both.columns if c not in cats]
    X = pd.get_dummies(both, columns=cats, dtype=float)
    mu = X.loc["tr", num].mean()
    sd = X.loc["tr", num].std().replace(0, 1).fillna(1)
    X[num] = ((X[num] - mu) / sd).fillna(0.0)
    return X.loc["tr"].to_numpy(float), X.loc["te"].to_numpy(float), list(X.columns)


# ----------------------------------------------------------------------------------- GP
class GPModel:
    """GP with ARD. Hyperparameters are optimised once per feature set and target kind
    (on the first training set it sees) and then reused, as BO loops usually do; each
    later fit only re-conditions on the data. A full ARD fit on 700 rows x 40 inputs
    takes minutes on a CPU, so re-optimising on every fold would dominate the study."""

    name = "gp"
    dist_source = "gaussian"

    def __init__(self, n_restarts: int = 0, seed: int = 0, maxiter: int = 150):
        self.n_restarts = n_restarts
        self.seed = seed
        self.maxiter = maxiter
        self.last_lengthscales = None
        self._kernels: dict = {}

    def _optimizer(self, obj_func, initial_theta, bounds):
        from scipy.optimize import minimize

        res = minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds,
                       options={"maxiter": self.maxiter})
        return res.x, res.fun

    def _fit(self, Xtr, cats, ytr, Xte):
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

        A, B, cols = one_hot(Xtr, Xte, cats)
        y = np.asarray(ytr, float)
        kind = "binary" if set(np.unique(y)) <= {0.0, 1.0} else ("logit" if y.min() < 0 or y.max() > 1 else "unit")
        key = (tuple(cols), kind)
        if key in self._kernels:
            gp = GaussianProcessRegressor(self._kernels[key], normalize_y=True, optimizer=None)
        else:
            k = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(np.full(A.shape[1], 3.0), (0.05, 1e3), nu=2.5) \
                + WhiteKernel(0.05, (1e-5, 1.0))
            gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=self.n_restarts,
                                          random_state=self.seed, optimizer=self._optimizer)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(A, y)
        self._kernels.setdefault(key, gp.kernel_)
        self.last_lengthscales = dict(zip(cols, np.atleast_1d(gp.kernel_.k1.k2.length_scale).tolist()))
        return gp, B

    def point(self, Xtr, cats, ytr, Xte):
        gp, B = self._fit(Xtr, cats, ytr, Xte)
        return gp.predict(B)

    def dist(self, Xtr, cats, ytr, Xte):
        gp, B = self._fit(Xtr, cats, ytr, Xte)
        m, s = gp.predict(B, return_std=True)
        return Dist.gaussian(m, s, "gaussian")


# --------------------------------------------------------------------------------- GBDT
class GBDTModel:
    """Histogram gradient boosting; a small grouped inner search picks the settings."""

    name = "gbdt"
    dist_source = "quantile-boosting"
    GRID = [dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=10),
            dict(learning_rate=0.05, max_iter=600, max_leaf_nodes=7, min_samples_leaf=15),
            dict(learning_rate=0.1, max_iter=200, max_leaf_nodes=31, min_samples_leaf=5),
            dict(learning_rate=0.03, max_iter=800, max_leaf_nodes=15, min_samples_leaf=20)]
    QLEVELS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

    def __init__(self, seed: int = 0, tune: bool = True):
        self.seed = seed
        self.tune = tune

    @staticmethod
    def _frame(Xtr, Xte, cats):
        both = pd.concat([Xtr, Xte], keys=["tr", "te"]).copy()
        for c in cats:
            both[c] = both[c].astype("category")
        return both.loc["tr"], both.loc["te"]

    def _model(self, params, loss="squared_error", quantile=None):
        from sklearn.ensemble import HistGradientBoostingRegressor

        kw = dict(params, l2_regularization=1.0, categorical_features="from_dtype", random_state=self.seed,
                  early_stopping=False)
        if loss == "quantile":
            kw.update(loss="quantile", quantile=quantile)
        return HistGradientBoostingRegressor(**kw)

    def _params(self, A, ytr):
        if not self.tune:
            return self.GRID[0]
        rng = np.random.default_rng(self.seed)
        folds = np.array_split(rng.permutation(len(ytr)), 3)
        best, best_err = None, np.inf
        for p in self.GRID:
            err = 0.0
            for f in folds:
                m = np.ones(len(ytr), bool)
                m[f] = False
                mdl = self._model(p).fit(A[m], ytr[m])
                err += float(np.mean((mdl.predict(A.iloc[f]) - ytr[f]) ** 2))
            if err < best_err:
                best, best_err = p, err
        return best

    def point(self, Xtr, cats, ytr, Xte):
        A, B = self._frame(Xtr, Xte, cats)
        p = self._params(A, np.asarray(ytr, float))
        return self._model(p).fit(A, ytr).predict(B)

    def dist(self, Xtr, cats, ytr, Xte):
        A, B = self._frame(Xtr, Xte, cats)
        ytr = np.asarray(ytr, float)
        p = self._params(A, ytr)
        mean = self._model(p).fit(A, ytr).predict(B)
        qs = np.column_stack([self._model(p, "quantile", a).fit(A, ytr).predict(B) for a in self.QLEVELS])
        return Dist.from_quantiles(self.QLEVELS, qs, mean, "quantile-boosting")


# ------------------------------------------------------------------- in-context models
def schema_for(Xtr: pd.DataFrame, Xte: pd.DataFrame, cats: list[str]):
    from ..schema import Kind, Role, Schema, Variable

    both = pd.concat([Xtr, Xte])
    vs = []
    for c in Xtr.columns:
        if c in cats:
            vs.append(Variable(c, c, Kind.CATEGORICAL, Role.SET_BEFORE, categories=sorted(both[c].astype(str).unique())))
        else:
            lo, hi = float(np.nanmin(both[c])), float(np.nanmax(both[c]))
            vs.append(Variable(c, c, Kind.CONTINUOUS, Role.SET_BEFORE, lower=lo, upper=hi if hi > lo else lo + 1))
    vs.append(Variable("__y__", "y", Kind.CONTINUOUS, Role.TARGET, lower=0, upper=1))
    return Schema(vs, date_column=None, reference_column=None)


class InContextModel:
    """Wraps a yieldengine in-context adapter (TabFM, EXAONE, LimiX, Mitra)."""

    def __init__(self, name: str, n_bootstrap: int = 8, n_folds: int = 5, native: bool = True, **load_kw):
        from ..models import REGISTRY

        self.name = name
        self.cls = REGISTRY[name]
        self.n_bootstrap = n_bootstrap
        self.n_folds = n_folds
        self.native = native
        self.load_kw = load_kw
        self._runners: dict = {}
        self.dist_source = "bootstrap"
        self.native_error: str | None = None

    def adapter(self, Xtr, Xte, cats, n_bootstrap=None, n_folds=None):
        schema = schema_for(Xtr, Xte, cats)
        nb = self.n_bootstrap if n_bootstrap is None else n_bootstrap
        nf = self.n_folds if n_folds is None else n_folds
        # LimiX bakes the categorical column indices into its predictor; the others don't
        key = tuple(i for i, c in enumerate(Xtr.columns) if c in cats) if self.name == "limix-16m" else ()
        if key not in self._runners:
            first = self.cls(schema, n_bootstrap=nb, n_folds=nf, **self.load_kw)
            self._runners[key] = first._runner
            return first
        return self.cls(schema, n_bootstrap=nb, n_folds=nf, runner=self._runners[key])

    @staticmethod
    def _with_y(X, y):
        return X.assign(__y__=np.asarray(y, float))

    def point(self, Xtr, cats, ytr, Xte):
        a = self.adapter(Xtr, Xte, cats, n_folds=0)
        return a.point_predict(self._with_y(Xtr, ytr), Xte)

    def dist(self, Xtr, cats, ytr, Xte, native: bool | None = None):
        use_native = self.native if native is None else native
        if use_native and self.name in ("exaone-tabular", "limix-16m"):
            try:
                a = self.adapter(Xtr, Xte, cats, n_folds=0)
                Xc, _ = a.encoder.coded(Xtr)
                Xq, _ = a.encoder.coded(Xte)
                yc = np.asarray(ytr, float)
                f = exaone_quantiles if self.name == "exaone-tabular" else limix_quantiles
                d = f(a, Xc, yc, Xq)
                self.dist_source = "native"
                return d
            except Exception as e:  # fall back to the engine's bootstrap spread
                self.native_error = f"{type(e).__name__}: {e}"[:400]
                warnings.warn(f"{self.name} native distribution failed, using bootstrap: {self.native_error}")
        a = self.adapter(Xtr, Xte, cats)
        a.fit(self._with_y(Xtr, ytr))
        p = a.predict(Xte)
        self.dist_source = "bootstrap"
        return Dist.gaussian(p.mean, p.sd, "bootstrap")


def exaone_quantiles(adapter, Xc, yc, Xq) -> Dist:
    """EXAONE-Tabular's own 999-quantile head, pooled over ensemble members the way its
    trimmed read-out does (quantile functions averaged after trimming 20 % per tail)."""
    import torch
    import exaonetabular.regressor as R

    reg = adapter._runner
    reg.fit(Xc, yc)
    state = reg._state()
    query = R.TabularPreprocessor._validate_features(np.asarray(Xq, float), allow_zero_rows=True)
    sx = torch.as_tensor(state["support_x"], dtype=torch.float64)
    sy = torch.as_tensor(state["support_y"], dtype=torch.float32)
    qs = torch.cat([reg._member_quantiles(sx, sy, query, n_svd=n_svd, seed=seed,
                                          preprocessor=state["preprocessors"][seed])
                    for n_svd, seed in state["passes"]], dim=0)
    qs = (qs * state["scale"] + state["center"]).sort(dim=-1).values  # (members, n, 999)
    cut = int(0.2 * qs.shape[0])
    if cut:
        qs = qs.sort(dim=0).values[cut:-cut]
    qf = qs.mean(dim=0).cpu().numpy()  # (n, 999)
    levels = np.arange(1, qf.shape[1] + 1) / (qf.shape[1] + 1)
    idx = np.clip(np.searchsorted(levels, LEVELS), 0, qf.shape[1] - 1)
    q99 = qf[:, idx]
    mean = np.asarray(reg.predict(np.asarray(Xq, float)), float)
    return Dist(mean, np.sort(q99, axis=1), "native")


def limix_quantiles(adapter, Xc, yc, Xq) -> Dist:
    """LimiX's bucket histogram, captured from inside its regression read-out. The
    adapter feeds LimiX a standardised target, so the buckets are mapped back."""
    import torch

    pred = adapter._runner
    inner = getattr(pred, "get_reg_pred_result", None)
    if inner is None:
        raise RuntimeError("this LimiX build has no bucket read-out (get_reg_pred_result)")
    mod = __import__(type(pred).__module__, fromlist=["translate_probs_across_borders"])
    captured = {}

    def capture(inputs, y_train):
        tmp = [ele[0].squeeze(0) for ele in inputs]
        borders = pred.model._reg_borders
        probs = []
        for lg in tmp:
            if pred.softmax_temperature != 1:
                lg = lg.float() / pred.softmax_temperature
            probs.append(mod.translate_probs_across_borders(lg.squeeze(0), frm=borders.to(lg.device),
                                                            to=borders.to(lg.device)))
        captured["p"] = torch.stack(probs, 0).mean(0).float().cpu().numpy()
        captured["b"] = borders.float().cpu().numpy()
        return inner(inputs, y_train)

    yc = np.asarray(yc, float)
    mu, sd = float(yc.mean()), float(yc.std()) or 1.0
    pred.get_reg_pred_result = capture
    try:
        out = pred.predict(Xc, (yc - mu) / sd, Xq, task_type="Regression")
    finally:
        del pred.get_reg_pred_result
    if "p" not in captured:
        raise RuntimeError(f"LimiX used the '{getattr(pred, 'regression_decoder_type', '?')}' head, no histogram")
    p, b = captured["p"], captured["b"]
    p = p.reshape(-1, p.shape[-1])
    p = p / p.sum(axis=1, keepdims=True)
    cdf = np.concatenate([np.zeros((p.shape[0], 1)), np.cumsum(p, axis=1)], axis=1)
    q = np.stack([np.interp(LEVELS, cdf[i], b) for i in range(p.shape[0])])
    return Dist(np.asarray(out, float).reshape(-1) * sd + mu, q * sd + mu, "native")


class KNNInContext:
    """Stand-in in-context model for testing the study code without GPU weights."""

    name = "knn"
    dist_source = "bootstrap"

    def point(self, Xtr, cats, ytr, Xte):
        from sklearn.neighbors import KNeighborsRegressor

        A, B, _ = one_hot(Xtr, Xte, cats)
        return KNeighborsRegressor(10, weights="distance").fit(A, ytr).predict(B)

    def dist(self, Xtr, cats, ytr, Xte):
        rng = np.random.default_rng(0)
        ytr = np.asarray(ytr, float)
        base = self.point(Xtr, cats, ytr, Xte)
        boots = [base]
        for _ in range(4):
            i = rng.integers(0, len(ytr), len(ytr))
            boots.append(self.point(Xtr.iloc[i], cats, ytr[i], Xte))
        return Dist.gaussian(base, np.std(boots, 0) + 0.05, "bootstrap")


def build(name: str, fast: bool = False):
    if name == "gp":
        return GPModel()
    if name == "gbdt":
        return GBDTModel()
    if name == "knn":
        return KNNInContext()
    nb, nf = (4, 3) if (fast or name == "mitra") else (8, 5)
    return InContextModel(name, n_bootstrap=nb, n_folds=nf)


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, key):
        timer = self

        class _T:
            def __enter__(self):
                self.s = time.perf_counter()

            def __exit__(self, *a):
                timer.t.setdefault(key, []).append(time.perf_counter() - self.s)

        return _T()
