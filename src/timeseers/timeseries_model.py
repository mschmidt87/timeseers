import pandas as pd
import pymc3 as pm
from timeseers.utils import MinMaxScaler, StdScaler, add_subplot, get_group_definition
import numpy as np
from abc import ABC, abstractmethod
import theano.tensor as tt

class TimeSeriesModel(ABC):
    def __init__(self, likelihood='gaussian', variance_prior=0.5, pool_cols=None, pool_type='complete'):
        self.likelihood = likelihood
        self.variance_prior = variance_prior
        self.pool_cols = pool_cols
        self.pool_type = pool_type

    def fit(self, X, y, X_scaler=MinMaxScaler, y_scaler=StdScaler, **sample_kwargs):
        with self.__call__(X, y, X_scaler=MinMaxScaler, y_scaler=StdScaler, **sample_kwargs):
            self.trace_ = pm.sample(**sample_kwargs)

    def plot_components(self, X_true=None, y_true=None, groups=None, fig=None):
        import matplotlib.pyplot as plt

        if fig is None:
            fig = plt.figure(figsize=(18, 1))

        lookahead_scale = 0.3
        t_min, t_max = self._X_scaler_.min_["t"], self._X_scaler_.max_["t"]
        t_max += (t_max - t_min) * lookahead_scale
        t = pd.date_range(t_min, t_max, freq='D')

        scaled_t = np.linspace(0, 1 + lookahead_scale, len(t))
        total = self.plot(self.trace_, scaled_t, self._y_scaler_)

        ax = add_subplot()
        ax.set_title("overall")
        ax.plot(t, self._y_scaler_.inv_transform(total))

        if X_true is not None and y_true is not None:
            if groups is not None:
                for group in groups.cat.categories:
                    mask = groups == group
                    ax.scatter(X_true["t"][mask], y_true[mask], label=group, marker='.', alpha=0.2)
            else:
                ax.scatter(X_true["t"], y_true, c="k", marker='.', alpha=0.2)
        fig.tight_layout()
        return fig

    def __call__(self, X, y, X_scaler=MinMaxScaler, y_scaler=StdScaler, ppc=False):
        if not X.index.is_monotonic_increasing:
            raise ValueError(
                'index of X is not monotonically increasing. You might want to call `.reset_index()`')

        X_to_scale = X.select_dtypes(exclude='category')
        self._X_scaler_ = X_scaler()
        self._y_scaler_ = y_scaler()

        X_scaled = self._X_scaler_.fit_transform(X_to_scale)
        y_scaled = self._y_scaler_.fit_transform(y)
        model = pm.Model()
        X_scaled = X_scaled.join(X.select_dtypes('category'))
        del X
        mu = self.definition(
            model, X_scaled, self._X_scaler_.scale_factor_
        )
        with model:
            if self.likelihood == 'gaussian':
                sigma = pm.HalfCauchy("sigma", 0.5)
                pm.Normal("obs", mu=mu, sd=sigma, observed=y_scaled)
            elif self.likelihood == 'negative_binomial':
                alpha = pm.HalfCauchy("alpha", self.variance_prior)
                lam = pm.Deterministic('lambda', tt.exp(mu))
                pm.NegativeBinomial("obs", mu=lam, alpha=alpha, observed=y_scaled)
            elif self.likelihood == 'poisson':
                lam = pm.Deterministic('lambda', tt.exp(mu))
                pm.Poisson ("obs", mu=lam, observed=y_scaled, testval=1.e6)
            elif self.likelihood == 'multi_outcome_negative_binomial':
                lam = pm.Deterministic('lambda', tt.exp(mu))
                alpha = pm.HalfCauchy("alpha", self.variance_prior)

                group, n_groups, self.groups_ = get_group_definition(X_scaled, self.pool_cols,
                                                                     self.pool_type)
                idx = y_scaled[:, 0]
                # if not isinstance(y_scaled, np.ndarray):
                #     idx = idx.eval()
                p = pm.Dirichlet('p', a=np.array([1., 1., 1.]), shape=(n_groups, 3))
                pm.Categorical("obs_sign", p=p[group[idx]], observed=y_scaled[:, 2])
                # if isinstance(y_scaled, np.ndarray):
                idn0 = y_scaled[:, 1] != 0
                # else:
                #     idn0 = y_scaled[:, 1].eval() != 0
                if ppc:
                    idn0 = np.ones_like(y_scaled[:, 1], dtype=np.bool)
                pm.NegativeBinomial("obs_abs_value", mu=lam[idx][idn0], alpha=alpha,
                                    observed=y_scaled[:, 1][idn0],
                                    total_size=len(X_scaled))

        return model

    @abstractmethod
    def plot(self, trace, t, y_scaler):
        pass

    @abstractmethod
    def definition(self, model, X_scaled, scale_factor):
        pass

    def _param_name(self, param):
        return f"{self.name}-{param}"

    def __add__(self, other):
        return AdditiveTimeSeries(self, other)

    def __mul__(self, other):
        return MultiplicativeTimeSeries(self, other)

    def __str__(self):
        return self.name


class AdditiveTimeSeries(TimeSeriesModel):
    def __init__(self, left, right):
        self.left = left
        self.right = right
        assert self.left.likelihood == self.right.likelihood
        super().__init__(likelihood=self.left.likelihood,)

    def definition(self, *args, **kwargs):
        return self.left.definition(*args, **kwargs) + self.right.definition(
            *args, **kwargs
        )

    def plot(self, *args, **kwargs):
        left = self.left.plot(*args, **kwargs)
        right = self.right.plot(*args, **kwargs)
        return left + right

    def __repr__(self):
        return (
            f"AdditiveTimeSeries( \n"
            f"    left={self.left} \n"
            f"    right={self.right} \n"
            f")"
        )


class MultiplicativeTimeSeries(TimeSeriesModel):
    def __init__(self, left, right):
        self.left = left
        self.right = right
        assert self.left.likelihood == self.right.likelihood
        super().__init__(likelihood=self.left.likelihood)

    def definition(self, *args, **kwargs):
        return self.left.definition(*args, **kwargs) * (
            1 + self.right.definition(*args, **kwargs)
        )

    def plot(self, trace, scaled_t, y_scaler):
        left = self.left.plot(trace, scaled_t, y_scaler)
        right = self.right.plot(trace, scaled_t, y_scaler)
        return left + (left * right)

    def __repr__(self):
        return (
            f"MultiplicativeTimeSeries( \n"
            f"    left={self.left} \n"
            f"    right={self.right} \n"
            f")"
        )
