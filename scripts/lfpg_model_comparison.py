"""Модели поправки, контроль весов и парное сравнение целыми бортами"""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def make_model(kind, parameters, seed):
    classes = {'ridge': Ridge, 'elasticnet': ElasticNet, 'huber': HuberRegressor,
               'svr': SVR, 'randomforest': RandomForestRegressor,
               'extratrees': ExtraTreesRegressor, 'histgb': HistGradientBoostingRegressor}
    params = dict(parameters)
    if kind in {'elasticnet', 'randomforest', 'extratrees', 'histgb'}:
        params['random_state'] = seed
    estimator = classes[kind](**params)
    if kind in {'ridge', 'elasticnet', 'huber', 'svr'}:
        return Pipeline([('scale', StandardScaler()), ('regressor', estimator)])
    return estimator


def fit_model(model, frame, features, weights):
    if isinstance(model, Pipeline):
        model.fit(frame[features], frame.residual_ft, scale__sample_weight=weights,
                  regressor__sample_weight=weights)
        expected_mean = np.average(frame[features], axis=0, weights=weights)
        assert np.allclose(model.named_steps['scale'].mean_, expected_mean)
    else:
        model.fit(frame[features], frame.residual_ft, sample_weight=weights)
    return model


def forecast(model, frame, features):
    if model == 'Constant vertical rate':
        return frame.baseline_ft.to_numpy()
    if model == 'Past height rate':
        return (frame.height_ft + frame.past_height_rate_fps * frame.horizon_s).to_numpy()
    return frame.baseline_ft.to_numpy() + model.predict(frame[features])


def paired_intervals(per_event, reference, seed, repetitions):
    wide = per_event.pivot(index=['event_id', 'icao24'], columns='model', values='mae_ft')
    assert wide.notna().all().all()
    delta = wide.subtract(wide[reference], axis=0)
    sums = delta.groupby(level='icao24').sum()
    counts = delta.groupby(level='icao24').size().to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(counts), size=(repetitions, len(counts)))
    bootstrap = sums.to_numpy()[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None]
    rows = []
    for i, name in enumerate(delta.columns):
        rows.append({'model': name, 'reference': reference,
            'delta_macro_mae_ft': float(delta[name].mean()),
            'interval_2_5_ft': float(np.quantile(bootstrap[:, i], .025)),
            'interval_97_5_ft': float(np.quantile(bootstrap[:, i], .975)),
            'independent_aircraft': len(counts), 'events': len(wide),
            'repetitions': repetitions, 'resampling_unit': 'icao24'})
    return pd.DataFrame(rows)


def current_rate_response(model, frame, features, delta):
    # Изолированная смена Vz служит диагностикой формулы, не имитацией полета
    plus, minus = frame.copy(), frame.copy()
    plus['vertical_rate_fps'] += delta
    minus['vertical_rate_fps'] -= delta
    for changed in [plus, minus]:
        changed['baseline_ft'] = changed.height_ft + changed.vertical_rate_fps * changed.horizon_s
    response = (forecast(model, plus, features) - forecast(model, minus, features))
    response = response / (2 * delta * frame.horizon_s.to_numpy())
    return {'median_vz_response_ratio': float(np.median(response)),
            'p10_vz_response_ratio': float(np.quantile(response, .1)),
            'p90_vz_response_ratio': float(np.quantile(response, .9))}
