"""Обучение высотного прогноза на нескольких днях LFPG 09L"""
from pathlib import Path
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit, ParameterGrid
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits
from lfpg_model_comparison import make_model, paired_intervals
from lfpg_training import summarize, event_errors

SEED = 20261013
BASELINES = ['Constant vertical rate', 'Past height rate']


def families(project):
    prior = json.loads((project / 'configs/lfpg_09l_model_comparison.json').read_text(encoding='utf-8'))
    return prior['model_families'] + [
        {'name': 'Gradient Boosting', 'kind': 'gb',
         'fixed': {'n_estimators': 200, 'min_samples_leaf': 10},
         'grid': {'max_depth': [2, 3], 'learning_rate': [.03, .1]}},
        {'name': 'Polynomial Ridge', 'kind': 'poly', 'fixed': {},
         'grid': {'alpha': [1, 10, 100, 1000]}},
        {'name': 'CatBoost', 'kind': 'cat',
         'fixed': {'iterations': 400, 'loss_function': 'RMSE', 'l2_leaf_reg': 3,
                   'verbose': False, 'allow_writing_files': False, 'thread_count': 1},
         'grid': {'depth': [3, 5], 'learning_rate': [.03, .1]}},
        {'name': 'MLP', 'kind': 'mlp',
         'fixed': {'solver': 'lbfgs', 'max_iter': 3000, 'max_fun': 100000, 'tol': .00001},
         'grid': {'hidden_layer_sizes': [(32,), (64, 32)], 'alpha': [.1]}}]


def model_for(kind, parameters):
    if kind == 'gb':
        return GradientBoostingRegressor(**parameters, random_state=SEED)
    if kind == 'cat':
        return CatBoostRegressor(**parameters, random_seed=SEED)
    if kind == 'poly':
        return Pipeline([('scale', StandardScaler()),
                         ('poly', PolynomialFeatures(2, include_bias=False)),
                         ('expanded_scale', StandardScaler()),
                         ('regressor', Ridge(**parameters, solver='svd'))])
    if kind == 'mlp':
        return Pipeline([('scale', StandardScaler()),
            ('regressor', TransformedTargetRegressor(
                regressor=MLPRegressor(**parameters, random_state=SEED),
                transformer=StandardScaler()))])
    return make_model(kind, parameters, SEED)


def weights_for(frame):
    weights = 1 / frame.groupby('event_id').event_id.transform('size')
    return weights * len(frame) / weights.sum()


def fit(model, frame, features, mode):
    weights = weights_for(frame)
    target = frame.residual_ft if mode == 'residual' else frame.target_height_ft
    with threadpool_limits(limits=1):
        if isinstance(model, Pipeline):
            kwargs = {'scale__sample_weight': weights, 'regressor__sample_weight': weights}
            if 'expanded_scale' in model.named_steps:
                kwargs['expanded_scale__sample_weight'] = weights
            model.fit(frame[features], target, **kwargs)
            assert np.allclose(model.named_steps['scale'].mean_,
                               np.average(frame[features], axis=0, weights=weights))
        else:
            model.fit(frame[features], target, sample_weight=weights)
    return model


def predict(model, frame, features, mode):
    if isinstance(model, str):
        if model == BASELINES[0]:
            return frame.baseline_ft.to_numpy()
        return (frame.height_ft + frame.past_height_rate_fps * frame.horizon_s).to_numpy()
    result = model.predict(frame[features])
    if mode == 'residual':
        result = result + frame.baseline_ft.to_numpy()
    assert np.isfinite(result).all()
    return result


def load_data(base):
    files = {'2025-01-15': 'training_04_examples.parquet',
             '2025-01-16': 'external_examples.parquet',
             '2025-01-17': 'external_examples.parquet',
             '2025-01-18': 'external_examples.parquet'}
    data = pd.concat([pd.read_parquet(base / 'data' / day / name).assign(day=day)
                      for day, name in files.items()], ignore_index=True)
    features = json.loads((base.parents[1] / 'configs/lfpg_09l_training.json').read_text(encoding='utf-8'))['features']
    assert data.groupby('event_id').day.nunique().eq(1).all()
    assert not data.duplicated(['event_id', 'origin_time', 'target_time']).any()
    assert np.isfinite(data[features + ['target_height_ft']]).all().all()
    assert data.origin_metar_available_time.le(data.origin_time).all()
    assert data.target_time.gt(data.origin_time).all()
    assert data.target_time.lt(data.boundary_time).all()
    assert data.horizon_s.between(27, 33).all()
    assert np.allclose(data.target_time - data.origin_time, data.horizon_s)
    assert np.allclose(data.baseline_ft, data.height_ft + data.vertical_rate_fps * data.horizon_s)
    assert np.allclose(data.residual_ft, data.target_height_ft - data.baseline_ft)
    assert len(data) == 3259 and data.event_id.nunique() == 185
    data['example_id'] = np.arange(len(data))
    return data, features


def split_data(data, scheme):
    frame = data.copy()
    if scheme == 'next_day':
        frame['split'] = frame.day.map({'2025-01-15': 'train', '2025-01-16': 'train',
                                        '2025-01-17': 'validation', '2025-01-18': 'test'})
        assert frame.loc[frame.split.eq('train'), 'target_time'].max() < frame.loc[frame.split.eq('validation'), 'origin_time'].min()
    else:
        events = frame[['event_id', 'icao24']].drop_duplicates().sort_values('event_id').reset_index(drop=True)
        dev, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(events, groups=events.icao24))
        train, val = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SEED+1).split(events.iloc[dev], groups=events.iloc[dev].icao24))
        events['split'] = 'test'
        events.loc[dev[train], 'split'] = 'train'
        events.loc[dev[val], 'split'] = 'validation'
        frame['split'] = frame.event_id.map(events.set_index('event_id').split)
        assert frame.groupby('icao24').split.nunique().eq(1).all()
    assert frame.groupby('event_id').split.nunique().eq(1).all()
    assert frame.groupby(['day', 'trace_id']).split.nunique().eq(1).all()
    return frame


def run(base, progress=None):
    project = base.parents[1]
    reports, artifacts = base / 'reports', base / 'artifacts/training_06'
    artifacts.mkdir(parents=True, exist_ok=True)
    data, features = load_data(base)
    specs = families(project)
    metrics, searches, chosen_rows, errors, predictions, split_rows = [], [], [], [], [], []
    bundle = {'features': features, 'models': {}, 'winner_by_validation': {}}
    selections = {}
    for scheme in ['next_day', 'new_aircraft']:
        frame = split_data(data, scheme)
        train, val, test = [frame.loc[frame.split.eq(s)].copy() for s in ['train', 'validation', 'test']]
        split_rows.append(frame[['example_id', 'day', 'event_id', 'icao24', 'split']].assign(scheme=scheme))
        selected = {name: {'model': name, 'mode': 'physical', 'family': name} for name in BASELINES}
        for spec in specs:
            for mode in ['residual', 'direct']:
                candidates = []
                for index, grid in enumerate(ParameterGrid(spec['grid'])):
                    parameters = {**spec['fixed'], **grid}
                    model = model_for(spec['kind'], parameters)
                    start = time.perf_counter()
                    with warnings.catch_warnings(record=True) as recorded:
                        warnings.simplefilter('always')
                        fit(model, train, features, mode)
                    converged = not any(issubclass(w.category, ConvergenceWarning) for w in recorded)
                    score = summarize(val, predict(model, val, features, mode), spec['name'])
                    row = {'scheme': scheme, 'family': spec['name'], 'mode': mode,
                           'candidate': index, 'parameters': json.dumps(parameters),
                           'validation_mae_ft': score['macro_mae_ft'], 'converged': converged,
                           'seconds': time.perf_counter()-start,
                           'warnings': ' | '.join(str(w.message) for w in recorded)}
                    searches.append(row)
                    if converged:
                        candidates.append((score['macro_mae_ft'], index, model, parameters))
                if not candidates:
                    raise RuntimeError(f'Нет сошедшейся модели {spec["name"]} {mode}')
                score, index, model, parameters = min(candidates, key=lambda item: (item[0], item[1]))
                name = spec['name'] + ' / ' + mode
                selected[name] = {'model': model, 'mode': mode, 'family': spec['name']}
                chosen_rows.append({'scheme': scheme, 'model': name, 'family': spec['name'],
                                    'mode': mode, 'parameters': json.dumps(parameters), 'validation_mae_ft': score})
                if progress:
                    progress(f'{scheme}: {name}, validation MAE {score:.2f} ft')
        selection = pd.DataFrame(chosen_rows)
        selection = selection.loc[selection.scheme.eq(scheme)].sort_values(['validation_mae_ft', 'model'])
        selections[scheme] = selection.groupby('family', sort=False).first().reset_index()
        winner = selection.iloc[0]['model']
        bundle['winner_by_validation'][scheme] = winner
        bundle['models'][scheme] = selected
        # Все параметры уже выбраны на validation перед первой оценкой test
        for name, item in selected.items():
            validation_score = summarize(val, predict(item['model'], val, features, item['mode']), name)['macro_mae_ft']
            forecast = predict(item['model'], test, features, item['mode'])
            result = summarize(test, forecast, name)
            result.update({'scheme': scheme, 'family': item['family'], 'mode': item['mode'],
                           'validation_mae_ft': validation_score, 'winner_by_validation': name == winner})
            metrics.append(result)
            errors.append(event_errors(test, forecast, name).assign(scheme=scheme))
            predictions.append(test[['example_id', 'event_id', 'icao24', 'day', 'target_height_ft']].assign(
                scheme=scheme, model=name, prediction_ft=forecast))
    metrics = pd.DataFrame(metrics)
    per_event = pd.concat(errors, ignore_index=True)
    intervals = pd.concat([paired_intervals(per_event.loc[per_event.scheme.eq(scheme)],
                BASELINES[0], SEED, 5000).assign(scheme=scheme)
                for scheme in bundle['models']], ignore_index=True)
    family_tables = []
    for scheme, selection in selections.items():
        selected_names = set(selection.model) | set(BASELINES)
        family_tables.append(metrics.loc[metrics.scheme.eq(scheme) & metrics.model.isin(selected_names)])
    final = pd.concat(family_tables, ignore_index=True)
    learning = []
    selected_name = bundle['winner_by_validation']['next_day']
    item = bundle['models']['next_day'][selected_name]
    spec = next(s for s in specs if s['name'] == item['family'])
    row = next(r for r in chosen_rows if r['scheme'] == 'next_day' and r['model'] == selected_name)
    params = json.loads(row['parameters'])
    temporal = split_data(data, 'next_day')
    train = temporal.loc[temporal.split.eq('train')]
    test = temporal.loc[temporal.split.eq('test')]
    for days in [['2025-01-15'], ['2025-01-16'], ['2025-01-15', '2025-01-16']]:
        subset = train.loc[train.day.isin(days)]
        if len(days) == 2:
            model = item['model']
        else:
            model = fit(model_for(spec['kind'], params), subset, features, item['mode'])
        learning.append({'training_days': ', '.join(days), 'training_events': subset.event_id.nunique(),
                         'training_examples': len(subset), 'model': selected_name,
                         **summarize(test, predict(model, test, features, item['mode']), selected_name)})
    # Диагностика влияния признаков выполняется только на validation нового дня
    val = temporal.loc[temporal.split.eq('validation')]
    reference = summarize(val, predict(item['model'], val, features, item['mode']), selected_name)['macro_mae_ft']
    rng = np.random.default_rng(SEED)
    importance = []
    for feature in features:
        changes = []
        for _ in range(10):
            changed = val.copy()
            changed[feature] = rng.permutation(changed[feature].to_numpy())
            changed['baseline_ft'] = changed.height_ft + changed.vertical_rate_fps * changed.horizon_s
            changes.append(summarize(changed, predict(item['model'], changed, features, item['mode']), selected_name)['macro_mae_ft']-reference)
        importance.append({'feature': feature, 'validation_mae_increase_ft': np.mean(changes),
                           'std_ft': np.std(changes)})
    response = []
    for name, selected_item in bundle['models']['next_day'].items():
        plus, minus = val.copy(), val.copy()
        for changed, delta in [(plus, 1), (minus, -1)]:
            changed['vertical_rate_fps'] += delta
            changed['baseline_ft'] = changed.height_ft + changed.vertical_rate_fps * changed.horizon_s
        ratio = (predict(selected_item['model'], plus, features, selected_item['mode']) -
                 predict(selected_item['model'], minus, features, selected_item['mode'])) / (2*val.horizon_s)
        response.append({'model': name, 'median_vz_response_ratio': np.median(ratio),
                         'p10': np.quantile(ratio,.1), 'p90':np.quantile(ratio,.9)})
    outputs = {'metrics': metrics, 'family_results': final, 'search': pd.DataFrame(searches),
               'selected': pd.DataFrame(chosen_rows), 'event_errors': per_event, 'intervals': intervals,
               'learning': pd.DataFrame(learning), 'importance': pd.DataFrame(importance),
               'vz_response': pd.DataFrame(response)}
    for name, table in outputs.items():
        table.to_csv(reports / ('06_' + name + '.csv'), index=False)
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_parquet(artifacts / 'test_predictions.parquet', index=False)
    splits = pd.concat(split_rows, ignore_index=True)
    splits.to_parquet(artifacts / 'splits.parquet', index=False)
    joblib.dump(bundle, artifacts / 'models.joblib', compress=3)
    return {'data': data, 'features': features, 'bundle': bundle, 'splits': splits,
            'predictions': predictions, **outputs}
