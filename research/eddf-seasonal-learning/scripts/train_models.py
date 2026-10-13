"""Простые модели, перенос между сезонами и влияние числа обучающих дней"""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
from lfpg_training import summarize

SEED = 20261013
SEASONS = ['winter', 'spring', 'summer', 'autumn']
FAMILIES = {
    'Ridge': [{'alpha': a} for a in [1, 10, 100]],
    'Polynomial Ridge': [{'alpha': a} for a in [1, 10, 100]]}


def make_model(name, parameters):
    steps = [('scale', StandardScaler())]
    if name == 'Polynomial Ridge':
        steps.extend([('poly', PolynomialFeatures(2, include_bias=False)),
                      ('expanded_scale', StandardScaler())])
    assert name in FAMILIES
    estimator = Ridge(**parameters, solver='svd')
    return Pipeline(steps + [('regressor', estimator)])


def fit(name, parameters, frame, features):
    weight = 1 / frame.groupby('event_id').event_id.transform('size')
    weight = weight * len(frame) / weight.sum()
    model = make_model(name, parameters)
    kwargs = {'scale__sample_weight': weight, 'regressor__sample_weight': weight}
    if name == 'Polynomial Ridge':
        kwargs['expanded_scale__sample_weight'] = weight
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(frame[features], frame.residual_ft, **kwargs)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    assert np.allclose(model.named_steps['scale'].mean_,
                       np.average(frame[features], weights=weight, axis=0))
    return model, converged


def forecast(model, frame, features):
    result = frame.baseline_ft.to_numpy() + model.predict(frame[features])
    assert np.isfinite(result).all()
    return result


def choose_days(expected):
    candidates = pd.read_csv(BASE / 'data/weather_candidates_2025.csv').sort_values('date')
    rows = []
    for item in candidates.itertuples(index=False):
        file = BASE / 'data' / item.date / 'examples.parquet'
        if file.exists():
            examples = pd.read_parquet(file)
            if examples.event_id.nunique() >= 10 and len(examples) >= 300:
                rows.append({'day': item.date, 'season': item.season,
                             'approaches': examples.event_id.nunique(), 'examples': len(examples)})
    inventory = pd.DataFrame(rows, columns=['day', 'season', 'approaches', 'examples'])
    found = inventory.groupby('season').size().to_dict()
    missing = {s: {'found': found.get(s, 0), 'needed': n}
               for s, n in expected.items() if found.get(s, 0) < n}
    if missing:
        raise RuntimeError('Не хватает подтвержденных дней: ' + json.dumps(missing))
    chosen = pd.concat([inventory.loc[inventory.season.eq(s)].head(n)
                        for s, n in expected.items()], ignore_index=True)
    chosen['split'] = 'train'
    for season in expected:
        indexes = chosen.index[chosen.season.eq(season)]
        chosen.loc[indexes[-2], 'split'] = 'validation'
        chosen.loc[indexes[-1], 'split'] = 'test'
    (BASE / 'reports/csv').mkdir(parents=True, exist_ok=True)
    chosen.to_csv(BASE / 'reports/csv/selected_days.csv', index=False)
    return chosen


def run(winter_days=9, seasons=None, days_per_season=5):
    seasons = seasons or ['spring', 'summer', 'autumn']
    expected = {s: winter_days if s == 'winter' else days_per_season for s in seasons}
    chosen = choose_days(expected)
    refresh = []
    for day in chosen.day:
        snapshot = pd.read_parquet(BASE / 'data' / day / 'examples.parquet')
        if 'airport_elevation_ft' not in snapshot or not snapshot.airport_elevation_ft.eq(364).all():
            refresh.append(day)
    if refresh:
        from prepare_examples import prepare
        with ProcessPoolExecutor(max_workers=3) as pool:
            list(pool.map(prepare, refresh))
        chosen = choose_days(expected)
    features = json.loads((PROJECT / 'configs/eddf_07l_training.json').read_text(encoding='utf-8'))['features']
    frame = pd.concat([pd.read_parquet(BASE / 'data' / row.day / 'examples.parquet')
                       .assign(season=row.season, split=row.split)
                       for row in chosen.itertuples(index=False)], ignore_index=True)
    assert frame.groupby('event_id').split.nunique().eq(1).all()
    assert frame.groupby('event_id').day.nunique().eq(1).all()
    assert frame.origin_metar_available_time.le(frame.origin_time).all()
    assert frame.target_time.lt(frame.boundary_time).all()
    assert frame.horizon_s.between(27, 33).all()
    search, parameters, models, metrics, learning = [], {}, {}, [], []
    tests = {s: frame.loc[frame.season.eq(s) & frame.split.eq('test')] for s in seasons}
    tests['all'] = frame.loc[frame.split.eq('test')]
    for source in seasons + ['pooled']:
        source_mask = frame.season.eq(source) if source != 'pooled' else pd.Series(True, index=frame.index)
        train = frame.loc[source_mask & frame.split.eq('train')]
        validation = frame.loc[source_mask & frame.split.eq('validation')]
        for name, options in FAMILIES.items():
            candidates = []
            for option in options:
                model, converged = fit(name, option, train, features)
                score = summarize(validation, forecast(model, validation, features), name)['macro_mae_ft']
                search.append({'source': source, 'model': name, 'parameters': json.dumps(option),
                               'validation_mae_ft': score, 'converged': converged})
                if converged:
                    candidates.append((score, option, model))
            assert candidates, f'Нет сошедшихся вариантов {source}: {name}'
            _, option, model = min(candidates, key=lambda item: item[0])
            parameters[source, name], models[source, name] = option, model
            for target, test in tests.items():
                result = summarize(test, forecast(model, test, features), name)
                metrics.append({**result, 'source': source, 'target': target,
                                'train_days': train.day.nunique(), 'train_approaches': train.event_id.nunique()})
        print('TRAINED', source, train.day.nunique(), 'days', flush=True)
    for target, test in tests.items():
        metrics.append({**summarize(test, test.baseline_ft, 'Constant vertical rate'),
                        'source': 'baseline', 'target': target, 'train_days': 0, 'train_approaches': 0})
    for season in seasons:
        train = frame.loc[frame.season.eq(season) & frame.split.eq('train')]
        pool = sorted(train.day.unique())
        sizes = sorted({n for n in [1, 2, 3, 5, len(pool)] if n <= len(pool)})
        test = tests[season]
        for size in sizes:
            days = pool[:size]
            subset = train.loc[train.day.isin(days)]
            for name in FAMILIES:
                model, converged = fit(name, parameters[season, name], subset, features)
                result = summarize(test, forecast(model, test, features), name)
                learning.append({**result, 'season': season,
                                 'train_days': size, 'train_approaches': subset.event_id.nunique(),
                                 'training_dates': '|'.join(days), 'converged': converged})
    report = pd.DataFrame(metrics)
    report.to_csv(BASE / 'reports/csv/model_metrics.csv', index=False)
    pd.DataFrame(search).to_csv(BASE / 'reports/csv/model_search.csv', index=False)
    pd.DataFrame(learning).to_csv(BASE / 'reports/csv/learning_curve.csv', index=False)
    final = report.loc[report.source.isin(['pooled', 'baseline'])].pivot(
        index='model', columns='target', values='macro_mae_ft').reindex(
        list(FAMILIES) + ['Constant vertical rate'])[seasons + ['all']]
    final.to_csv(BASE / 'reports/csv/final_table.csv')
    artifacts = BASE / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    joblib.dump({'models': models, 'parameters': parameters, 'features': features}, artifacts / 'models.joblib')
    print(final.to_string(), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--winter-days', type=int, default=9, choices=range(4, 10))
    parser.add_argument('--seasons', nargs='+', choices=SEASONS,
                        default=['spring', 'summer', 'autumn'])
    parser.add_argument('--days-per-season', type=int, default=5, choices=range(3, 11))
    args = parser.parse_args()
    run(args.winter_days, args.seasons, args.days_per_season)
