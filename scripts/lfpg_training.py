"""Причинные признаки и независимая проверка прогноза высотной динамики"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def select_cohort(events, protocol):
    gap = protocol['maximum_trace_gap_s']
    selected = events.loc[events.direction.eq('arrival') & events.runway.eq('09L')
        & events.cross300_observed & events.cross300_gap_s.le(gap)
        & events.precontrol_max_gap_s.le(gap)
        & events.precontrol_span_s.ge(protocol['minimum_precontrol_span_s'])
        & ~events.parallel_ambiguity].sort_values('start_time').copy()
    assert len(selected) == protocol['cohort_size']
    assert selected.event_id.is_unique
    return selected.reset_index(drop=True)


def split_cohort(cohort, protocol):
    result = cohort.copy()
    # Фиксируем группы до построения примеров и до просмотра ошибок моделей
    first = GroupShuffleSplit(n_splits=1, test_size=protocol['test_group_fraction'],
                              random_state=protocol['seed'])
    development, test = next(first.split(result, groups=result.icao24))
    second = GroupShuffleSplit(n_splits=1,
        test_size=protocol['validation_remaining_group_fraction'],
        random_state=protocol['seed'] + 1)
    train, validation = next(second.split(result.iloc[development],
                                          groups=result.iloc[development].icao24))
    result['split'] = 'test'
    result.loc[result.index[development[train]], 'split'] = 'train'
    result.loc[result.index[development[validation]], 'split'] = 'validation'
    assert result.groupby('icao24').split.nunique().eq(1).all()
    assert result.groupby('trace_id').split.nunique().eq(1).all()
    return result


def qnh_factor(qnh):
    return (1013.25 / qnh) ** (287.05287 * .0065 / 9.80665)


def height_in_setting(pressure_altitude_ft, qnh, elevation_ft):
    factor = qnh_factor(qnh)
    return (288.15 / .0065) * (1 - factor) / .3048 + factor * pressure_altitude_ft - elevation_ft


def causal_features(prefix, horizon_s, bearing_deg, protocol):
    current = prefix.iloc[-1]
    # Вход функции содержит только наблюдения не позже момента прогноза
    age = current.time - prefix.time
    history = prefix.loc[age.between(protocol['history_minimum_s'],
                                      protocol['history_maximum_s'])]
    if history.empty or bool(current.geometric_rate):
        return None
    previous = history.iloc[-1]
    gap = float(current.time - previous.time)
    factor = qnh_factor(float(current.qnh_hpa))
    values = {
        'height_ft': height_in_setting(float(current.altitude_ft), float(current.qnh_hpa),
                                      protocol['airport_elevation_ft']),
        'vertical_rate_fps': factor * float(current.vertical_rate_fpm) / 60,
        'past_height_rate_fps': factor * float(current.altitude_ft - previous.altitude_ft) / gap,
        'speed_knots': float(current.speed_knots),
        'distance_m': float(current.distance_to_threshold_m),
        'lateral_m': float(current.lateral_m),
        'heading_error_deg': float((current.track_deg - bearing_deg + 180) % 360 - 180),
        'past_gap_s': gap,
        'horizon_s': float(horizon_s)}
    if not np.isfinite(list(values.values())).all():
        return None
    return values


def build_examples(cohort, points, geometry, protocol):
    rows, audits = [], []
    bearing = float(geometry.loc[geometry.runway.eq('09L'), 'bearing_deg'].iloc[0])
    for event in cohort.itertuples(index=False):
        group = points.loc[points.event_id.eq(event.event_id)].sort_values('time').reset_index(drop=True)
        assert group.time.diff().dropna().gt(0).all()
        cross = group.height_proxy_ft.shift().gt(protocol['control_height_proxy_ft']) & group.height_proxy_ft.le(protocol['control_height_proxy_ft'])
        boundary = float(group.loc[cross, 'time'].iloc[0])
        # Нижнюю контрольную границу не используем как целевую точку
        group = group.loc[group.time.lt(boundary)
                          & group.height_proxy_ft.gt(protocol['control_height_proxy_ft'])].reset_index(drop=True)
        count = {'event_id': event.event_id, 'icao24': event.icao24, 'split': event.split,
                 'precontrol_points': len(group), 'boundary_time': boundary,
                 'no_target_at_horizon': 0, 'missing_causal_features': 0,
                 'target_below_control_in_origin_qnh': 0, 'examples': 0}
        times = group.time.to_numpy()
        for index, current in group.iterrows():
            future = np.flatnonzero(times > current.time)
            if not len(future):
                count['no_target_at_horizon'] += 1
                continue
            target_index = future[np.argmin(np.abs(times[future] - current.time - protocol['nominal_horizon_s']))]
            horizon = float(times[target_index] - current.time)
            if abs(horizon - protocol['nominal_horizon_s']) > protocol['target_time_tolerance_s']:
                count['no_target_at_horizon'] += 1
                continue
            features = causal_features(group.iloc[:index + 1], horizon, bearing, protocol)
            if features is None:
                count['missing_causal_features'] += 1
                continue
            target = group.iloc[target_index]
            target_height = height_in_setting(float(target.altitude_ft), float(current.qnh_hpa),
                                             protocol['airport_elevation_ft'])
            if target_height <= protocol['control_height_proxy_ft']:
                count['target_below_control_in_origin_qnh'] += 1
                continue
            baseline = features['height_ft'] + features['vertical_rate_fps'] * horizon
            rows.append({**features, 'event_id': event.event_id, 'icao24': event.icao24,
                'trace_id': event.trace_id, 'split': event.split, 'origin_time': float(current.time),
                'target_time': float(target.time), 'target_height_ft': target_height,
                'baseline_ft': baseline, 'residual_ft': target_height - baseline,
                'origin_qnh_hpa': float(current.qnh_hpa), 'target_qnh_hpa': float(target.qnh_hpa),
                'origin_metar_time': float(current.observation_utc.timestamp()),
                'origin_metar_available_time': float(current.assumed_available_utc.timestamp()),
                'boundary_time': boundary})
            count['examples'] += 1
        audits.append(count)
    frame, audit = pd.DataFrame(rows), pd.DataFrame(audits)
    assert frame.event_id.nunique() == len(cohort), 'Не все кандидаты дают обучающие примеры'
    assert frame.groupby('icao24').split.nunique().eq(1).all()
    assert frame.target_time.gt(frame.origin_time).all()
    assert frame.target_time.lt(frame.boundary_time).all()
    assert frame.origin_metar_available_time.le(frame.origin_time).all()
    assert frame.origin_metar_time.le(frame.origin_time).all()
    assert frame.horizon_s.between(27, 33).all()
    return frame, audit


def event_errors(frame, prediction, name):
    errors = pd.DataFrame({'event_id': frame.event_id, 'icao24': frame.icao24,
                          'error': np.asarray(prediction) - frame.target_height_ft.to_numpy()})
    errors['absolute'] = errors.error.abs()
    errors['squared'] = errors.error ** 2
    result = errors.groupby(['event_id', 'icao24']).agg(
        mae_ft=('absolute', 'mean'), mse_ft2=('squared', 'mean'),
        bias_ft=('error', 'mean'), examples=('error', 'size')).reset_index()
    result['model'] = name
    return result


def summarize(frame, prediction, name):
    events = event_errors(frame, prediction, name)
    error = np.asarray(prediction) - frame.target_height_ft.to_numpy()
    return {'model': name, 'events': len(events), 'examples': len(frame),
            'macro_mae_ft': float(events.mae_ft.mean()),
            'macro_rmse_ft': float(np.sqrt(events.mse_ft2.mean())),
            'point_mae_ft': float(np.abs(error).mean()),
            'point_p95_abs_ft': float(np.quantile(np.abs(error), .95)),
            'macro_bias_ft': float(events.bias_ft.mean())}


def paired_bootstrap(per_event, seed, repetitions):
    wide = per_event.pivot(index=['event_id', 'icao24'], columns='model', values='mae_ft').reset_index()
    groups = [group for _, group in wide.groupby('icao24')]
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        sample = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        differences.append(float((sample['Ridge correction'] - sample['Constant vertical rate']).mean()))
    observed = float((wide['Ridge correction'] - wide['Constant vertical rate']).mean())
    return {'resampling_unit': 'icao24', 'independent_groups': len(groups),
            'repetitions': repetitions, 'ridge_minus_baseline_mae_ft': observed,
            'conditional_interval_2_5_ft': float(np.quantile(differences, .025)),
            'conditional_interval_97_5_ft': float(np.quantile(differences, .975))}
