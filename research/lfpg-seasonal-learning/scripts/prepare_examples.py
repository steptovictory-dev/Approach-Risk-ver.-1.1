"""Подготовить наблюдаемые пары прогноза в выбранных сезонных днях"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
from lfpg_training import causal_features, height_in_setting


def examples_for(selected, points, geometry, settings):
    rows, audits = [], []
    bearing = float(geometry.loc[geometry.runway.eq('09L'), 'bearing_deg'].iloc[0])
    for event in selected.itertuples(index=False):
        group = points.loc[points.event_id.eq(event.event_id)].sort_values('time').reset_index(drop=True)
        audit = {'event_id': event.event_id, 'icao24': event.icao24, 'examples': 0, 'status': 'ready'}
        if not group.time.diff().dropna().gt(0).all():
            audit['status'] = 'nonunique_or_nonincreasing_time'
            audits.append(audit)
            continue
        cross = group.height_proxy_ft.shift().gt(settings['control_height_proxy_ft'])
        cross &= group.height_proxy_ft.le(settings['control_height_proxy_ft'])
        boundary = float(group.loc[cross, 'time'].iloc[0])
        group = group.loc[group.time.lt(boundary)
                          & group.height_proxy_ft.gt(settings['control_height_proxy_ft'])].reset_index(drop=True)
        times = group.time.to_numpy()
        for index, current in group.iterrows():
            future = np.flatnonzero(times > current.time)
            if not len(future):
                continue
            target_index = future[np.argmin(np.abs(times[future] - current.time - settings['nominal_horizon_s']))]
            horizon = float(times[target_index] - current.time)
            if abs(horizon - settings['nominal_horizon_s']) > settings['target_time_tolerance_s']:
                continue
            features = causal_features(group.iloc[:index + 1], horizon, bearing, settings)
            if features is None:
                continue
            target = group.iloc[target_index]
            target_height = height_in_setting(float(target.altitude_ft), float(current.qnh_hpa),
                                             settings['airport_elevation_ft'])
            if target_height <= settings['control_height_proxy_ft']:
                continue
            baseline = features['height_ft'] + features['vertical_rate_fps'] * horizon
            rows.append({**features, 'event_id': event.event_id, 'icao24': event.icao24,
                         'trace_id': event.trace_id, 'origin_time': float(current.time),
                         'target_time': float(target.time), 'target_height_ft': target_height,
                         'baseline_ft': baseline, 'residual_ft': target_height - baseline,
                         'origin_metar_time': float(current.observation_utc.timestamp()),
                         'origin_metar_available_time': float(current.assumed_available_utc.timestamp()),
                         'boundary_time': boundary})
            audit['examples'] += 1
        if not audit['examples']:
            audit['status'] = 'no_observed_target_pairs'
        audits.append(audit)
    columns = settings['features'] + ['event_id', 'icao24', 'trace_id', 'origin_time', 'target_time',
        'target_height_ft', 'baseline_ft', 'residual_ft', 'origin_metar_time',
        'origin_metar_available_time', 'boundary_time']
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        assert not frame.duplicated(['event_id', 'origin_time', 'target_time']).any()
        assert np.isfinite(frame[settings['features'] + ['target_height_ft']]).all().all()
        assert frame.origin_metar_available_time.le(frame.origin_time).all()
        assert frame.target_time.gt(frame.origin_time).all()
        assert frame.target_time.lt(frame.boundary_time).all()
        assert frame.horizon_s.between(27, 33).all()
    return frame, pd.DataFrame(audits, columns=['event_id', 'icao24', 'examples', 'status'])


def prepare(date):
    folder = BASE / 'data' / date
    selected = pd.read_csv(folder / 'selected_09l.csv', dtype={'icao24': str})
    points = pd.read_parquet(folder / 'selected_09l_points.parquet')
    geometry = pd.read_csv(PROJECT / 'airports/LFPG/reports/03_runway_geometry.csv')
    settings = json.loads((PROJECT / 'configs/lfpg_09l_training.json').read_text(encoding='utf-8'))
    examples, audit = examples_for(selected, points, geometry, settings)
    examples['day'] = date
    examples.to_parquet(folder / 'examples.parquet', index=False)
    audit.to_csv(folder / 'examples_audit.csv', index=False)
    result = {'date': date, 'qualified_09l': examples.event_id.nunique(),
              'examples': len(examples),
              'enough_approaches': examples.event_id.nunique() >= 10 and len(examples) >= 300}
    print('PREPARED_DAY', json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    for date in sys.argv[1:]:
        prepare(date)
