"""Проверить выполненные ноутбуки, разделение дней и сохраненные прогнозы"""
from pathlib import Path
import joblib
import nbformat
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]


def run():
    days = pd.read_csv(BASE / 'reports/csv/selected_days.csv')
    assert days.day.is_unique and len(days) == 15
    assert days.groupby('season').size().eq(5).all()
    assert days.groupby(['season', 'split']).size().unstack().to_dict() == {
        'test': {'autumn': 1, 'spring': 1, 'summer': 1},
        'train': {'autumn': 3, 'spring': 3, 'summer': 3},
        'validation': {'autumn': 1, 'spring': 1, 'summer': 1}}
    weather = pd.read_csv(BASE / 'data/weather_candidates_2025.csv')
    assert weather.set_index('date').loc[days.day, 'fair_fraction'].eq(1).all()
    test = pd.concat([pd.read_parquet(BASE / 'data' / row.day / 'examples.parquet')
                      .assign(season=row.season) for row in days.loc[days.split.eq('test')].itertuples()],
                     ignore_index=True)
    assert test.groupby('event_id').season.nunique().eq(1).all()
    assert not test.duplicated(['event_id', 'origin_time', 'target_time']).any()
    assert test.origin_metar_available_time.le(test.origin_time).all()
    assert test.target_time.lt(test.boundary_time).all()
    assert test.horizon_s.between(27, 33).all()
    assert test.airport.eq('EGLL').all() and test.airport_elevation_ft.eq(83).all()
    source = pd.concat([pd.read_parquet(BASE / 'data' / row.day / 'selected_09l_points.parquet')
                        for row in days.loc[days.split.eq('test')].itertuples()], ignore_index=True)
    source = source.loc[source.event_id.isin(test.event_id)].set_index(['event_id', 'time'])
    assert source.index.is_unique
    origins = source.loc[pd.MultiIndex.from_arrays([test.event_id, test.origin_time])]
    targets = source.loc[pd.MultiIndex.from_arrays([test.event_id, test.target_time])]
    factor = (1013.25 / origins.qnh_hpa.to_numpy()) ** (287.05287 * .0065 / 9.80665)
    expected = ((288.15 / .0065) * (1 - factor)
                + factor * targets.altitude_ft.to_numpy() * .3048) / .3048 - 83
    assert np.allclose(expected, test.target_height_ft, rtol=0, atol=1e-8)
    assert np.allclose(origins.height_proxy_ft, test.height_ft, rtol=0, atol=1e-8)
    saved = joblib.load(BASE / 'artifacts/models.joblib')
    final = pd.read_csv(BASE / 'reports/csv/final_table.csv', index_col=0)
    for name in final.index:
        predicted = test.baseline_ft.to_numpy()
        if name != 'Constant vertical rate':
            predicted = predicted + saved['models']['pooled', name].predict(test[saved['features']])
        error = pd.DataFrame({'event_id': test.event_id, 'season': test.season,
                              'absolute_error': np.abs(predicted - test.target_height_ft.to_numpy())})
        for season in final.columns:
            subset = error if season == 'all' else error.loc[error.season.eq(season)]
            mae = subset.groupby('event_id').absolute_error.mean().mean()
            assert np.isclose(mae, final.loc[name, season], rtol=0, atol=1e-9)
    for path in (BASE / 'notebooks').glob('*.ipynb'):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                assert cell.execution_count is not None
                assert not any(o.output_type == 'error' for o in cell.outputs)
    print('VERIFIED: 15 fair days, day splits, observation pairs, saved predictions, executed notebooks')


if __name__ == '__main__':
    run()
