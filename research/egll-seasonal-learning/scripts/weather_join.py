"""QNH EGLL с условной задержкой доступности, включая архивные COR"""
import numpy as np
import pandas as pd

def attach_qnh(points, weather_path, airport_elevation_ft, delay_minutes=15):
    metar = pd.read_csv(weather_path)
    metar['observation_utc'] = pd.to_datetime(metar.valid, utc=True).dt.floor('s').astype('datetime64[s, UTC]')
    # Используем только наблюдаемую часть METAR до трендов и замечаний
    body = metar.metar.str.split(r'\s(?:TEMPO|BECMG|RMK)\s', n=1, regex=True).str[0]
    metar['qnh_hpa'] = pd.to_numeric(body.str.extract(r'\bQ(\d{4})\b', expand=False), errors='coerce')
    metar['correction'] = body.str.contains(r'\bCOR\b', regex=True)
    eligible = metar.loc[metar.qnh_hpa.notna()].copy()
    conflicts = eligible.groupby('observation_utc').qnh_hpa.nunique()
    assert not conflicts.gt(1).any(), 'Неоднозначный QNH в одну отметку времени'
    eligible = eligible.sort_values('observation_utc').drop_duplicates('observation_utc')
    eligible['assumed_available_utc'] = (eligible.observation_utc
        + pd.Timedelta(minutes=delay_minutes)).astype('datetime64[s, UTC]')
    frame = points.copy()
    # Отбрасываем доли секунды для причинного соединения с METAR
    frame['utc'] = pd.to_datetime(frame.time, unit='s', utc=True).dt.floor('s').astype('datetime64[s, UTC]')
    frame = pd.merge_asof(frame.sort_values('utc'),
        eligible[['observation_utc', 'assumed_available_utc', 'qnh_hpa']].sort_values('assumed_available_utc'),
        left_on='utc', right_on='assumed_available_utc', direction='backward')
    frame['qnh_age_s'] = (frame.utc - frame.observation_utc).dt.total_seconds()
    frame.loc[frame.qnh_age_s.gt(3600), 'qnh_hpa'] = np.nan
    assert (frame.assumed_available_utc.dropna() <= frame.loc[frame.assumed_available_utc.notna(), 'utc']).all()
    # Перенастройка барометрической высоты по модели стандартной атмосферы
    scale = 288.15 / .0065
    exponent = 287.05287 * .0065 / 9.80665
    factor = (1013.25 / frame.qnh_hpa) ** exponent
    frame['qnh_altitude_ft'] = (scale * (1 - factor) + factor * frame.altitude_ft * .3048) / .3048
    frame['height_proxy_ft'] = frame.qnh_altitude_ft - airport_elevation_ft
    frame['raw_minus_airport_ft'] = frame.altitude_ft - airport_elevation_ft
    return frame, metar
