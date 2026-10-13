"""Геометрический поиск заходов LFPG и диагностика высотного покрытия"""
import hashlib
import json
import numpy as np
import pandas as pd

def local_xy(lat, lon, lat0, lon0):
    # Локальные метры по радиусам кривизны эллипсоида WGS84
    phi = np.deg2rad(lat0)
    a, e2 = 6378137.0, 6.6943799901413165e-3
    n = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    m = a * (1 - e2) / (1 - e2 * np.sin(phi) ** 2) ** 1.5
    return (np.deg2rad(np.asarray(lon) - lon0) * n * np.cos(phi),
            np.deg2rad(np.asarray(lat) - lat0) * m)

def runway_geometry(runways):
    rows = []
    for item in runways.itertuples(index=False):
        for end, opposite in [('le', 'he'), ('he', 'le')]:
            lat, lon = getattr(item, end + '_latitude_deg'), getattr(item, end + '_longitude_deg')
            lat2, lon2 = getattr(item, opposite + '_latitude_deg'), getattr(item, opposite + '_longitude_deg')
            east, north = local_xy(lat2, lon2, lat, lon)
            length = float(np.hypot(east, north))
            ux, uy = float(east / length), float(north / length)
            displacement = getattr(item, end + '_displaced_threshold_ft')
            displacement = 0.0 if pd.isna(displacement) else float(displacement * .3048)
            # Учитываем смещение порога внутрь полосы по данным справочника
            rows.append({'runway': getattr(item, end + '_ident'), 'lat0': lat, 'lon0': lon,
                         'ux': ux, 'uy': uy, 'length_m': length,
                         'displacement_m': displacement,
                         'bearing_deg': float(np.rad2deg(np.arctan2(ux, uy)) % 360),
                         'end_elevation_ft': getattr(item, end + '_elevation_ft')})
    return pd.DataFrame(rows)

def attach_qnh(points, weather_path, airport_elevation_ft, delay_minutes=15):
    metar = pd.read_csv(weather_path)
    metar['observation_utc'] = pd.to_datetime(metar.valid, utc=True).dt.floor('s').astype('datetime64[s, UTC]')
    # Используем только наблюдаемую часть METAR до трендов и замечаний
    body = metar.metar.str.split(r'\s(?:TEMPO|BECMG|RMK)\s', n=1, regex=True).str[0]
    metar['qnh_hpa'] = pd.to_numeric(body.str.extract(r'\bQ(\d{4})\b', expand=False), errors='coerce')
    metar['correction'] = body.str.contains(r'\bCOR\b', regex=True)
    eligible = metar.loc[~metar.correction & metar.qnh_hpa.notna()].copy()
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

def find_events(points, geometry, corridor_m=200, heading_tolerance_deg=15, gap_s=60):
    clean = points.loc[~points.stale_position & ~points.geometric_altitude
                       & points.height_proxy_ft.notna() & points.speed_knots.ge(80)].copy()
    events, selected = [], []
    for runway in geometry.itertuples(index=False):
        east, north = local_xy(clean.lat, clean.lon, runway.lat0, runway.lon0)
        along = east * runway.ux + north * runway.uy - runway.displacement_m
        lateral = east * runway.uy - north * runway.ux
        heading_error = (clean.track_deg - runway.bearing_deg + 180) % 360 - 180
        # Коридор задает поиск кандидатов и не является допуском стабилизированного захода
        mask = (np.abs(lateral) <= corridor_m) & heading_error.abs().le(heading_tolerance_deg)
        mask &= clean.height_proxy_ft.between(-200, 6000)
        matched = clean.loc[mask].copy()
        matched['distance_to_threshold_m'] = -along[mask]
        matched['lateral_m'] = lateral[mask]
        for direction in ['arrival', 'departure']:
            if direction == 'arrival':
                group_points = matched.loc[matched.distance_to_threshold_m.between(0, 15000)].copy()
            else:
                group_points = matched.loc[matched.distance_to_threshold_m.between(-runway.length_m - 15000, 0)].copy()
            group_points = group_points.sort_values(['trace_id', 'time'])
            group_points['part'] = ((group_points.trace_id != group_points.trace_id.shift())
                | group_points.groupby('trace_id').time.diff().gt(gap_s)).cumsum()
            for _, group in group_points.groupby('part'):
                if len(group) < 5 or group.time.iloc[-1] - group.time.iloc[0] < 20:
                    continue
                delta_height = group.height_proxy_ft.iloc[-1] - group.height_proxy_ft.iloc[0]
                progress = group.distance_to_threshold_m.iloc[0] - group.distance_to_threshold_m.iloc[-1]
                near = group.distance_to_threshold_m.between(0, 1500).any() if direction == 'arrival' else group.distance_to_threshold_m.between(-runway.length_m, 0).any()
                if not near or progress < 1000 or group.height_proxy_ft.min() > 1000:
                    continue
                median_rate = group.vertical_rate_fpm.median()
                if direction == 'arrival' and not (delta_height < -200 and median_rate < -100):
                    continue
                if direction == 'departure' and not (delta_height > 200 and median_rate > 100):
                    continue
                key = f'{group.trace_id.iloc[0]}:{runway.runway}:{direction}:{group.time.iloc[0]:.2f}'
                event_id = hashlib.sha256(key.encode()).hexdigest()[:16]
                previous = group.height_proxy_ft.shift()
                bracket = previous.gt(300) & group.height_proxy_ft.le(300)
                first_cross = group.loc[bracket].head(1)
                cross_index = first_cross.index[0] if len(first_cross) else None
                # Покрытие оцениваем по паре наблюдений без интерполяции признаков
                cross_gap = float(group.time.diff().loc[cross_index]) if cross_index is not None else np.nan
                precontrol = group.loc[group.time.le(first_cross.time.iloc[0])] if len(first_cross) else group
                events.append({'event_id': event_id, 'trace_id': group.trace_id.iloc[0],
                    'icao24': group.icao24.iloc[0], 'runway': runway.runway, 'direction': direction,
                    'start_time': float(group.time.iloc[0]), 'end_time': float(group.time.iloc[-1]),
                    'rows': len(group), 'span_s': float(group.time.iloc[-1] - group.time.iloc[0]),
                    'height_min_ft': float(group.height_proxy_ft.min()),
                    'height_start_ft': float(group.height_proxy_ft.iloc[0]),
                    'height_end_ft': float(group.height_proxy_ft.iloc[-1]),
                    'distance_end_m': float(group.distance_to_threshold_m.iloc[-1]),
                    'median_abs_lateral_m': float(group.lateral_m.abs().median()),
                    'median_vertical_rate_fpm': float(median_rate),
                    'max_gap_s': float(group.time.diff().max()),
                    'cross300_observed': bool(len(first_cross)), 'cross300_gap_s': cross_gap,
                    'precontrol_span_s': float(precontrol.time.iloc[-1] - precontrol.time.iloc[0]),
                    'precontrol_max_gap_s': float(precontrol.time.diff().max()),
                    'qnh_age_max_s': float(group.qnh_age_s.max())})
                group = group.copy()
                group['event_id'], group['runway'], group['direction'] = event_id, runway.runway, direction
                selected.append(group)
    table = pd.DataFrame(events, columns=['event_id', 'trace_id', 'icao24', 'runway',
        'direction', 'start_time', 'end_time', 'rows', 'span_s', 'height_min_ft',
        'height_start_ft', 'height_end_ft', 'distance_end_m', 'median_abs_lateral_m',
        'median_vertical_rate_fpm', 'max_gap_s', 'cross300_observed', 'cross300_gap_s',
        'precontrol_span_s', 'precontrol_max_gap_s', 'qnh_age_max_s'])
    table['parallel_ambiguity'] = False
    # Сохраняем конкурирующие назначения, если окна одного трека существенно перекрываются
    for _, group in table.groupby(['trace_id', 'direction']):
        for index, row in group.iterrows():
            overlap = np.minimum(row.end_time, group.end_time) - np.maximum(row.start_time, group.start_time)
            other = group.index != index
            ambiguous = other & (group.runway != row.runway) & (overlap >= .5 * np.minimum(row.span_s, group.span_s))
            if ambiguous.any():
                table.loc[index, 'parallel_ambiguity'] = True
    table['coverage300_candidate'] = (table.direction.eq('arrival') & table.cross300_observed
        & table.cross300_gap_s.le(15) & table.precontrol_max_gap_s.le(15)
        & table.precontrol_span_s.ge(60) & ~table.parallel_ambiguity)
    event_points = (pd.concat(selected, ignore_index=True) if selected
                    else points.iloc[:0].assign(event_id='', runway='', direction=''))
    return table, event_points
