"""Отбор пригодных заходов EGLL 09L"""

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
