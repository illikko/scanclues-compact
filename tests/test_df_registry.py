import pandas as pd

from core.df_registry import DFState, get_df, init_df_registry, set_df, sync_registry_from_aliases


def test_registry_roundtrip():
    ss = {}
    init_df_registry(session_state=ss)
    df = pd.DataFrame({"a": [1, 2]})
    set_df(DFState.RAW, df, session_state=ss, step_name="test")

    out = get_df(DFState.RAW, session_state=ss)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (2, 1)
    assert "df_raw" in ss  # alias legacy conservé


def test_registry_hydrates_from_legacy_alias():
    ss = {"df_ready": pd.DataFrame({"x": [10]})}
    init_df_registry(session_state=ss)
    sync_registry_from_aliases(session_state=ss)

    out = get_df(DFState.READY, session_state=ss)
    assert isinstance(out, pd.DataFrame)
    assert out.iloc[0, 0] == 10
