from __future__ import annotations

import warnings

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")

from traffic_flow.models import (  # noqa: E402
    BSplineKANLayer,
    GatedSelfAttentionBlock,
    get_gsa_kan,
)


def test_gsa_kan_round_trip_has_no_unbuilt_state_warning(tmp_path):
    model = get_gsa_kan([12, 64, 64, 1])
    sample = np.zeros((2, 12, 1), dtype=np.float32)
    expected = model.predict(sample, verbose=0)
    path = tmp_path / "gsa_kan.keras"
    model.save(path)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        loaded = tf.keras.models.load_model(
            path,
            custom_objects={
                "BSplineKANLayer": BSplineKANLayer,
                "GatedSelfAttentionBlock": GatedSelfAttentionBlock,
            },
            compile=False,
        )
        actual = loaded.predict(sample, verbose=0)

    assert np.allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert not any("does not have a `build()` method" in str(item.message) for item in captured)

