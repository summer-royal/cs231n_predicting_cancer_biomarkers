"""
No-download smoke tests for encoder registry and feature-file metadata.

This does not load gated UNI/CONCH weights. It checks the parts of the encoder
upgrade that should work in every local environment.
"""

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import infer_feature_dim
from models import list_encoders
from models.encoder import get_encoder
from scripts.extract_features import write_feature_h5


def test_registry() -> None:
    encoders = list_encoders()
    assert encoders["resnet50"] == 2048
    assert encoders["uni"] == 1024
    assert encoders["conch"] == 512
    print(f"[ok] registry {encoders}")


def test_unknown_encoder_error() -> None:
    try:
        get_encoder("not_an_encoder", device="cpu")
    except ValueError as exc:
        assert "Supported encoders" in str(exc)
        print("[ok] unknown encoder gives clear ValueError")
        return
    raise AssertionError("unknown encoder did not raise ValueError")


def test_feature_h5_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "TCGA-XX-0000.h5"
        features = np.ones((3, 7), dtype=np.float32)
        coords = np.array([[0, 0], [256, 0], [0, 256]], dtype=np.int32)
        write_feature_h5(
            out,
            features,
            coords,
            encoder_name="fake",
            feature_dim=7,
            checkpoint_source="synthetic",
            stain_norm=True,
            patch_size=256,
            patch_level=0,
        )

        with h5py.File(out, "r") as f:
            assert f["features"].shape == (3, 7)
            assert f["coords"].shape == (3, 2)
            assert f.attrs["encoder"] == "fake"
            assert int(f.attrs["feature_dim"]) == 7
            assert f.attrs["checkpoint_source"] == "synthetic"
            assert bool(f.attrs["stain_norm"]) is True

        assert infer_feature_dim(tmp) == 7
        print("[ok] feature HDF5 metadata and in_dim inference")


def main() -> int:
    test_registry()
    test_unknown_encoder_error()
    test_feature_h5_metadata()
    print("\nAll encoder smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
