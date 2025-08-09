# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Testing the skais_mapper.generate module."""

import pytest
from unittest.mock import MagicMock
import numpy as np
from astropy import units as au
import skais_mapper.generate as generate


@pytest.fixture
def mock_TNGGalaxy():
    """Create a minimal mock of TNGGalaxy for testing."""
    # Minimal mock with required attributes and methods
    mock_tng_instance = MagicMock(
        N_particles_type=[100_000] * 5,
        halo_index=10,
        particle_type="dm",
    )
    mock_tng = MagicMock(name="TNGGalaxy", return_value=mock_tng_instance)
    return mock_tng


@pytest.fixture
def map_TNG_sample_default_kwargs(tmp_path):
    """Provide default keyword arguments for testing."""
    return dict(
        gid=1,
        group="gas",
        projected_unit=None,
        cmap=None,
        hdf5_file=tmp_path / "test.hdf5",
        hdf5_save=True,
        npy_save=False,
        png_save=False,
        subdir_save=False,
        grid_size=32,
        fh=3,
        rot=None,
        xaxis=0,
        yaxis=1,
        periodic=False,
        rng_seed=42,
        flag_lim=1000,
        flag_N=64,
        dry_run=False,
        verbose=False,
    )


@pytest.fixture
def map_TNG_galaxies_default_kwargs(tmp_path):
    """Provide default keyword arguments for testing map_TNG_galaxies."""
    return dict(
        snapshots=[99],
        gids=[1, 2, 3],
        groups=["gas"],
        output=None,
        src_dir=tmp_path,
        sim_type="illustris/tng50-1",
        csv_file=tmp_path / "test.csv",
        part_max=None,
        part_min=20_000,
        retries=None,
        subfind_limit=15_000,
        grid_size=512,
        rotations=None,
        random_rotations=True,
        rng_seed=42,
        dry_run=True,
        verbose=True,
    )


@pytest.fixture
def illustris_dir(tmp_path):
    """Patch the illustris / tng50-1 directory."""
    src_dir = tmp_path
    illustris_dir = src_dir / "illustris" / "tng50-1"
    illustris_dir.mkdir(parents=True, exist_ok=True)
    (illustris_dir / "snapshots" / "099").mkdir(parents=True, exist_ok=True)
    (illustris_dir / "groupcats" / "099").mkdir(parents=True, exist_ok=True)
    (illustris_dir / "offsets" / "099").mkdir(parents=True, exist_ok=True)
    return src_dir


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    """Patch dependencies for the generate module."""
    monkeypatch.setattr(
        generate,
        "Img2H5Buffer",
        MagicMock(return_value=MagicMock(index=1, cosmology=MagicMock())),
    )
    monkeypatch.setattr(generate, "voronoi_NGP_2D", MagicMock(return_value=(np.zeros((4, 4)), {})))
    monkeypatch.setattr(generate, "plot_image", MagicMock())
    monkeypatch.setattr(generate, "get_run_id", MagicMock(return_value="runid"))
    monkeypatch.setattr(generate, "compress_encode", MagicMock(return_value="gz"))


@pytest.fixture
def mock_cfg():
    """Mock configuration object for testing."""
    class DummyCfg(dict):
        exclude_git_state = False
        include_git_diff = False
        verbose = True
        save_configs = False
        output = "output.yaml"
    return DummyCfg()


@pytest.fixture
def mock_opt(tmp_path):
    """Mock optimization object for testing."""
    dummy_opt = {}
    dummy_opt["run_id"] = "test_run"
    dummy_opt["snapshots"] = [99]
    dummy_opt["num_samples"] = (3,)
    dummy_opt["groups"] = ["gas"]
    dummy_opt["simulation_type"] = "illustris/tng50-1"
    dummy_opt["output"] = tmp_path / "test.hdf5"
    dummy_opt["source"] = tmp_path
    dummy_opt["part_min"] = None
    dummy_opt["retries"] = None
    dummy_opt["subfind_limit"] = None
    dummy_opt["random_seed"] = 42
    dummy_opt["grid_size"] = 512
    dummy_opt["dry_run"] = False
    dummy_opt["verbose"] = True
    return dummy_opt


@pytest.mark.parametrize(
    "kwargs,mock_unit",
    [
        [{"group": "gas"}, (au.Msun * au.kpc ** (-2), au.Msun * au.kpc ** (-2))],
        [
            {"group": "gas", "projected_unit": au.Msun * au.kpc ** (-2)},
            (au.Msun * au.kpc ** (-2), au.Msun * au.kpc ** (-2)),
        ],
        [{"group": "hi"}, (au.Msun * au.kpc ** (-2), au.Msun * au.kpc ** (-2))],
        [
            {"group": "hi", "projected_unit": au.Msun * au.kpc ** (-2)},
            (au.Msun * au.kpc ** (-2), au.Msun * au.kpc ** (-2)),
        ],
        [{"group": "hi/21cm"}, (au.mK, au.Msun * au.kpc ** (-2))],
        [
            {"group": "hi/21cm", "projected_unit": au.mK},
            (au.mK, au.Msun * au.kpc ** (-2)),
        ],
        [{"group": "temp"}, (au.K * au.Msun * au.kpc ** (-2), au.Msun * au.kpc ** (-2))],
        [
            {"group": "temp", "projected_unit": au.K},
            (au.K * au.Msun * au.kpc ** (-2), au.Msun * au.kpc ** (-2)),
        ],
        [{"group": "bfield"}, (au.G * au.Msun * au.kpc ** (-2), au.Msun * au.kpc ** (-2))],
        [
            {"group": "bfield", "projected_unit": au.G},
            (au.G * au.Msun * au.kpc ** (-2), au.Msun * au.kpc ** (-2)),
        ],
        [{"group": "star"}, (au.Msun * au.kpc ** (-2), au.Msun * au.kpc ** (-2))],
        [
            {"group": "star", "projected_unit": au.Msun * au.kpc ** (-2)},
            (au.Msun * au.kpc ** (-2), au.Msun * au.kpc ** (-2)),
        ],
        [
            {"group": "dm", "flag_lim": int(1e8), "subdir_save": True, "dry_run": True},
            (au.Msun * au.kpc ** (-2), au.Msun * au.kpc ** (-2)),
        ],
        [
            {
                "group": "dm",
                "projected_unit": au.Msun * au.kpc ** (-2),
                "subdir_save": True,
                "verbose": True,
            },
            (
                au.Msun * au.kpc ** (-2),
                au.Msun * au.kpc ** (-2),
            ),
        ],
    ],
)
def test_map_TNG_sample_group_types(
    kwargs, mock_unit, mock_TNGGalaxy, map_TNG_sample_default_kwargs
):
    """Test basic functionality of map_TNG_sample with various groups."""
    # Basic usage, hdf5_save enabled (default)
    kwargs = {**map_TNG_sample_default_kwargs, **kwargs}
    mock_TNGGalaxy.generate_map.side_effect = [
        [np.ones((512, 512)) * mock_unit[0], np.array([-1, 1, -1, 1]) * au.kpc, int(1e5)],
        [np.ones((512, 512)) * mock_unit[1], np.array([-1, 1, -1, 1]) * au.kpc, int(1e5)],
    ]
    mock_TNGGalaxy.snapshot = 99
    mock_TNGGalaxy.halo_index = 1001
    generate.map_TNG_sample(
        mock_TNGGalaxy,
        post_hook=MagicMock(side_effect=(lambda x, y: x)) if kwargs["group"] != "dm" else None,
        **kwargs,
    )
    generate.plot_image.assert_called()
    generate.Img2H5Buffer.assert_called()


def test_map_TNG_sample_group_type_invalid(mock_TNGGalaxy, map_TNG_sample_default_kwargs):
    """Test basic functionality of map_TNG_sample with various groups."""
    # Basic usage, hdf5_save enabled (default)
    mock_unit = (au.Msun * au.kpc ** (-2), au.Msun * au.kpc ** (-2))
    kwargs = {**map_TNG_sample_default_kwargs, **{"group": "unknown_group"}}
    mock_TNGGalaxy.generate_map.side_effect = [
        [np.ones((512, 512)) * mock_unit[0], np.array([-1, 1, -1, 1]) * au.kpc, int(1e5)],
        [np.ones((512, 512)) * mock_unit[1], np.array([-1, 1, -1, 1]) * au.kpc, int(1e5)],
    ]
    mock_TNGGalaxy.snapshot = 99
    mock_TNGGalaxy.halo_index = 1001
    with pytest.raises(ValueError):
        generate.map_TNG_sample(
            mock_TNGGalaxy, post_hook=MagicMock(side_effect=(lambda x, y: x)), **kwargs
        )


@pytest.mark.parametrize(
    "kwargs,mock_unit",
    [
        [
            {"group": "gas", "hdf5_save": False, "npy_save": True},
            (au.Msun * au.kpc ** (-2), au.Msun * au.kpc ** (-2)),
        ],
        [
            {
                "group": "gas",
                "hdf5_save": False,
                "npy_save": True,
                "subdir_save": True,
                "verbose": True,
            },
            (au.Msun * au.kpc ** (-2), au.Msun * au.kpc ** (-2)),
        ],
        [
            {
                "group": "dm",
                "hdf5_save": False,
                "npy_save": True,
                "subdir_save": True,
                "dry_run": True,
                "verbose": False,
            },
            (au.Msun * au.kpc ** (-2), au.Msun * au.kpc ** (-2)),
        ],
    ],
)
def test_map_TNG_sample_save_npy(
    kwargs, mock_unit, mock_TNGGalaxy, map_TNG_sample_default_kwargs, tmp_path
):
    """Test basic functionality of map_TNG_sample to save NPY."""
    # Basic usage, hdf5_save enabled (default)
    kwargs = {**map_TNG_sample_default_kwargs, **kwargs}
    mock_TNGGalaxy.generate_map.side_effect = [
        [np.ones((512, 512)) * mock_unit[0], np.array([-1, 1, -1, 1]) * au.kpc, int(1e5)],
        [np.ones((512, 512)) * mock_unit[1], np.array([-1, 1, -1, 1]) * au.kpc, int(1e5)],
    ]
    mock_TNGGalaxy.snapshot = 99
    mock_TNGGalaxy.halo_index = 1001
    generate.map_TNG_sample(
        mock_TNGGalaxy, post_hook=MagicMock(side_effect=(lambda x, y: x)), **kwargs
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"groups": None, "random_rotations": True},
        {"random_rotations": False},
        {"groups": None, "output": "test_output.hdf5", "verbose": True},
        {"gids": [1, 2, 3], "retries": 5, "part_max": 1_001, "part_min": 1_000, "verbose": True},
        {
            "gids": [1, 2, 3],
            "part_max": 1_001,
            "part_min": 1_000,
            "subfind_limit": 2,
            "retries": 10,
            "verbose": False,
        },
        {
            "gids": [1],
            "part_max": 1_001,
            "part_min": 1_000,
            "retries": None,
            "subfind_limit": None,
            "verbose": True,
        },
        {
            "gids": [1],
            "random_rotations": False,
            "part_max": 1_001,
            "part_min": 1_000,
            "retries": None,
            "subfind_limit": None,
            "verbose": True,
        },
        {"gids": [1, 2, 3], "subfind_limit": 10_000, "grid_size": 256},
        {"gids": [1, 2, 3], "subfind_limit": 10_000, "grid_size": 256, "verbose": False},
        {"gids": [1, 2, 3], "csv_file": "test.csv", "verbose": True},
        {"gids": [1, 2, 3], "csv_file": None, "verbose": True},
        {"gids": [1], "output": "test_{}.hdf5", "verbose": True},
        {"gids": [1], "output": "test_{}_{}.hdf5", "verbose": True},
        
    ],
)
def test_map_TNG_galaxies(
    kwargs, monkeypatch, illustris_dir, mock_TNGGalaxy, map_TNG_galaxies_default_kwargs
):
    """Test basic functionality of map_TNG_galaxies."""
    kwargs = {**map_TNG_galaxies_default_kwargs, **kwargs}
    kwargs["src_dir"] = illustris_dir
    monkeypatch.setattr(generate, "map_TNG_sample", MagicMock())
    monkeypatch.setattr(generate, "TNGGalaxy", mock_TNGGalaxy)
    if isinstance(kwargs["csv_file"], str):
        kwargs["csv_file"] = illustris_dir / kwargs["csv_file"]
    generate.map_TNG_galaxies(**kwargs)



def test_run_default(monkeypatch, mock_cfg, mock_opt):
    """Test the run function with default arguments."""
    hydra_config = MagicMock()
    hydra_config.runtime.output_dir = "mock_output_dir"
    hydra_config.output_subdir = "mock_subdir"
    monkeypatch.setattr(generate.HydraConfig, "get", lambda: hydra_config)
    monkeypatch.setattr(generate, "compress_encode", lambda x: f"compressed({x})")
    # Patch skais_mapper.GIT_STATE and GIT_DIFF
    monkeypatch.setattr(generate.skais_mapper, "GIT_STATE", "FAKE_GIT_STATE")
    monkeypatch.setattr(generate.skais_mapper, "GIT_DIFF", ["diff1", "diff2"])
    # Patch OmegaConf.to_yaml
    monkeypatch.setattr(generate.OmegaConf, "to_yaml", lambda cfg: "yaml_output")
    monkeypatch.setattr(generate, "instantiate", lambda cfg, **kwargs: mock_opt)
    monkeypatch.setattr(generate, "map_TNG_galaxies", MagicMock())
    mock_cfg.exclude_git_state = True
    mock_cfg.include_git_diff = True
    generate.run(mock_cfg)


def test_run_save_configs(monkeypatch, mock_cfg, mock_opt):
    """Test the run function with default arguments."""
    hydra_config = MagicMock()
    hydra_config.runtime.output_dir = "mock_output_dir"
    hydra_config.output_subdir = "mock_subdir"
    monkeypatch.setattr(generate.HydraConfig, "get", lambda: hydra_config)
    monkeypatch.setattr(generate, "compress_encode", lambda x: f"compressed({x})")
    # Patch skais_mapper.GIT_STATE and GIT_DIFF
    monkeypatch.setattr(generate.skais_mapper, "GIT_STATE", "FAKE_GIT_STATE")
    monkeypatch.setattr(generate.skais_mapper, "GIT_DIFF", ["diff1", "diff2"])
    # Patch OmegaConf.to_yaml
    monkeypatch.setattr(generate.OmegaConf, "to_yaml", lambda cfg: "yaml_output")
    monkeypatch.setattr(generate, "instantiate", lambda cfg, **kwargs: mock_opt)
    monkeypatch.setattr(generate, "map_TNG_galaxies", MagicMock())
    mock_path = MagicMock()
    mock_path.read_bytes.return_value = b"cfg"
    monkeypatch.setattr(generate, "Path", lambda x=None: mock_path)
    mock_cfg.save_configs = True
    generate.run(mock_cfg)


def test_run_save_no_sim_type(monkeypatch, mock_cfg, mock_opt):
    """Test the run function with default arguments."""
    hydra_config = MagicMock()
    hydra_config.runtime.output_dir = "mock_output_dir"
    hydra_config.output_subdir = "mock_subdir"
    monkeypatch.setattr(generate.HydraConfig, "get", lambda: hydra_config)
    monkeypatch.setattr(generate, "compress_encode", lambda x: f"compressed({x})")
    # Patch skais_mapper.GIT_STATE and GIT_DIFF
    monkeypatch.setattr(generate.skais_mapper, "GIT_STATE", "FAKE_GIT_STATE")
    monkeypatch.setattr(generate.skais_mapper, "GIT_DIFF", ["diff1", "diff2"])
    # Patch OmegaConf.to_yaml
    monkeypatch.setattr(generate.OmegaConf, "to_yaml", lambda cfg: "yaml_output")
    monkeypatch.setattr(generate, "instantiate", lambda cfg, **kwargs: mock_opt)
    monkeypatch.setattr(generate, "map_TNG_galaxies", MagicMock())
    mock_opt["simulation_type"] = "unknown/sim"
    mock_path = MagicMock()
    mock_path.suffix = ".txt"
    mock_path.read_bytes.return_value = b"cfg"
    monkeypatch.setattr(generate, "Path", lambda x=None: mock_path)
    mock_cfg.verbose = False
    generate.run(mock_cfg)
