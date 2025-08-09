# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Testing the skais_mapper.configure module."""

import pytest
import logging
from unittest.mock import MagicMock
import skais_mapper.configure as configure


@pytest.fixture
def mock_cfg():
    """Mock configuration object for testing."""
    class DummyCfg(dict):
        exclude_git_state = False
        include_git_diff = False
        verbose = False
        save_configs = False
        output = "output.yaml"
    return DummyCfg()


@pytest.fixture
def mock_opt():
    """Mock optimization object for testing."""
    class DummyOpt:
        run_id = "test_run"
        verbose = False
        save_configs = False
        output = "output.yaml"
    return DummyOpt()


@pytest.fixture
def patch_dependencies(monkeypatch, mock_opt):
    """Patch dependencies for the configure module."""
    # Patch HydraConfig.get()
    hydra_config = MagicMock()
    hydra_config.runtime.output_dir = "mock_output_dir"
    hydra_config.output_subdir = "mock_subdir"
    monkeypatch.setattr(configure.HydraConfig, "get", lambda: hydra_config)

    # Patch instantiate
    monkeypatch.setattr(configure, "instantiate", lambda cfg: mock_opt)
    # Patch compress_encode
    monkeypatch.setattr(configure, "compress_encode", lambda x: f"compressed({x})")
    # Patch skais_mapper.GIT_STATE and GIT_DIFF
    monkeypatch.setattr(configure.skais_mapper, "GIT_STATE", "FAKE_GIT_STATE")
    monkeypatch.setattr(configure.skais_mapper, "GIT_DIFF", ["diff1", "diff2"])
    # Patch OmegaConf.to_yaml
    monkeypatch.setattr(configure.OmegaConf, "to_yaml", lambda cfg: "yaml_output")
    # Patch Path.read_bytes and write_bytes
    mock_path = MagicMock()
    mock_path.read_bytes.return_value = b"cfg"
    monkeypatch.setattr(configure, "Path", lambda x=None: mock_path)


@pytest.mark.usefixtures("patch_dependencies")
def test_configure_logs_git_state_and_diff(monkeypatch, caplog, mock_cfg, mock_opt):
    """Test that create function logs git state and diff correctly."""
    mock_cfg.exclude_git_state = False
    mock_cfg.include_git_diff = True
    mock_cfg.verbose = False
    mock_cfg.save_configs = False

    with caplog.at_level(logging.INFO):
        configure.create(mock_cfg)
        assert "Git state: FAKE_GIT_STATE" in caplog.text
        assert "Git diff: compressed(diff1)" in caplog.text
        assert "Git diff: compressed(diff2)" in caplog.text
        assert "Job id: test_run" in caplog.text
        assert "Output directory: mock_output_dir" in caplog.text


@pytest.mark.usefixtures("patch_dependencies")
def test_configure_exclude_git_state(monkeypatch, caplog, mock_cfg, mock_opt):
    """Test that create function does not log git state when excluded."""
    mock_cfg.exclude_git_state = True
    mock_cfg.include_git_diff = False
    mock_cfg.verbose = False
    mock_cfg.save_configs = False

    with caplog.at_level(logging.INFO):
        configure.create(mock_cfg)
        assert "Git state:" not in caplog.text
        assert "Git diff:" not in caplog.text


@pytest.mark.usefixtures("patch_dependencies")
def test_create_prints_config(monkeypatch, mock_cfg, mock_opt, capsys):
    """Test that create function prints configuration when verbose."""
    mock_cfg.verbose = True
    mock_opt.verbose = True
    mock_cfg.save_configs = False

    configure.create(mock_cfg)
    captured = capsys.readouterr()
    assert "Configuration:" in captured.out
    assert "yaml_output" in captured.out


@pytest.mark.usefixtures("patch_dependencies")
def test_create_saves_config(monkeypatch, mock_cfg, mock_opt):
    """Test that create function saves configuration to file."""
    mock_cfg.save_configs = True
    mock_opt.save_configs = True
    mock_opt.output = "output.yaml"

    mock_path = MagicMock()
    mock_path.suffix = ".yaml"
    mock_path.read_bytes.return_value = b"cfg"
    monkeypatch.setattr(configure, "Path", lambda x=None: mock_path)

    configure.create(mock_cfg)
    # Should attempt to write config bytes
    assert mock_path.write_bytes.called


@pytest.mark.usefixtures("patch_dependencies")
def test_create_saves_config_non_yaml(monkeypatch, mock_cfg, mock_opt):
    """Test that create function saves configuration to file."""
    mock_cfg.save_configs = True
    mock_opt.save_configs = True
    mock_opt.output = "output.txt"

    mock_path = MagicMock()
    mock_path.suffix = ".txt"
    mock_path.read_bytes.return_value = b"cfg"
    monkeypatch.setattr(configure, "Path", lambda x=None: mock_path)

    configure.create(mock_cfg)
    # Should attempt to write config bytes
    assert mock_path.write_bytes.called
