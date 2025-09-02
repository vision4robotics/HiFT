import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import pytest
import numpy as np


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config():
    """Mock configuration object for testing."""
    config = Mock()
    config.DATASET = Mock()
    config.MODEL = Mock()
    config.TRAIN = Mock()
    config.TEST = Mock()
    return config


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)


@pytest.fixture
def sample_bbox():
    """Create a sample bounding box for testing."""
    return [50, 50, 100, 100]  # x, y, width, height


@pytest.fixture
def mock_tracker():
    """Mock tracker object for testing."""
    tracker = Mock()
    tracker.init = Mock()
    tracker.track = Mock(return_value={'bbox': [50, 50, 100, 100], 'best_score': 0.8})
    return tracker


@pytest.fixture
def mock_model():
    """Mock model object for testing."""
    model = Mock()
    model.forward = Mock()
    model.eval = Mock()
    model.train = Mock()
    return model


@pytest.fixture
def sample_dataset_config():
    """Sample dataset configuration for testing."""
    return {
        'name': 'test_dataset',
        'root': '/tmp/test_data',
        'anno': '/tmp/test_data/annotations.json'
    }


@pytest.fixture
def mock_video_data():
    """Mock video data for testing."""
    return {
        'frames': [f'frame_{i:06d}.jpg' for i in range(10)],
        'gt_bbox': [[50 + i, 50 + i, 100, 100] for i in range(10)],
        'video_name': 'test_video'
    }


@pytest.fixture(autouse=True)
def clean_environment():
    """Clean environment variables before each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture
def sample_anchor_config():
    """Sample anchor configuration for testing."""
    return {
        'stride': 8,
        'ratios': [0.33, 0.5, 1, 2, 3],
        'scales': [8],
        'anchor_num': 5
    }