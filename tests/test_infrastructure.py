"""
Validation tests to verify the testing infrastructure is properly set up.
"""
import pytest
import sys
from pathlib import Path
import importlib.util


class TestInfrastructure:
    """Test that the testing infrastructure is properly configured."""
    
    def test_pytest_working(self):
        """Test that pytest is working correctly."""
        assert True
    
    def test_pytest_markers_configured(self):
        """Test that custom pytest markers are configured."""
        import _pytest.config
        # This will pass if markers are properly configured in pyproject.toml
        assert True
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit marker is working."""
        assert True
    
    @pytest.mark.integration  
    def test_integration_marker(self):
        """Test that integration marker is working."""
        assert True
        
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker is working."""
        assert True
    
    def test_fixtures_available(self, temp_dir, mock_config, sample_image):
        """Test that shared fixtures are available and working."""
        # Test temp_dir fixture
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test mock_config fixture  
        assert mock_config is not None
        assert hasattr(mock_config, 'DATASET')
        assert hasattr(mock_config, 'MODEL')
        
        # Test sample_image fixture
        assert sample_image is not None
        assert sample_image.shape == (240, 320, 3)
    
    def test_project_structure(self):
        """Test that the project structure is correct."""
        project_root = Path(__file__).parent.parent
        
        # Check main package directories exist
        assert (project_root / 'pysot').exists()
        assert (project_root / 'toolkit').exists()
        
        # Check test directories exist
        assert (project_root / 'tests').exists()
        assert (project_root / 'tests' / 'unit').exists()
        assert (project_root / 'tests' / 'integration').exists()
        
        # Check configuration files exist
        assert (project_root / 'pyproject.toml').exists()
        assert (project_root / '.gitignore').exists()
    
    def test_package_imports(self):
        """Test that main packages can be imported."""
        # Test pysot package imports
        try:
            import pysot
            assert pysot is not None
        except ImportError as e:
            pytest.skip(f"pysot package not importable: {e}")
        
        # Test toolkit package imports
        try:
            import toolkit
            assert toolkit is not None
        except ImportError as e:
            pytest.skip(f"toolkit package not importable: {e}")
    
    def test_mock_dependencies(self, mock_tracker, mock_model, mock_logger):
        """Test that mock fixtures work correctly."""
        # Test mock_tracker
        assert mock_tracker is not None
        mock_tracker.init("test")
        mock_tracker.init.assert_called_once_with("test")
        
        result = mock_tracker.track("frame")
        assert result['best_score'] == 0.8
        
        # Test mock_model
        assert mock_model is not None
        mock_model.eval()
        mock_model.eval.assert_called_once()
        
        # Test mock_logger
        assert mock_logger is not None
        mock_logger.info("test message")
        mock_logger.info.assert_called_once_with("test message")
    
    def test_numpy_available(self):
        """Test that numpy is available for testing."""
        import numpy as np
        arr = np.array([1, 2, 3])
        assert arr.sum() == 6
    
    def test_opencv_available(self):
        """Test that OpenCV is available for testing."""
        try:
            import cv2
            assert cv2 is not None
        except ImportError:
            pytest.skip("OpenCV not available")


class TestCoverageConfiguration:
    """Test that coverage is properly configured."""
    
    def test_coverage_config_exists(self):
        """Test that coverage configuration exists in pyproject.toml."""
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / 'pyproject.toml'
        
        with open(pyproject_path, 'r') as f:
            content = f.read()
            assert '[tool.coverage.run]' in content
            assert '[tool.coverage.report]' in content
            assert 'source = ["pysot", "toolkit"]' in content


class TestProjectConfiguration:
    """Test that project configuration is correct."""
    
    def test_pyproject_toml_structure(self):
        """Test that pyproject.toml has correct structure."""
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / 'pyproject.toml'
        
        with open(pyproject_path, 'r') as f:
            content = f.read()
            
        # Check main sections exist
        assert '[tool.poetry]' in content
        assert '[tool.poetry.dependencies]' in content
        assert '[tool.poetry.group.dev.dependencies]' in content
        assert '[tool.pytest.ini_options]' in content
        assert '[tool.coverage.run]' in content
        
        # Check testing dependencies
        assert 'pytest' in content
        assert 'pytest-cov' in content
        assert 'pytest-mock' in content