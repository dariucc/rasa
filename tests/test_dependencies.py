import importlib.metadata


def test_tensorflow_text_not_installed():
    """tensorflow-text was removed from the dependencies (only needed by the
    unused ConveRTFeaturizer), so it must not be installed on any platform."""
    installed_packages_list = [
        dist.metadata["Name"].lower() for dist in importlib.metadata.distributions()
    ]
    assert "tensorflow-text" not in installed_packages_list
