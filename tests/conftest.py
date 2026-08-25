"""
pytest configuration.
Registers the 'smoke' marker so `pytest -m smoke` works cleanly.
"""

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "smoke: end-to-end test that hits the OpenAI API and costs tokens",
    )
