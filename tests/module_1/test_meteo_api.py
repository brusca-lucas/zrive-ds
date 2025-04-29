import pytest
import pandas as pd
import src.module_1.module_1_meteo_api as module_1_meteo_api

# Simulación básica de requests y WEATHER_SCHEMA


class MockResponse:
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        return self._json_data


class MockSchema:
    @staticmethod
    def validate(data):
        if "expected_key" in data:
            return data
        raise ValueError("Invalid schema")


module_1_meteo_api.WEATHER_SCHEMA = MockSchema


def fake_requests_get(url, params=None, headers=None):
    if url.endswith("success"):
        return MockResponse(200, {"expected_key": "value"})
    elif url.endswith("validation_error"):
        return MockResponse(200, {"unexpected_key": "value"})
    elif url.endswith("bad_request"):
        return MockResponse(400, text="Bad Request")
    elif url.endswith("rate_limit"):
        return MockResponse(429)
    else:
        raise Exception("Network Error")


def test_call_api_success(monkeypatch):
    monkeypatch.setattr(module_1_meteo_api.requests, "get", fake_requests_get)
    result = module_1_meteo_api.call_api("http://fakeurl.com/success")
    assert result == {"expected_key": "value"}


def test_call_api_validation_error(monkeypatch):
    monkeypatch.setattr(module_1_meteo_api.requests, "get", fake_requests_get)
    with pytest.raises(ValueError):
        module_1_meteo_api.call_api("http://fakeurl.com/validation_error")


def test_call_api_400_error(monkeypatch):
    monkeypatch.setattr(module_1_meteo_api.requests, "get", fake_requests_get)
    with pytest.raises(ValueError):
        module_1_meteo_api.call_api("http://fakeurl.com/bad_request")


def test_call_api_rate_limit(monkeypatch):
    llamada = {"count": 0}

    def custom_requests_get(url, params=None, headers=None):
        if llamada["count"] == 0:
            llamada["count"] += 1
            return MockResponse(429)
        return MockResponse(200, {"expected_key": "value"})

    monkeypatch.setattr(module_1_meteo_api.requests, "get", custom_requests_get)
    result = module_1_meteo_api.call_api(
        "http://fakeurl.com/rate_limit", retries=2, cooldown=0
    )
    assert result == {"expected_key": "value"}


def test_call_api_request_exception(monkeypatch):
    def failing_requests_get(url, params=None, headers=None):
        raise Exception("Network Error")

    monkeypatch.setattr(module_1_meteo_api.requests, "get", failing_requests_get)
    with pytest.raises(Exception):
        module_1_meteo_api.call_api("http://fakeurl.com/error")


def test_get_data_meteo_api_success(monkeypatch):
    monkeypatch.setattr(
        module_1_meteo_api,
        "COORDINATES",
        {"Madrid": {"latitude": 40.4, "longitude": -3.7}},
    )
    monkeypatch.setattr(
        module_1_meteo_api,
        "VARIABLES",
        ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"],
    )
    monkeypatch.setattr(module_1_meteo_api, "API_URL", "http://fakeurl.com/success")
    monkeypatch.setattr(
        module_1_meteo_api,
        "call_api",
        lambda url, params=None, verbose=True: {
            "daily": {
                "time": ["2024-01-01"],
                "temperature_2m_mean": [10],
                "precipitation_sum": [5],
                "wind_speed_10m_max": [30],
            }
        },
    )
    df = module_1_meteo_api.get_data_meteo_api("Madrid", verbose=False)
    assert not df.empty
    assert set(df.columns) == {
        "datetime",
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
        "ciudad",
    }


def test_get_data_meteo_api_invalid_city(monkeypatch):
    monkeypatch.setattr(module_1_meteo_api, "COORDINATES", {})
    with pytest.raises(ValueError):
        module_1_meteo_api.get_data_meteo_api("CiudadInventada", verbose=False)


def test_get_data_meteo_api_call_api_failure(monkeypatch):
    monkeypatch.setattr(
        module_1_meteo_api,
        "COORDINATES",
        {"Madrid": {"latitude": 40.4, "longitude": -3.7}},
    )
    monkeypatch.setattr(
        module_1_meteo_api,
        "VARIABLES",
        ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"],
    )
    monkeypatch.setattr(module_1_meteo_api, "API_URL", "http://fakeurl.com/error")

    def failing_call_api(*args, **kwargs):
        raise Exception("API failed")

    monkeypatch.setattr(module_1_meteo_api, "call_api", failing_call_api)
    df = module_1_meteo_api.get_data_meteo_api("Madrid", verbose=False)
    assert df.empty


def test_convert_to_monthly():
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=3, freq="D"),
            "temperature_2m_mean": [10, 12, 14],
            "precipitation_sum": [0, 5, 2],
            "wind_speed_10m_max": [20, 30, 25],
            "ciudad": ["Madrid"] * 3,
        }
    )
    monthly_df = module_1_meteo_api.convert_to_monthly(df)
    assert monthly_df.shape[0] == 1
    assert set(monthly_df.columns) == {
        "datetime",
        "ciudad",
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
    }


def test_plot_weather_series(tmp_path):
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=3, freq="M"),
            "temperature_2m_mean": [10, 12, 14],
            "precipitation_sum": [20, 30, 25],
            "wind_speed_10m_max": [40, 45, 50],
            "ciudad": ["Madrid"] * 3,
        }
    )
    save_path = tmp_path / "output_plot.png"
    module_1_meteo_api.plot_weather_series(df, save_path=str(save_path))
    assert save_path.exists()
