import requests
from time import sleep
import pandas as pd
from schema import Schema
import matplotlib.pyplot as plt

COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
WEATHER_SCHEMA = Schema(
    {
        "latitude": float,
        "longitude": float,
        "timezone": str,
        "daily": {
            "time": list,
            "temperature_2m_mean": list,
            "precipitation_sum": list,
            "wind_speed_10m_max": list,
        },
    },
    ignore_extra_keys=True,
)

VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]

API_URL = "https://archive-api.open-meteo.com/v1/archive"


def call_api(url, params=None, headers=None, retries=3, cooldown=2, verbose=True):
    """
    Realiza una llamada a la API con reintentos,
    manejo de errores y validación de esquema.
    """
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers)

            if response.status_code == 200:
                data = response.json()
                try:
                    validated_data = WEATHER_SCHEMA.validate(data)
                    return validated_data
                except Exception as e:
                    if verbose:
                        print(f"Error de validación: {e}")
                    raise
            elif response.status_code == 400:
                if verbose:
                    print(f"Error 400: {response.text}")
                raise ValueError("Error en la solicitud a la API.")
            elif response.status_code == 429:
                if verbose:
                    print(f"Rate limit alcanzado. Esperando {cooldown} segundos...")
                sleep(cooldown * 2)
                continue

            else:
                if verbose:
                    print(f"Error HTTP {response.status_code}: {response.text}")
                if attempt < retries - 1:
                    sleep(cooldown)
                    continue
                response.raise_for_status()

        except requests.exceptions.RequestException as e:
            if verbose:
                print(f"Intento {attempt + 1}/{retries} falló: {str(e)}")
            if attempt < retries - 1:
                sleep(cooldown)
                continue
            raise

    raise Exception(f"Error al llamar a la API tras {retries} intentos.")


def get_data_meteo_api(ciudad, start_date=None, end_date=None, verbose=True):
    """
    Obtiene los datos meteorológicos de la API Open-Meteo para una ciudad.
    Permite definir fechas de inicio y fin opcionales.
    """
    if ciudad not in COORDINATES:
        raise ValueError(f"Ciudad no reconocida: {ciudad}")

    lat = COORDINATES[ciudad]["latitude"]
    lon = COORDINATES[ciudad]["longitude"]

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(VARIABLES),
        "timezone": "Europe/Madrid",
    }

    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    try:
        response = call_api(url=API_URL, params=params, verbose=verbose)

        if verbose:
            print(f"\nProcesando datos para {ciudad}")

        dates = pd.to_datetime(response["daily"]["time"])
        df = pd.DataFrame(
            {
                "datetime": dates,
                "temperature_2m_mean": response["daily"]["temperature_2m_mean"],
                "precipitation_sum": response["daily"]["precipitation_sum"],
                "wind_speed_10m_max": response["daily"]["wind_speed_10m_max"],
            }
        )

        df["ciudad"] = ciudad

        if verbose:
            print(f"Datos obtenidos para {ciudad}: {len(df)} registros")

        return df

    except Exception as e:
        if verbose:
            print(f"Error al obtener datos para {ciudad}: {e}")
        return pd.DataFrame()


def convert_to_monthly(df):
    """
    Convierte un DataFrame con datos diarios a resolución mensual.

    Args:
        df (pd.DataFrame): DataFrame con columnas 'datetime', 'temperature_2m',
                          'precipitation', 'wind_speed' y 'ciudad'

    Returns:
        pd.DataFrame: DataFrame con medias mensuales
    """
    df["datetime"] = pd.to_datetime(df["datetime"])

    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month

    monthly_df = (
        df.groupby(["ciudad", "year", "month"])
        .agg(
            {
                "temperature_2m_mean": "mean",
                "precipitation_sum": "sum",
                "wind_speed_10m_max": "max",
            }
        )
        .reset_index()
    )

    monthly_df["datetime"] = pd.to_datetime(monthly_df[["year", "month"]].assign(day=1))

    monthly_df = monthly_df.drop(["year", "month"], axis=1)
    monthly_df = monthly_df[
        [
            "datetime",
            "ciudad",
            "temperature_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_max",
        ]
    ]
    for col in VARIABLES:
        monthly_df[col] = monthly_df[col].round(2)
    return monthly_df


def plot_weather_series(df, save_path=None):
    """
    Crea gráficos de series temporales para las variables meteorológicas.

    Args:
        df (pd.DataFrame): DataFrame con los datos procesados
        save_path (str, optional): Ruta donde guardar el gráfico.
            Si es None, muestra el gráfico.
    """

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("Series Temporales de Variables Meteorológicas", fontsize=14)

    for ciudad in df["ciudad"].unique():
        city_data = df[df["ciudad"] == ciudad]
        ax1.plot(city_data["datetime"], city_data["temperature_2m_mean"], label=ciudad)
    ax1.set_ylabel("Temperatura (°C)")
    ax1.legend()
    ax1.grid(True)

    for ciudad in df["ciudad"].unique():
        city_data = df[df["ciudad"] == ciudad]
        ax2.plot(city_data["datetime"], city_data["precipitation_sum"], label=ciudad)
    ax2.set_ylabel("Precipitación (mm)")
    ax2.legend()
    ax2.grid(True)

    for ciudad in df["ciudad"].unique():
        city_data = df[df["ciudad"] == ciudad]
        ax3.plot(city_data["datetime"], city_data["wind_speed_10m_max"], label=ciudad)
    ax3.set_ylabel("Velocidad del Viento (km/h)")
    ax3.set_xlabel("Fecha")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def main():
    data = pd.DataFrame()

    for ciudad in COORDINATES.keys():
        df_ciudad = get_data_meteo_api(
            ciudad=ciudad, start_date="2010-01-01", end_date="2020-12-31", verbose=True
        )
        if not df_ciudad.empty:
            data = pd.concat([data, df_ciudad], ignore_index=True)

    data_processed = convert_to_monthly(data)
    data_processed.to_csv("meteo_data.csv", index=False)

    plot_weather_series(data_processed)


if __name__ == "__main__":
    main()
