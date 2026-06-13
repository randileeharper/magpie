"""Neon Hail weather API client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from ..config import Settings
from ..errors import WeatherError
from ..models import Reference, SourceKind, WeatherKind, WeatherReport, utc_now


@dataclass(slots=True)
class NeonHailWeatherClient:
    settings: Settings
    transport: httpx.BaseTransport | None = None

    def get_weather(self, zip_code: str, kind: WeatherKind) -> WeatherReport:
        if len(zip_code) != 5 or not zip_code.isdigit():
            raise WeatherError("Weather requests require a five-digit US ZIP code.")
        url = f"{self.settings.weather_base_url.rstrip('/')}/{kind.value}/{zip_code}"
        try:
            with httpx.Client(
                timeout=self.settings.weather_timeout_seconds,
                verify=self.settings.verify_tls,
                transport=self.transport,
            ) as client:
                response = client.get(url)
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise WeatherError(f"Neon Hail weather request failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise WeatherError("Neon Hail returned an invalid response.")

        if kind == WeatherKind.FORECAST:
            summary, answer = self._format_forecast(zip_code, payload)
        else:
            summary, answer = self._format_conditions(zip_code, payload)
        return WeatherReport(
            summary=summary,
            answer=answer,
            reference=Reference(
                source_id=f"neonhail:{kind.value}:{zip_code}",
                title=f"Neon Hail {kind.value} for {zip_code}",
                url=url,
                site_name="Neon Hail",
                published_at=None,
                fetched_at=utc_now(),
                source_kind=SourceKind.WEATHER_API,
            ),
        )

    def _format_conditions(self, zip_code: str, payload: dict[str, Any]) -> tuple[str, str]:
        temperature = self._number(payload.get("temperature"))
        humidity = self._number(payload.get("relativeHumidity"))
        wind_speed = self._number(payload.get("windSpeed"))
        wind_gust = self._number(payload.get("windGust"))
        description = str(payload.get("textDescription") or "").strip()
        lines = [f"Current conditions for {zip_code}:"]
        if description:
            lines.append(description)
        if temperature is not None:
            lines.append(f"Temperature: {temperature}°F")
        if humidity is not None:
            lines.append(f"Humidity: {humidity}%")
        if wind_speed is not None:
            wind = f"Wind: {wind_speed} km/h"
            if wind_gust is not None:
                wind += f", gusting to {wind_gust} km/h"
            lines.append(wind)
        if len(lines) == 1:
            raise WeatherError("Neon Hail returned no usable current conditions.")
        return f"Current weather for {zip_code}.", "\n".join(lines)

    def _format_forecast(self, zip_code: str, payload: dict[str, Any]) -> tuple[str, str]:
        periods = payload.get("periods")
        if not isinstance(periods, list) or not periods:
            raise WeatherError("Neon Hail returned no forecast periods.")
        lines = [f"Forecast for {zip_code}:"]
        for period in periods[:4]:
            if not isinstance(period, dict):
                continue
            name = str(period.get("name") or "Upcoming period")
            detail = str(period.get("detailedForecast") or period.get("shortForecast") or "").strip()
            if detail:
                lines.append(f"{name}: {detail}")
        if len(lines) == 1:
            raise WeatherError("Neon Hail returned no usable forecast details.")
        return f"Weather forecast for {zip_code}.", "\n".join(lines)

    @staticmethod
    def _number(value: Any) -> str | None:
        if not isinstance(value, (int, float)):
            return None
        return f"{value:.1f}".rstrip("0").rstrip(".")
