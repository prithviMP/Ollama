#!/usr/bin/env python3
"""
Weather Information Tool for Agents

Provides comprehensive weather data including current conditions, forecasts,
alerts, and historical data using multiple weather APIs.
"""

import json
import time
import requests
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from langchain.tools import BaseTool


class WeatherTool(BaseTool):
    """Comprehensive weather information tool."""
    
    name = "weather"
    description = """
    Get weather information for any location worldwide.
    
    Input format: "location" or "location|type|options"
    Types: current, forecast, alerts, historical
    Options: units=metric/imperial, days=N, lang=en
    
    Examples:
    - "New York"
    - "London|forecast|days=5"
    - "Tokyo|current|units=metric"
    - "Paris|alerts"
    
    Returns: Formatted weather information with conditions, temperature, and forecasts
    """
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key  # Would be loaded from environment
        self.base_urls = {
            "openweathermap": "https://api.openweathermap.org/data/2.5",
            "weatherapi": "https://api.weatherapi.com/v1",
            # Mock API for demo purposes
            "mock": "https://mock-weather-api.com"
        }
        self.default_provider = "mock"  # Use mock for demo
        self.cache = {}
        self.cache_ttl = 600  # 10 minutes
    
    def _run(self, query_input: str) -> str:
        """Get weather information."""
        try:
            # Parse input
            location, weather_type, options = self._parse_input(query_input)
            
            if not location.strip():
                return "Error: No location provided"
            
            # Check cache
            cache_key = f"{location}|{weather_type}|{json.dumps(options, sort_keys=True)}"
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if time.time() - cached_data["timestamp"] < self.cache_ttl:
                    return cached_data["result"]
            
            # Get weather data
            if weather_type == "current":
                result = self._get_current_weather(location, options)
            elif weather_type == "forecast":
                result = self._get_forecast(location, options)
            elif weather_type == "alerts":
                result = self._get_weather_alerts(location, options)
            elif weather_type == "historical":
                result = self._get_historical_weather(location, options)
            else:
                result = self._get_current_weather(location, options)  # Default
            
            # Cache result
            self.cache[cache_key] = {
                "timestamp": time.time(),
                "result": result
            }
            
            return result
        
        except Exception as e:
            return f"Error getting weather information: {str(e)}"
    
    def _parse_input(self, query_input: str) -> tuple:
        """Parse weather query input."""
        parts = query_input.split("|")
        location = parts[0].strip()
        weather_type = parts[1].strip() if len(parts) > 1 else "current"
        
        options = {
            "units": "metric",
            "days": 5,
            "lang": "en"
        }
        
        # Parse options
        if len(parts) > 2:
            for option in parts[2].split(","):
                if "=" in option:
                    key, value = option.strip().split("=", 1)
                    if key == "days":
                        try:
                            options[key] = int(value)
                        except ValueError:
                            pass
                    else:
                        options[key] = value.strip()
        
        return location, weather_type, options
    
    def _get_current_weather(self, location: str, options: Dict) -> str:
        """Get current weather conditions."""
        try:
            # Mock weather data (in production, would call real API)
            weather_data = self._get_mock_current_weather(location, options)
            
            return self._format_current_weather(location, weather_data, options)
        
        except Exception as e:
            return f"Error getting current weather: {str(e)}"
    
    def _get_mock_current_weather(self, location: str, options: Dict) -> Dict:
        """Generate mock current weather data."""
        # Simple mock data based on location hash for consistency
        location_hash = hash(location) % 100
        
        # Base temperature around 20°C with location variation
        base_temp = 20 + (location_hash % 30) - 15
        
        conditions = ["clear", "partly cloudy", "cloudy", "rain", "snow", "thunderstorm"]
        condition = conditions[location_hash % len(conditions)]
        
        # Adjust temperature based on condition
        if condition == "snow":
            base_temp -= 10
        elif condition == "rain":
            base_temp -= 5
        elif condition == "clear":
            base_temp += 5
        
        return {
            "location": location,
            "temperature": base_temp,
            "condition": condition,
            "humidity": 60 + (location_hash % 40),
            "wind_speed": 5 + (location_hash % 20),
            "wind_direction": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][location_hash % 8],
            "pressure": 1013 + (location_hash % 30) - 15,
            "visibility": 10 + (location_hash % 10),
            "uv_index": max(0, min(11, (location_hash % 12))),
            "sunrise": "06:30",
            "sunset": "18:45"
        }
    
    def _format_current_weather(self, location: str, data: Dict, options: Dict) -> str:
        """Format current weather data."""
        units = options.get("units", "metric")
        temp_unit = "°C" if units == "metric" else "°F"
        speed_unit = "km/h" if units == "metric" else "mph"
        
        # Convert temperature if needed
        temp = data["temperature"]
        if units == "imperial":
            temp = (temp * 9/5) + 32
        
        # Convert wind speed if needed
        wind_speed = data["wind_speed"]
        if units == "imperial":
            wind_speed = wind_speed * 0.621371  # km/h to mph
        
        formatted = f"🌤️ Current Weather for {location}\\n"
        formatted += "=" * 50 + "\\n\\n"
        
        # Main conditions
        formatted += f"🌡️ Temperature: {temp:.1f}{temp_unit}\\n"
        formatted += f"☁️ Condition: {data['condition'].title()}\\n"
        formatted += f"💧 Humidity: {data['humidity']}%\\n"
        formatted += f"💨 Wind: {wind_speed:.1f} {speed_unit} {data['wind_direction']}\\n"
        formatted += f"📊 Pressure: {data['pressure']} hPa\\n"
        formatted += f"👁️ Visibility: {data['visibility']} km\\n"
        formatted += f"☀️ UV Index: {data['uv_index']}/11\\n\\n"
        
        # Sun times
        formatted += f"🌅 Sunrise: {data['sunrise']}\\n"
        formatted += f"🌇 Sunset: {data['sunset']}\\n\\n"
        
        # Weather advice
        formatted += self._get_weather_advice(data)
        
        formatted += f"\\n🕒 Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return formatted
    
    def _get_forecast(self, location: str, options: Dict) -> str:
        """Get weather forecast."""
        try:
            days = min(options.get("days", 5), 10)  # Limit to 10 days
            forecast_data = self._get_mock_forecast(location, days, options)
            
            return self._format_forecast(location, forecast_data, options)
        
        except Exception as e:
            return f"Error getting forecast: {str(e)}"
    
    def _get_mock_forecast(self, location: str, days: int, options: Dict) -> List[Dict]:
        """Generate mock forecast data."""
        forecast = []
        location_hash = hash(location) % 100
        
        # Get current weather as base
        current = self._get_mock_current_weather(location, options)
        base_temp = current["temperature"]
        
        for day in range(days):
            # Vary temperature over days
            temp_variation = (day * 2) - days + (location_hash % 10) - 5
            high_temp = base_temp + temp_variation + 5
            low_temp = base_temp + temp_variation - 5
            
            # Vary conditions
            conditions = ["clear", "partly cloudy", "cloudy", "rain", "thunderstorm"]
            condition_index = (location_hash + day) % len(conditions)
            condition = conditions[condition_index]
            
            # Calculate rain probability
            rain_prob = 10 if condition == "clear" else 30 if condition == "partly cloudy" else 60 if condition == "cloudy" else 90
            
            forecast_day = {
                "date": (datetime.now() + timedelta(days=day)).strftime("%Y-%m-%d"),
                "day_name": (datetime.now() + timedelta(days=day)).strftime("%A"),
                "high_temp": high_temp,
                "low_temp": low_temp,
                "condition": condition,
                "rain_probability": rain_prob,
                "humidity": 50 + ((location_hash + day) % 40),
                "wind_speed": 5 + ((location_hash + day) % 15)
            }
            
            forecast.append(forecast_day)
        
        return forecast
    
    def _format_forecast(self, location: str, forecast_data: List[Dict], options: Dict) -> str:
        """Format forecast data."""
        units = options.get("units", "metric")
        temp_unit = "°C" if units == "metric" else "°F"
        speed_unit = "km/h" if units == "metric" else "mph"
        
        formatted = f"📅 {len(forecast_data)}-Day Weather Forecast for {location}\\n"
        formatted += "=" * 60 + "\\n\\n"
        
        for day_data in forecast_data:
            # Convert temperatures if needed
            high_temp = day_data["high_temp"]
            low_temp = day_data["low_temp"]
            wind_speed = day_data["wind_speed"]
            
            if units == "imperial":
                high_temp = (high_temp * 9/5) + 32
                low_temp = (low_temp * 9/5) + 32
                wind_speed = wind_speed * 0.621371
            
            # Get weather emoji
            emoji = self._get_weather_emoji(day_data["condition"])
            
            formatted += f"📆 **{day_data['day_name']}, {day_data['date']}**\\n"
            formatted += f"{emoji} {day_data['condition'].title()}\\n"
            formatted += f"🌡️ High: {high_temp:.0f}{temp_unit} | Low: {low_temp:.0f}{temp_unit}\\n"
            formatted += f"🌧️ Rain: {day_data['rain_probability']}% | 💨 Wind: {wind_speed:.0f} {speed_unit}\\n"
            formatted += f"💧 Humidity: {day_data['humidity']}%\\n\\n"
        
        formatted += f"🕒 Forecast updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return formatted
    
    def _get_weather_alerts(self, location: str, options: Dict) -> str:
        """Get weather alerts and warnings."""
        try:
            # Mock alerts data
            alerts = self._get_mock_alerts(location)
            
            if not alerts:
                return f"✅ No active weather alerts for {location}\\n\\nAll clear! No weather warnings or advisories are currently in effect."
            
            formatted = f"🚨 Weather Alerts for {location}\\n"
            formatted += "=" * 50 + "\\n\\n"
            
            for alert in alerts:
                severity_emoji = {"Low": "🟡", "Medium": "🟠", "High": "🔴", "Extreme": "🟣"}
                emoji = severity_emoji.get(alert["severity"], "⚠️")
                
                formatted += f"{emoji} **{alert['title']}**\\n"
                formatted += f"📊 Severity: {alert['severity']}\\n"
                formatted += f"🕒 Effective: {alert['start_time']} - {alert['end_time']}\\n"
                formatted += f"📝 {alert['description']}\\n\\n"
            
            return formatted
        
        except Exception as e:
            return f"Error getting weather alerts: {str(e)}"
    
    def _get_mock_alerts(self, location: str) -> List[Dict]:
        """Generate mock weather alerts."""
        location_hash = hash(location) % 100
        
        # 30% chance of having alerts
        if location_hash % 10 < 3:
            return []
        
        alerts = []
        
        # Generate 1-2 alerts
        alert_types = [
            ("High Wind Warning", "High", "Strong winds expected with gusts up to 80 km/h. Secure loose objects."),
            ("Heavy Rain Advisory", "Medium", "Heavy rainfall expected. Watch for flooding in low-lying areas."),
            ("Heat Advisory", "Medium", "Dangerously hot conditions. Stay hydrated and avoid prolonged sun exposure."),
            ("Winter Storm Watch", "High", "Significant snowfall expected. Travel may become dangerous."),
            ("Fog Advisory", "Low", "Dense fog reducing visibility. Drive carefully and use headlights.")
        ]
        
        for i in range((location_hash % 2) + 1):
            alert_type = alert_types[(location_hash + i) % len(alert_types)]
            
            start_time = datetime.now() + timedelta(hours=i*6)
            end_time = start_time + timedelta(hours=12 + i*6)
            
            alerts.append({
                "title": alert_type[0],
                "severity": alert_type[1],
                "description": alert_type[2],
                "start_time": start_time.strftime("%Y-%m-%d %H:%M"),
                "end_time": end_time.strftime("%Y-%m-%d %H:%M")
            })
        
        return alerts
    
    def _get_historical_weather(self, location: str, options: Dict) -> str:
        """Get historical weather data."""
        try:
            # Mock historical data for past 7 days
            historical_data = self._get_mock_historical(location, 7, options)
            
            return self._format_historical(location, historical_data, options)
        
        except Exception as e:
            return f"Error getting historical weather: {str(e)}"
    
    def _get_mock_historical(self, location: str, days: int, options: Dict) -> List[Dict]:
        """Generate mock historical weather data."""
        historical = []
        location_hash = hash(location) % 100
        
        # Get current base temperature
        current = self._get_mock_current_weather(location, options)
        base_temp = current["temperature"]
        
        for day in range(days):
            date = datetime.now() - timedelta(days=days-day)
            
            # Vary temperature for historical data
            temp_variation = (day * 1.5) - (days/2) + (location_hash % 8) - 4
            avg_temp = base_temp + temp_variation
            
            # Historical conditions
            conditions = ["clear", "partly cloudy", "cloudy", "rain"]
            condition = conditions[(location_hash + day) % len(conditions)]
            
            historical_day = {
                "date": date.strftime("%Y-%m-%d"),
                "day_name": date.strftime("%A"),
                "avg_temp": avg_temp,
                "high_temp": avg_temp + 4,
                "low_temp": avg_temp - 4,
                "condition": condition,
                "rainfall": 0 if condition in ["clear", "partly cloudy"] else (location_hash + day) % 10,
                "humidity": 40 + ((location_hash + day) % 50)
            }
            
            historical.append(historical_day)
        
        return historical
    
    def _format_historical(self, location: str, historical_data: List[Dict], options: Dict) -> str:
        """Format historical weather data."""
        units = options.get("units", "metric")
        temp_unit = "°C" if units == "metric" else "°F"
        
        formatted = f"📊 Historical Weather for {location} (Past {len(historical_data)} Days)\\n"
        formatted += "=" * 60 + "\\n\\n"
        
        total_rainfall = 0
        avg_temp_sum = 0
        
        for day_data in historical_data:
            # Convert temperatures if needed
            avg_temp = day_data["avg_temp"]
            high_temp = day_data["high_temp"]
            low_temp = day_data["low_temp"]
            
            if units == "imperial":
                avg_temp = (avg_temp * 9/5) + 32
                high_temp = (high_temp * 9/5) + 32
                low_temp = (low_temp * 9/5) + 32
            
            emoji = self._get_weather_emoji(day_data["condition"])
            rainfall = day_data["rainfall"]
            total_rainfall += rainfall
            avg_temp_sum += avg_temp
            
            formatted += f"📅 {day_data['day_name']}, {day_data['date']}\\n"
            formatted += f"{emoji} {day_data['condition'].title()}\\n"
            formatted += f"🌡️ Avg: {avg_temp:.1f}{temp_unit} (High: {high_temp:.1f}{temp_unit}, Low: {low_temp:.1f}{temp_unit})\\n"
            
            if rainfall > 0:
                formatted += f"🌧️ Rainfall: {rainfall} mm\\n"
            
            formatted += f"💧 Humidity: {day_data['humidity']}%\\n\\n"
        
        # Summary statistics
        avg_period_temp = avg_temp_sum / len(historical_data)
        formatted += f"📈 **Period Summary:**\\n"
        formatted += f"🌡️ Average Temperature: {avg_period_temp:.1f}{temp_unit}\\n"
        formatted += f"🌧️ Total Rainfall: {total_rainfall} mm\\n"
        
        return formatted
    
    def _get_weather_emoji(self, condition: str) -> str:
        """Get appropriate emoji for weather condition."""
        emoji_map = {
            "clear": "☀️",
            "partly cloudy": "⛅",
            "cloudy": "☁️",
            "rain": "🌧️",
            "heavy rain": "⛈️",
            "snow": "❄️",
            "thunderstorm": "⛈️",
            "fog": "🌫️",
            "windy": "💨"
        }
        return emoji_map.get(condition.lower(), "🌤️")
    
    def _get_weather_advice(self, data: Dict) -> str:
        """Generate weather-based advice."""
        condition = data["condition"].lower()
        temp = data["temperature"]
        uv_index = data["uv_index"]
        wind_speed = data["wind_speed"]
        
        advice = "💡 **Weather Advice:**\\n"
        
        # Temperature advice
        if temp < 0:
            advice += "🧥 Very cold - dress warmly and watch for ice\\n"
        elif temp < 10:
            advice += "🧥 Cold - wear warm clothing and layers\\n"
        elif temp > 30:
            advice += "🌡️ Hot - stay hydrated and seek shade\\n"
        elif temp > 25:
            advice += "☀️ Warm - perfect weather for outdoor activities\\n"
        
        # UV advice
        if uv_index >= 8:
            advice += "🕶️ Very high UV - use sunscreen and protective clothing\\n"
        elif uv_index >= 6:
            advice += "🧴 High UV - sunscreen recommended\\n"
        
        # Wind advice
        if wind_speed > 25:
            advice += "💨 Strong winds - secure loose objects\\n"
        
        # Condition-specific advice
        if "rain" in condition:
            advice += "☂️ Rain expected - bring an umbrella\\n"
        elif "snow" in condition:
            advice += "❄️ Snow conditions - drive carefully\\n"
        elif "thunderstorm" in condition:
            advice += "⛈️ Thunderstorms - stay indoors if possible\\n"
        
        return advice
    
    async def _arun(self, query_input: str) -> str:
        """Async version of weather tool."""
        return self._run(query_input)


# Example usage and testing
def test_weather_tool():
    """Test the weather tool."""
    weather = WeatherTool()
    
    test_queries = [
        "New York",
        "London|forecast|days=3",
        "Tokyo|current|units=imperial",
        "Paris|alerts",
        "Sydney|historical",
        "Moscow|forecast|days=7,units=metric"
    ]
    
    print("🌤️ Weather Tool Test Results:")
    print("=" * 60)
    
    for query in test_queries:
        result = weather._run(query)
        print(f"Query: {query}")
        print(f"Result:\\n{result}")
        print("-" * 60)


if __name__ == "__main__":
    test_weather_tool()