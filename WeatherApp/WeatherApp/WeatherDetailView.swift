//
//  WeatherDetailView.swift
//  WeatherApp
//

import SwiftUI

struct WeatherDetailView: View {
    let city: SavedCity
    @ObservedObject var weatherService: WeatherService
    let onDelete: (SavedCity) -> Void
    
    var body: some View {
        ScrollView {
            VStack(spacing: 30) {
                // City Header
                VStack(spacing: 8) {
                    Text(city.name)
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    if let country = city.country {
                        Text(country)
                            .font(.title3)
                            .foregroundColor(.secondary)
                    }
                }
                .padding(.top, 20)
                
                // Current Weather
                if weatherService.isLoading {
                    ProgressView()
                        .scaleEffect(1.5)
                        .padding()
                } else if let error = weatherService.errorMessage {
                    VStack(spacing: 16) {
                        Image(systemName: "exclamationmark.triangle")
                            .font(.system(size: 40))
                            .foregroundColor(.orange)
                        Text(error)
                            .foregroundColor(.secondary)
                        Button("Retry") {
                            Task {
                                await weatherService.fetchWeather(
                                    latitude: city.latitude,
                                    longitude: city.longitude
                                )
                            }
                        }
                        .buttonStyle(.borderedProminent)
                    }
                    .padding()
                } else if let current = weatherService.currentWeather {
                    CurrentWeatherView(current: current)
                }
                
                Divider()
                
                // 7-Day Forecast
                if !weatherService.dailyForecast.isEmpty {
                    VStack(alignment: .leading, spacing: 16) {
                        Text("7-Day Forecast")
                            .font(.headline)
                            .foregroundColor(.secondary)
                        
                        ForEach(weatherService.dailyForecast) { item in
                            ForecastRow(item: item)
                        }
                    }
                    .padding(.horizontal)
                }
            }
        }
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button(role: .destructive) {
                    onDelete(city)
                } label: {
                    Image(systemName: "trash")
                }
            }
        }
        .task {
            await weatherService.fetchWeather(
                latitude: city.latitude,
                longitude: city.longitude
            )
        }
        .refreshable {
            await weatherService.fetchWeather(
                latitude: city.latitude,
                longitude: city.longitude
            )
        }
    }
}

struct CurrentWeatherView: View {
    let current: CurrentWeather
    
    var body: some View {
        VStack(spacing: 20) {
            HStack(spacing: 30) {
                Image(systemName: weatherIcon(for: current.weatherCode))
                    .font(.system(size: 80))
                    .foregroundColor(.blue)
                
                VStack(alignment: .leading, spacing: 8) {
                    Text("\(Int(current.temperature))°C")
                        .font(.system(size: 60))
                        .fontWeight(.bold)
                    
                    Text(weatherDescription(for: current.weatherCode))
                        .font(.title2)
                        .foregroundColor(.secondary)
                }
            }
            
            HStack(spacing: 40) {
                WeatherStatItem(
                    icon: "wind",
                    value: "\(Int(current.windSpeed)) km/h",
                    label: "Wind"
                )
                
                WeatherStatItem(
                    icon: "humidity",
                    value: "\(current.relativeHumidity)%",
                    label: "Humidity"
                )
            }
        }
        .padding()
    }
    
    private func weatherIcon(for code: Int) -> String {
        switch code {
        case 0: return "sun.max.fill"
        case 1, 2, 3: return "cloud.sun.fill"
        case 45, 48: return "cloud.fog.fill"
        case 51, 53, 55, 56, 57: return "cloud.drizzle.fill"
        case 61, 63, 65, 66, 67, 80, 81, 82: return "cloud.rain.fill"
        case 71, 73, 75, 77, 85, 86: return "cloud.snow.fill"
        case 95, 96, 99: return "cloud.bolt.rain.fill"
        default: return "cloud.fill"
        }
    }
    
    private func weatherDescription(for code: Int) -> String {
        switch code {
        case 0: return "Clear sky"
        case 1: return "Mainly clear"
        case 2, 3: return "Partly cloudy"
        case 45, 48: return "Foggy"
        case 51, 53, 55: return "Drizzle"
        case 56, 57: return "Freezing drizzle"
        case 61, 63, 65: return "Rain"
        case 66, 67: return "Freezing rain"
        case 71, 73, 75: return "Snow fall"
        case 77: return "Snow grains"
        case 80, 81, 82: return "Rain showers"
        case 85, 86: return "Snow showers"
        case 95: return "Thunderstorm"
        case 96, 99: return "Thunderstorm with hail"
        default: return "Unknown"
        }
    }
}

struct WeatherStatItem: View {
    let icon: String
    let value: String
    let label: String
    
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(.secondary)
            Text(value)
                .font(.headline)
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
}

struct ForecastRow: View {
    let item: DailyForecastItem
    
    var body: some View {
        HStack {
            Text(item.day)
                .frame(width: 80, alignment: .leading)
            
            Image(systemName: weatherIcon(for: item.weatherCode))
                .frame(width: 40)
                .foregroundColor(.blue)
            
            Spacer()
            
            HStack(spacing: 4) {
                Text("\(Int(item.temperatureMax))°")
                    .fontWeight(.semibold)
                Text("/")
                    .foregroundColor(.secondary)
                Text("\(Int(item.temperatureMin))°")
                    .foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 4)
    }
    
    private func weatherIcon(for code: Int) -> String {
        switch code {
        case 0: return "sun.max.fill"
        case 1, 2, 3: return "cloud.sun.fill"
        case 45, 48: return "cloud.fog.fill"
        case 51, 53, 55, 56, 57: return "cloud.drizzle.fill"
        case 61, 63, 65, 66, 67, 80, 81, 82: return "cloud.rain.fill"
        case 71, 73, 75, 77, 85, 86: return "cloud.snow.fill"
        case 95, 96, 99: return "cloud.bolt.rain.fill"
        default: return "cloud.fill"
        }
    }
}

#Preview {
    NavigationView {
        WeatherDetailView(
            city: SavedCity(id: 1, name: "London", latitude: 51.5074, longitude: -0.1278, country: "United Kingdom"),
            weatherService: WeatherService(),
            onDelete: { _ in }
        )
    }
}
