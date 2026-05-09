//
//  WeatherService.swift
//  WeatherApp
//

import Foundation

class WeatherService: ObservableObject {
    @Published var currentWeather: CurrentWeather?
    @Published var dailyForecast: [DailyForecastItem] = []
    @Published var isLoading = false
    @Published var errorMessage: String?
    
    func fetchWeather(latitude: Double, longitude: Double) async {
        isLoading = true
        errorMessage = nil
        
        let urlString = "https://api.open-meteo.com/v1/forecast?latitude=\(latitude)&longitude=\(longitude)&current=temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m&daily=weather_code,temperature_2m_max,temperature_2m_min&timezone=auto"
        
        guard let url = URL(string: urlString) else {
            errorMessage = "Invalid URL"
            isLoading = false
            return
        }
        
        do {
            let (data, response) = try await URLSession.shared.data(from: url)
            
            guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
                errorMessage = "Failed to fetch weather data"
                isLoading = false
                return
            }
            
            let decoder = JSONDecoder()
            let weatherResponse = try decoder.decode(WeatherResponse.self, from: data)
            
            DispatchQueue.main.async {
                self.currentWeather = weatherResponse.current
                self.dailyForecast = self.parseDailyForecast(from: weatherResponse.daily)
                self.isLoading = false
            }
        } catch {
            DispatchQueue.main.async {
                self.errorMessage = "Error: \(error.localizedDescription)"
                self.isLoading = false
            }
        }
    }
    
    private func parseDailyForecast(from daily: DailyWeather?) -> [DailyForecastItem] {
        guard let daily = daily else { return [] }
        
        var forecastItems: [DailyForecastItem] = []
        
        for i in 0..<min(daily.time.count, 7) {
            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "yyyy-MM-dd"
            dateFormatter.locale = Locale(identifier: "en_US")
            
            if let date = dateFormatter.date(from: daily.time[i]) {
                let dayNameFormatter = DateFormatter()
                dayNameFormatter.dateFormat = "EEEE"
                dayNameFormatter.locale = Locale(identifier: "en_US")
                let dayName = dayNameFormatter.string(from: date)
                
                let item = DailyForecastItem(
                    day: dayName,
                    temperatureMax: daily.temperatureMax[i],
                    temperatureMin: daily.temperatureMin[i],
                    weatherCode: daily.weatherCode[i]
                )
                forecastItems.append(item)
            }
        }
        
        return forecastItems
    }
    
    func searchCities(query: String) async -> [GeocodeResult] {
        guard !query.isEmpty else { return [] }
        
        let urlString = "https://geocoding-api.open-meteo.com/v1/search?name=\(query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? "")&count=10&language=en&format=json"
        
        guard let url = URL(string: urlString) else { return [] }
        
        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            let decoder = JSONDecoder()
            let response = try decoder.decode(GeocodingResponse.self, from: data)
            return response.results
        } catch {
            print("Search error: \(error)")
            return []
        }
    }
}

struct DailyForecastItem: Identifiable {
    let id = UUID()
    let day: String
    let temperatureMax: Double
    let temperatureMin: Double
    let weatherCode: Int
}
