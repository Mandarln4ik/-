//
//  Models.swift
//  WeatherApp
//

import Foundation

struct GeocodingResponse: Codable {
    let results: [GeocodeResult]
}

struct GeocodeResult: Codable, Identifiable {
    let id: Int
    let name: String
    let latitude: Double
    let longitude: Double
    let country: String?
    let admin1: String?
    
    var displayName: String {
        var parts = [name]
        if let admin = admin1 { parts.append(admin) }
        if let country = country { parts.append(country) }
        return parts.joined(separator: ", ")
    }
}

struct WeatherResponse: Codable {
    let current: CurrentWeather?
    let daily: DailyWeather?
}

struct CurrentWeather: Codable {
    let temperature: Double
    let weatherCode: Int
    let windSpeed: Double
    let relativeHumidity: Int
    
    enum CodingKeys: String, CodingKey {
        case temperature = "temperature_2m"
        case weatherCode = "weather_code"
        case windSpeed = "wind_speed_10m"
        case relativeHumidity = "relative_humidity_2m"
    }
}

struct DailyWeather: Codable {
    let time: [String]
    let temperatureMax: [Double]
    let temperatureMin: [Double]
    let weatherCode: [Int]
    
    enum CodingKeys: String, CodingKey {
        case time
        case temperatureMax = "temperature_2m_max"
        case temperatureMin = "temperature_2m_min"
        case weatherCode = "weather_code"
    }
}

struct SavedCity: Codable, Identifiable, Equatable {
    let id: Int
    let name: String
    let latitude: Double
    let longitude: Double
    let country: String?
    
    var displayName: String {
        if let country = country {
            return "\(name), \(country)"
        }
        return name
    }
}
