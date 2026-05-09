//
//  ContentView.swift
//  WeatherApp
//

import SwiftUI

struct ContentView: View {
    @StateObject private var weatherService = WeatherService()
    @State private var selectedCity: SavedCity?
    @State private var savedCities: [SavedCity] = []
    @State private var showingSearch = false
    @State private var searchText = ""
    @State private var searchResults: [GeocodeResult] = []
    @State private var isSearching = false
    
    var body: some View {
        NavigationView {
            ZStack {
                if let city = selectedCity {
                    WeatherDetailView(
                        city: city,
                        weatherService: weatherService,
                        onDelete: deleteCity
                    )
                } else if savedCities.isEmpty {
                    EmptyStateView(showingSearch: $showingSearch)
                } else {
                    SavedCitiesListView(
                        savedCities: savedCities,
                        onSelect: selectCity,
                        onDelete: deleteCity
                    )
                }
            }
            .navigationTitle("Weather")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { showingSearch.toggle() }) {
                        Image(systemName: "plus")
                    }
                }
            }
            .sheet(isPresented: $showingSearch) {
                SearchView(
                    searchText: $searchText,
                    searchResults: $searchResults,
                    isSearching: $isSearching,
                    weatherService: weatherService,
                    onSelectCity: addCity,
                    onSaveSearchText: saveSearchText
                )
            }
            .onAppear {
                loadSavedCities()
            }
        }
    }
    
    private func loadSavedCities() {
        if let data = UserDefaults.standard.data(forKey: "savedCities"),
           let cities = try? JSONDecoder().decode([SavedCity].self, from: data) {
            savedCities = cities
            if selectedCity == nil && !cities.isEmpty {
                selectedCity = cities.first
            }
        }
    }
    
    private func saveSavedCities() {
        if let data = try? JSONEncoder().encode(savedCities) {
            UserDefaults.standard.set(data, forKey: "savedCities")
        }
    }
    
    private func selectCity(_ city: SavedCity) {
        selectedCity = city
    }
    
    private func addCity(_ result: GeocodeResult) {
        let city = SavedCity(
            id: result.id,
            name: result.name,
            latitude: result.latitude,
            longitude: result.longitude,
            country: result.country
        )
        
        if !savedCities.contains(where: { $0.id == city.id }) {
            savedCities.append(city)
            saveSavedCities()
            if selectedCity == nil {
                selectedCity = city
            }
        }
        showingSearch = false
        searchText = ""
        searchResults = []
    }
    
    private func deleteCity(_ city: SavedCity) {
        savedCities.removeAll { $0.id == city.id }
        saveSavedCities()
        if selectedCity?.id == city.id {
            selectedCity = savedCities.first
        }
    }
    
    private func saveSearchText(_ text: String) {
        searchText = text
    }
}

struct EmptyStateView: View {
    @Binding var showingSearch: Bool
    
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "cloud.sun")
                .font(.system(size: 60))
                .foregroundColor(.gray)
            
            Text("No cities added")
                .font(.title2)
                .foregroundColor(.secondary)
            
            Text("Tap + to search and add a city")
                .font(.subheadline)
                .foregroundColor(.secondary)
            
            Button(action: { showingSearch = true }) {
                Text("Add City")
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
        }
    }
}

struct SavedCitiesListView: View {
    let savedCities: [SavedCity]
    let onSelect: (SavedCity) -> Void
    let onDelete: (SavedCity) -> Void
    
    var body: some View {
        List(savedCities) { city in
            Button(action: { onSelect(city) }) {
                HStack {
                    VStack(alignment: .leading) {
                        Text(city.name)
                            .font(.headline)
                        if let country = city.country {
                            Text(country)
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                    }
                    Spacer()
                    Image(systemName: "chevron.right")
                        .foregroundColor(.gray)
                }
            }
            .buttonStyle(PlainButtonStyle())
            .swipeActions(edge: .trailing) {
                Button(role: .destructive) {
                    onDelete(city)
                } label: {
                    Label("Delete", systemImage: "trash")
                }
            }
        }
        .listStyle(InsetGroupedListStyle())
    }
}

#Preview {
    ContentView()
}
