//
//  SearchView.swift
//  WeatherApp
//

import SwiftUI

struct SearchView: View {
    @Environment(\.dismiss) var dismiss
    @Binding var searchText: String
    @Binding var searchResults: [GeocodeResult]
    @Binding var isSearching: Bool
    @ObservedObject var weatherService: WeatherService
    let onSelectCity: (GeocodeResult) -> Void
    let onSaveSearchText: (String) -> Void
    
    private let searchDebounceTime = 0.5
    private var debounceTimer: Timer?
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Search Bar
                HStack {
                    Image(systemName: "magnifyingglass")
                        .foregroundColor(.secondary)
                    
                    TextField("Search city...", text: $searchText)
                        .textFieldStyle(PlainTextFieldStyle())
                        .onChange(of: searchText) { newValue in
                            onSaveSearchText(newValue)
                            performSearch(query: newValue)
                        }
                    
                    if !searchText.isEmpty {
                        Button(action: {
                            searchText = ""
                            searchResults = []
                        }) {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundColor(.secondary)
                        }
                    }
                }
                .padding(12)
                .background(Color(.systemGray6))
                .cornerRadius(10)
                .padding()
                
                // Results
                if isSearching {
                    ProgressView()
                        .padding()
                } else if searchResults.isEmpty && !searchText.isEmpty {
                    VStack(spacing: 16) {
                        Image(systemName: "location.slash")
                            .font(.system(size: 40))
                            .foregroundColor(.gray)
                        Text("No cities found")
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    List(searchResults) { result in
                        Button(action: {
                            onSelectCity(result)
                        }) {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(result.name)
                                    .font(.headline)
                                if let admin = result.admin1 {
                                    Text(admin)
                                        .font(.subheadline)
                                        .foregroundColor(.secondary)
                                }
                                if let country = result.country {
                                    Text(country)
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                        }
                        .buttonStyle(PlainButtonStyle())
                    }
                }
            }
            .navigationTitle("Add City")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
        }
    }
    
    private func performSearch(query: String) {
        debounceTimer?.invalidate()
        
        guard !query.isEmpty else {
            searchResults = []
            return
        }
        
        isSearching = true
        
        debounceTimer = Timer.scheduledTimer(withTimeInterval: searchDebounceTime, repeats: false) { _ in
            Task {
                let results = await weatherService.searchCities(query: query)
                DispatchQueue.main.async {
                    searchResults = results
                    isSearching = false
                }
            }
        }
    }
}

#Preview {
    SearchView(
        searchText: .constant(""),
        searchResults: .constant([]),
        isSearching: .constant(false),
        weatherService: WeatherService(),
        onSelectCity: { _ in },
        onSaveSearchText: { _ in }
    )
}
