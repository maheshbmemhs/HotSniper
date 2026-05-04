#include "neighbor_prediction.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>

NeighborPrediction::NeighborPrediction(std::string benchmark_path){
    std::ifstream file(benchmark_path);
    if(!file){
        std::cerr << "[NeighborPrediction] Failed to open benchmark file: " << benchmark_path << std::endl;
        throw std::logic_error("Bad Benchmark path");
    }

    std::string line; 
    std::getline(file, line); // Skip header
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        core_status status;

        std::getline(iss, token, '\t'); // Name
        status.benchmarkName=token;
        std::getline(iss, token, '\t'); // State
        status.state=std::stof(token);
        std::getline(iss, token, '\t'); // IPS
        status.ips=std::stof(token);
        std::getline(iss, token, '\t'); // CPI
        status.cpi=std::stof(token);
        std::getline(iss, token, '\t'); // Temp
        status.temp=std::stof(token);
        std::getline(iss, token, '\t'); // Power
        status.power=std::stof(token);

        // Add to state map
        states[status.state].push_back(status);

        // Add to benchmark map
        benchmarks[status.benchmarkName][status.state]=status;
    }
}

/**
 * @brief find the benchmark with the closest IPS for a given core state
 * @param core_state state of the core the thread is running on (state = frequency)
 * @param ips current IPS of the thread
 * 
 * @return Map of core state to status for the nearest benchmark
 * 
 */
std::unordered_map<float,NeighborPrediction::core_status> NeighborPrediction::getNearestBenchmark(float core_state, float ips){
    if(!states.count(core_state)){
        std::cerr << "[NeighborPrediction] Could not find core state " << core_state << " in benchmark data" << std::endl;
        throw std::logic_error("Invalid core state");
    }
    const auto& statuses = states[core_state];
    std::string nearest_benchmark;
    float min = std::numeric_limits<float>::max();
    for(auto s:statuses){
        float diff = abs(s.ips-ips);
        if(diff<min){
            nearest_benchmark= s.benchmarkName;
            min = diff;
        }
    }
    std::cerr << "[NeighborPrediction] Nearest benchmark for IPS "<<ips<<" at state " << core_state <<"GHz is " << nearest_benchmark << std::endl;
    return benchmarks[nearest_benchmark];
}

