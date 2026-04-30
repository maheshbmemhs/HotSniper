#pragma once
#include <vector>
#include <unordered_map>
#include <string>

class NeighborPrediction{

public:
    // Status of a core at a specific state, all values are averages of a run for all the worker threads
    struct core_status{
        std::string benchmarkName;

        float state; // Clock Speed (GHz)
        float ips;   // Instructions per second / 1e9
        float cpi;   // Cycles per instruction
        float temp;  // Temperature (C)
        float power; // Power (Watts)
    };
    std::unordered_map<float,core_status> getNearestBenchmark(float core_state, float ips);
    NeighborPrediction(std::string benchmark_path);

private:
    std::unordered_map<float, std::vector<core_status>> states; // Map a core state (frequency) to a list of core_status for that state
    std::unordered_map<std::string,std::unordered_map<float,core_status>> benchmarks; // Map a benchmark name to a map of state to core_status for a given benchmark
};