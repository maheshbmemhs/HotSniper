#ifndef __DYN_THREAD_MAPPING_H
#define __DYN_THREAD_MAPPING_H
#include <fstream>
#include <string>
#include <vector>
#include "dvfspolicy.h"
#include "performance_counters.h"
#include "neighbor_prediction.h"
#include "thermal_regression_model.h"

class DynThreadMapping_dvfs : public DVFSPolicy {
public:
DynThreadMapping_dvfs(const PerformanceCounters *performanceCounters, 
                int coreRows, 
                int coreColumns, 
                std::string profile_path,
                std::string thermal_model_path,
                float temperature_constraint,
                std::vector<float> core_states,
                float dtmCriticalTemperature, 
                float dtmRecoveredTemperature);

    virtual std::vector<int> getFrequencies(const std::vector<int> &oldFrequencies,const std::vector<bool> &activeCores);

private:
    

    const PerformanceCounters *performanceCounters{nullptr};
    unsigned int coreRows{0};
    unsigned int coreColumns{0};
    float temperature_constraint{0.0f};
    std::vector<float> core_states;
    NeighborPrediction pred;
    ThermalRegressionModel thermal_model;
    float dtmCriticalTemperature;
    float dtmRecoveredTemperature;
    bool in_throttle_mode = false;
    bool throttle();

    std::ofstream sample_log;
    std::string experiment_name;
    unsigned long long next_cycle_id = 0;
    unsigned long long pending_cycle_id = 0;
    bool has_pending_cycle = false;
    double pending_start_peak_temp = 0.0;
    std::vector<double> pending_start_core_temps;
    std::vector<double> pending_start_core_powers;
    std::vector<double> pending_start_core_utils;
    std::vector<double> pending_start_core_cpis;
    std::vector<double> pending_start_core_rel_nuca_cpis;
    std::vector<double> pending_start_core_ips;
    std::vector<int> pending_old_frequencies;
    std::vector<int> pending_frequencies;
    std::vector<bool> pending_active_cores;

    std::vector<double> getCoreTemperatures() const;
    std::vector<double> getCorePowers() const;
    std::vector<double> getCoreUtilizations() const;
    std::vector<double> getCoreCpis() const;
    std::vector<double> getCoreRelNucaCpis() const;
    std::vector<double> getCoreIps() const;
    void rememberCycle(
        double start_peak_temp,
        const std::vector<double>& start_core_temps,
        const std::vector<double>& start_core_powers,
        const std::vector<double>& start_core_utils,
        const std::vector<double>& start_core_cpis,
        const std::vector<double>& start_core_rel_nuca_cpis,
        const std::vector<double>& start_core_ips,
        const std::vector<int>& old_frequencies,
        const std::vector<int>& frequencies,
        const std::vector<bool>& active_cores);
    void logCompletedCycle(
        double end_peak_temp,
        const std::vector<double>& end_core_temps,
        const std::vector<double>& end_core_powers,
        const std::vector<double>& end_core_utils,
        const std::vector<double>& end_core_cpis,
        const std::vector<double>& end_core_rel_nuca_cpis,
        const std::vector<double>& end_core_ips);
    std::string joinFrequencies(const std::vector<int>& values) const;
    std::string joinBools(const std::vector<bool>& values) const;
    std::string joinTemperatures(const std::vector<double>& values) const;
    std::string joinDoubles(const std::vector<double>& values) const;
    std::string csvEscape(const std::string& value) const;

    double getMeasuredIPSBillions(unsigned int coreId);

};
#endif
