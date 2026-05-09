#ifndef __DYN_THREAD_MAPPING_H
#define __DYN_THREAD_MAPPING_H
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include "dvfspolicy.h"
#include "performance_counters.h"
#include "neighbor_prediction.h"

class DynThreadMapping_dvfs : public DVFSPolicy {
public:
DynThreadMapping_dvfs(const PerformanceCounters *performanceCounters, 
                int coreRows, 
                int coreColumns, 
                std::string profile_path,
                float target_ips,
                std::vector<float> core_states,
                float dtmCriticalTemperature, 
                float dtmRecoveredTemperature);

    virtual std::vector<int> getFrequencies(const std::vector<int> &oldFrequencies,const std::vector<bool> &activeCores);
    
    struct Move {
        int core;
        int from_state;
        int to_state;

        double delta_ips;
        double delta_power;
        double score;
        Move(){
            score=0.0f;
        }
        Move(int core_, int from, int to, double d_ips, double d_pow, double score_)
            : core(core_),
            from_state(from),
            to_state(to),
            delta_ips(d_ips),
            delta_power(d_pow),
            score(score_) {}

        friend bool operator<(const Move& a, const Move& b) {
            return a.score < b.score;
        }
    };

private:
    

    const PerformanceCounters *performanceCounters{nullptr};
    unsigned int coreRows{0};
    unsigned int coreColumns{0};
    float target_ips{0.0f};
    std::vector<float> core_states;
    NeighborPrediction pred;
    float dtmCriticalTemperature;
    float dtmRecoveredTemperature;
    bool in_throttle_mode = false;
    bool throttle();
    std::vector<int> random_frequency_choices;
    std::mt19937 random_engine;

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

    Move get_best_move(const std::vector<NeighborPrediction::PredictionMap>& predictions,const std::vector<int>& currentStatesIdx,const std::vector<bool> &activeCores);
    void exchange();
    bool checkConstraints(const std::vector<NeighborPrediction::PredictionMap>& predictions, const std::vector<int>& newFrequencies, const std::vector<bool> &activeCores);
    double getMeasuredIPSBillions(unsigned int coreId);

};
#endif
