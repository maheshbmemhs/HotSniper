#ifndef __DYN_THREAD_MAPPING_H
#define __DYN_THREAD_MAPPING_H
#include <vector>
#include "dvfspolicy.h"
#include "migrationpolicy.h"
#include "performance_counters.h"
#include "neighbor_prediction.h"
#include "ml_temperature_predictor.h"

class DynThreadMapping : public DVFSPolicy, public MigrationPolicy {
public:
DynThreadMapping(const PerformanceCounters *performanceCounters, 
                int coreRows, 
                int coreColumns, 
                std::string profile_path,
                std::string thermal_model_path,
                float tolerance,
                float dvfs_interval,
                std::vector<float> core_states,
                float dtmCriticalTemperature, 
                float dtmRecoveredTemperature);

    virtual std::vector<int> getFrequencies(const std::vector<int> &oldFrequencies,const std::vector<bool> &activeCores);
    virtual std::vector<migration> migrate(SubsecondTime time, const std::vector<int> &taskIds, const std::vector<bool> &activeCores);

    struct Move {
        int core;
        int from_state;
        int to_state;

        double delta_ips;
        double delta_temp;
        double score;
        Move(){
            score=0.0f;
        }
        Move(int core_, int from, int to, double d_ips, double d_temp, double score_)
            : core(core_),
            from_state(from),
            to_state(to),
            delta_ips(d_ips),
            delta_temp(d_temp),
            score(score_) {}

        friend bool operator<(const Move& a, const Move& b) {
            return a.score < b.score;
        }
    };

private:
    
    bool migrationOccured {false};

    const PerformanceCounters *performanceCounters{nullptr};
    unsigned int coreRows{0};
    unsigned int coreColumns{0};
    std::vector<float> core_states;
    NeighborPrediction pred;
    float dtmCriticalTemperature;
    float dtmRecoveredTemperature;
    float tolerance {5.0f};

    float thermalInertia {3.0f};
    float dvfsInterval {1.0f};

    bool in_throttle_mode = false;
    bool throttle();

    MLTemperaturePredictor thermal_model;

    Move get_best_move(const std::vector<NeighborPrediction::PredictionMap>& predictions,const std::vector<int>& currentStatesIdx,const std::vector<bool> &activeCores);
    void get_max_freq(const std::vector<bool>& activeCores, std::vector<int>& currentStatesIdx);
    void exchange();
    bool checkConstraints(const std::vector<NeighborPrediction::PredictionMap>& predictions, const std::vector<int>& newFrequencies, const std::vector<bool> &activeCores);
    double getMeasuredIPSBillions(unsigned int coreId);

    double calc_temperature(double current_temp_c, double equilibrium_temp_c, double interval_ms, double tau_ms);
    void logUtilizations(const std::vector<int> &coreIds);
};
#endif