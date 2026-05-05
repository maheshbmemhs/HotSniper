#ifndef __DYN_THREAD_MAPPING_H
#define __DYN_THREAD_MAPPING_H
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

    Move get_best_move(const std::vector<NeighborPrediction::PredictionMap>& predictions,const std::vector<int>& currentStatesIdx,const std::vector<bool> &activeCores);
    void exchange();
    bool checkConstraints(const std::vector<NeighborPrediction::PredictionMap>& predictions, const std::vector<int>& newFrequencies, const std::vector<bool> &activeCores);
    double getMeasuredIPSBillions(unsigned int coreId);

};
#endif