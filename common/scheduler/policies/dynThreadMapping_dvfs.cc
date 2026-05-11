#include "dynThreadMapping_dvfs.h"
#include <iomanip>
#include <iostream>
#include <queue>
#include <algorithm>

DynThreadMapping_dvfs::DynThreadMapping_dvfs(const PerformanceCounters *performanceCounters, 
                                   int coreRows, 
                                   int coreColumns, 
                                   std::string profile_path,
                                   float target_ips_,
                                   std::vector<float> core_states_,
                                   float dtmCriticalTemperature, 
                                   float dtmRecoveredTemperature): 
performanceCounters(performanceCounters),
coreRows(coreRows),
coreColumns(coreColumns),
core_states(core_states_),
target_ips(target_ips_),
pred(profile_path),
dtmCriticalTemperature(dtmCriticalTemperature),
dtmRecoveredTemperature(dtmRecoveredTemperature) {
    std::string s_core_states = "[";
    for(float f:core_states){
        s_core_states+=std::to_string(f)+", ";
    }
    std::cout << "[Scheduler][DynThreadMapping_dvfs]: Initializing with: "
    << "target IPS: "<< target_ips_ 
    << "\n profile path: " << profile_path
    << "\n critical temp: " << dtmCriticalTemperature
    << "\n recovery temp: " << dtmRecoveredTemperature
    << "\n states: " << s_core_states << "]"
    << std::endl;
}

std::vector<int> DynThreadMapping_dvfs::getFrequencies(const std::vector<int> &oldFrequencies, const std::vector<bool> &activeCores) {
    if (throttle()) {
        std::vector<int> minFrequencies(coreRows * coreColumns, core_states[0]*1000);
        std::cout << "[Scheduler][DynThreadMapping_dvfs]: in throttle mode -> return min. frequencies" << std::endl;
        return minFrequencies;
    } else {
        std::cout << "[Scheduler][DynThreadMapping_dvfs]: Starting DVFS"<< std::endl;
        std::cout << "[Scheduler][DynThreadMapping_dvfs]: Active Cores: ";
        for(bool active:activeCores){
            std::cout<<active<<", ";
        }
        std::cout<<std::endl;

        // Non-Parallel Region
        if(std::count(activeCores.begin(), activeCores.end(), true)==1){
            std::cout << "[Scheduler][DynThreadMapping_dvfs]: Non-Parallel region, setting inactive cores to lowest frequency and active core to 2.0 Ghz" << std::endl;
            std::vector<int> newFrequencies(coreRows * coreColumns,core_states[0]);
            newFrequencies[std::distance(std::begin(activeCores), std::find(activeCores.begin(), activeCores.end(),true))]=2000; // Set only active core to 2 GHz
            return newFrequencies;
        }

        // Start all frequencies at the lowest
        std::vector<int> newFrequencies(coreRows * coreColumns,core_states[0]*1000);

        // Tracks the index into core_state, used to increment between states
        std::vector<int> currentStatesIdx(coreRows * coreColumns,0); // Initialization: start with lowest possible frequency, index 0 in the core states
        
        // Get Predictions for each of the active cores
        std::vector<NeighborPrediction::PredictionMap> predictions;

        for(int i =0;i<(coreRows * coreColumns);i++){
            if(activeCores.at(i)){
                float current_ips = getMeasuredIPSBillions(i);
                // Thread is doing next to nothing, give it predictions that make it ignored in any moves
                if(current_ips <= 0.1){
                    std::cout << "[Scheduler][DynThreadMapping_dvfs]: Warning IPS of core "<<i<<" is near zero, using ignored predictions"<< std::endl;
                    NeighborPrediction::PredictionMap temp;
                    for(float state: core_states){
                        temp.emplace(state,NeighborPrediction::core_status{"none",0.0f,0.0f,0.0f,0.0f,0.0f});
                    }
                    predictions.push_back(temp);
                } else{
                    float current_state = float(oldFrequencies[i])/1000.0f; // Mhz to GHz
                    predictions.push_back(pred.getNearestBenchmark(current_state,current_ips));
                }
            }else{
                predictions.push_back({});
            }
        }
        
        bool failed = false;
        while(!checkConstraints(predictions,currentStatesIdx,activeCores)){
            auto best = get_best_move(predictions,currentStatesIdx,activeCores);
            if(best.score==0.0f){ // No more moves but constraints are not met
                std::cout << "[Scheduler][DynThreadMapping_dvfs]: Failed to find core states that meet constraints"<< std::endl;
                failed = true;
                break;
            }
            // Update core state index
            std::cout << "[Scheduler][DynThreadMapping_dvfs]: Move Core "<<best.core<< " frequency from " <<core_states[best.from_state]<<" GHz to " << core_states[best.to_state]<<" GHz"<< std::endl;
            currentStatesIdx[best.core]=best.to_state;

        }
        if(!failed){
            std::cout << "[Scheduler][DynThreadMapping_dvfs]: Completed DVFS"<< std::endl;
        }
        std::cout << "[Scheduler][DynThreadMapping_dvfs]: Final States "<< std::endl;

        // Update core frequency
        for(int i=0;i<(coreRows * coreColumns);i++){
            newFrequencies[i]=int(1000*core_states[currentStatesIdx[i]] + 0.5);
            std::cout << "[Scheduler][DynThreadMapping_dvfs]: core "<< i << ": freq "<<core_states[currentStatesIdx[i]]<< std::endl;

        }
        return newFrequencies;
    }
}

DynThreadMapping_dvfs::Move DynThreadMapping_dvfs::get_best_move(const std::vector<NeighborPrediction::PredictionMap>& predictions,
                                                                 const std::vector<int>& currentStatesIdx,
                                                                 const std::vector<bool>& activeCores){
    Move best_move;
    float best_score = 0.0;
    for(int i = 0;i<currentStatesIdx.size(); i++){
        if(activeCores.at(i)){
            // Cant go any higher than this state
            if(currentStatesIdx[i]+1 >= core_states.size()){
                continue;
            }
            // Get prediction for core i        
            auto prediction = predictions[i];

            // Calculate Move
            // Current IPS and power values
            float current_state = core_states[currentStatesIdx[i]];
            float ips_before = prediction[current_state].ips;
            float power_before = prediction[current_state].power;

            // Potential IPS and Power values of next state
            float new_state = core_states[currentStatesIdx[i]+1]; // next state is just current for core i plus 1
            float ips_after = prediction[new_state].ips;   
            float power_after = prediction[new_state].power;   
            
            double delta_ips = ips_after-ips_before;
            double delta_power = power_after-power_before;
            if (delta_power <= 0.0 || delta_ips <= 0.0) {
                continue;
            }

            double score =delta_ips/delta_power;
            if(score>best_score){
                // Best move seen so far
                best_move = Move(i,currentStatesIdx[i],currentStatesIdx[i]+1,delta_ips,delta_power,score);
                best_score=score;
            }
        }
    }
    return best_move;
}


double DynThreadMapping_dvfs::getMeasuredIPSBillions(unsigned int coreId) {
    if (performanceCounters == NULL) {
        return 0.0;
    }

    try {
        double rawIPS = performanceCounters->getIPSOfCore(coreId);
        if (rawIPS > 0.0) {
            return rawIPS / 1e9;
        }
    } catch (...) {
    }

    return 0.0;
}

bool DynThreadMapping_dvfs::checkConstraints(const std::vector<NeighborPrediction::PredictionMap>& predictions, 
                                             const std::vector<int>& currentStateIdx, 
                                             const std::vector<bool>& activeCores){
    float total_ips = 0.0f;
    for (unsigned int coreCounter = 0; coreCounter < coreRows * coreColumns; coreCounter++) {
        if(activeCores.at(coreCounter)){
            auto prediction = predictions[coreCounter];
            float new_clock_speed = core_states[currentStateIdx[coreCounter]];
            total_ips+=prediction.at(new_clock_speed).ips;
        }
    }
    std::cout << "[Scheduler][DynThreadMapping_dvfs]: Predicted IPS: "<<total_ips<< std::endl;
    return total_ips>=target_ips;
}

bool DynThreadMapping_dvfs::throttle() {
    if (performanceCounters->getPeakTemperature() > dtmCriticalTemperature) {
        if (!in_throttle_mode) {
            std::cout << "[Scheduler][DynThreadMapping_dvfs]: detected thermal violation" <<std::endl;
        }
        in_throttle_mode = true;
    } else if (performanceCounters->getPeakTemperature() < dtmRecoveredTemperature) {
        if (in_throttle_mode) {
            std::cout << "[Scheduler][DynThreadMapping_dvfs]: thermal violation ended" << std::endl;
        }
        in_throttle_mode = false;
    }
    return in_throttle_mode;
}