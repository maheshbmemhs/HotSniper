#include "dynThreadMapping.h"
#include <iomanip>
#include <iostream>
#include <queue>
#include <algorithm>

DynThreadMapping::DynThreadMapping(const PerformanceCounters *performanceCounters, 
                                   int coreRows, 
                                   int coreColumns, 
                                   std::string profile_path,
                                   float tolerance_,
                                   float dvfs_interval,
                                   std::vector<float> core_states_,
                                   float dtmCriticalTemperature, 
                                   float dtmRecoveredTemperature): 
performanceCounters(performanceCounters),
coreRows(coreRows),
coreColumns(coreColumns),
core_states(core_states_),
pred(profile_path),
dtmCriticalTemperature(dtmCriticalTemperature),
dtmRecoveredTemperature(dtmRecoveredTemperature),
tolerance(tolerance_),
dvfsInterval(dvfs_interval) {
    std::string s_core_states = "[";
    for(float f:core_states){
        s_core_states+=std::to_string(f)+", ";
    }
    std::cout << "[Scheduler][DynThreadMapping]: Initializing with: "
    << "\n profile path: " << profile_path
    << "\n critical temp: " << dtmCriticalTemperature
    << "\n recovery temp: " << dtmRecoveredTemperature
    << "\n tolerance: " << tolerance
    << "\n dvfs interval: " << dvfsInterval
    << "\n states: " << s_core_states << "]"
    << std::endl;
}

std::vector<int> DynThreadMapping::getFrequencies(const std::vector<int> &oldFrequencies, const std::vector<bool> &activeCores) {
    if (throttle()) {
        std::vector<int> minFrequencies(coreRows * coreColumns, core_states[0]*1000);
        std::cout << "[Scheduler][DynThreadMapping]: in throttle mode -> return min. frequencies" << std::endl;
        return minFrequencies;
    } else if(!migrationOccured){
        std::cout << "[Scheduler][DynThreadMapping]: Starting DVFS"<< std::endl;
        std::cout << "[Scheduler][DynThreadMapping]: Active Cores: ";
        for(bool active:activeCores){
            std::cout<<active<<", ";
        }
        std::cout<<std::endl;

        // Non-Parallel Region
        if(std::count(activeCores.begin(), activeCores.end(), true)==1){
            std::cout << "[Scheduler][DynThreadMapping]: Non-Parallel region, setting inactive cores to lowest frequency and active core to 2.0 Ghz" << std::endl;
            std::vector<int> newFrequencies(coreRows * coreColumns,core_states[0]);
            newFrequencies[std::distance(std::begin(activeCores), std::find(activeCores.begin(), activeCores.end(),true))]=2000; // Set only active core to 2 GHz
            return newFrequencies;
        }

        // Start all frequencies at the highest (max IPS)
        std::vector<int> newFrequencies(coreRows * coreColumns,core_states.back()*1000);

        // Tracks the index into core_state, used to increment between states
        std::vector<int> currentStatesIdx(coreRows * coreColumns,core_states.size()-1); // Initialization: start with highest possible frequency, last index in the core states
        
        // Get Predictions for each of the active cores
        std::vector<NeighborPrediction::PredictionMap> predictions;

        for(size_t i =0;i<(coreRows * coreColumns);i++){
            if(activeCores.at(i)){
                float current_ips = getMeasuredIPSBillions(i);
                // Thread is doing next to nothing, give it predictions that make it ignored in any moves
                if(current_ips <= 0.1){
                    std::cout << "[Scheduler][DynThreadMapping]: Warning IPS of core "<<i<<" is near zero, using ignored predictions"<< std::endl;
                    NeighborPrediction::PredictionMap temp;
                    for(float state: core_states){
                        temp.emplace(state,NeighborPrediction::core_status{"none",0.0f,0.0f,0.0f,0.0f,0.0f});
                    }
                    predictions.push_back(temp);
                    currentStatesIdx[i]=0; // If we cant predict states accurately set core to lowest freq to be safe
                } else{
                    float current_state = float(oldFrequencies[i])/1000.0f; // Mhz to GHz
                    predictions.push_back(pred.getNearestBenchmark(current_state,current_ips));
                }
            }else{
                predictions.push_back({});
                currentStatesIdx[i]=0; // Inactive, set to lowest freq
            }
        }
        
        /*bool failed = false;
        while(!checkConstraints(predictions,currentStatesIdx,activeCores)){
            auto best = get_best_move(predictions,currentStatesIdx,activeCores);
            if(best.score==0.0f){ // No more moves but constraints are not met
                std::cout << "[Scheduler][DynThreadMapping]: Failed to find core states that meet constraints"<< std::endl;
                failed = true;
                break;
            }
            // Update core state index
            std::cout << "[Scheduler][DynThreadMapping]: Move Core "<<best.core<< " frequency from " <<core_states[best.from_state]<<" GHz to " << core_states[best.to_state]<<" GHz"<< std::endl;
            currentStatesIdx[best.core]=best.to_state;

        }
        if(!failed){
            std::cout << "[Scheduler][DynThreadMapping]: Completed DVFS"<< std::endl;
        }*/
        get_lowest_freq(predictions,activeCores,currentStatesIdx);
        std::cout << "[Scheduler][DynThreadMapping]: Final States "<< std::endl;

        // Update core frequency
        for(size_t i=0;i<(coreRows * coreColumns);i++){
            newFrequencies[i]=int(1000*core_states[currentStatesIdx[i]]);
            double cur_temp = performanceCounters->getTemperatureOfCore(i);
            double nn_pred_temp = predictions[i][core_states[currentStatesIdx[i]]].temp;
            std::cout << "[Scheduler][DynThreadMapping]: core "<< i << ": freq "<<core_states[currentStatesIdx[i]] 
                      <<" temp "<<cur_temp<<" -> "<<nn_pred_temp<<" = "<<calc_temperature(cur_temp,nn_pred_temp,dvfsInterval,thermalInertia)<< std::endl;

        }
        return newFrequencies;
    } else{
        migrationOccured = false;
        return {oldFrequencies};
    }
}

DynThreadMapping::Move DynThreadMapping::get_best_move(const std::vector<NeighborPrediction::PredictionMap>& predictions,
                                                                 const std::vector<int>& currentStatesIdx,
                                                                 const std::vector<bool>& activeCores){
    Move best_move;
    float best_score = 0.0;
    for(size_t i = 0;i<currentStatesIdx.size(); i++){
        if(activeCores.at(i)){
            // Cant go any lower than this state
            if(!currentStatesIdx[i]){
                continue;
            }
            // Get prediction for core i        
            auto prediction = predictions[i];

            // Calculate Move
            // Current IPS and Temp values
            float current_state = core_states[currentStatesIdx[i]];
            float ips_before = prediction[current_state].ips;
            float temp_before = calc_temperature(performanceCounters->getTemperatureOfCore(i),prediction[current_state].temp,dvfsInterval,thermalInertia);

            // Potential IPS and Temp values of next state
            float new_state = core_states[currentStatesIdx[i]-1]; // next state is just current for core i minus 1
            float ips_after = prediction[new_state].ips;   
            float temp_after = calc_temperature(performanceCounters->getTemperatureOfCore(i),prediction[new_state].temp,dvfsInterval,thermalInertia);
            
            double delta_ips = ips_before-ips_after;
            double delta_temp = temp_before-temp_after;

            // If no temp improvement
            if (delta_temp <= 0.0) {
                continue;
            }
            // No change in IPS is actually good, but we cant divide by zero so set it to epsilon
            if(delta_ips<=0.0){
                delta_ips=std::numeric_limits<double>::epsilon();
            }

            double score =delta_temp/delta_ips;
            if(score>best_score){
                // Best move seen so far
                best_move = Move(i,currentStatesIdx[i],currentStatesIdx[i]-1,delta_ips,delta_temp,score);
                best_score=score;
            }
        }
    }
    return best_move;
}

void DynThreadMapping::get_lowest_freq(const std::vector<NeighborPrediction::PredictionMap>& predictions,
                     const std::vector<bool>& activeCores,
                     std::vector<int>& currentStatesIdx){

    for(size_t i = 0;i<currentStatesIdx.size(); i++){
        if(activeCores.at(i)){
            // Cant go any lower than this state
            if(!currentStatesIdx[i]){
                continue;
            }
            auto prediction = predictions[i];
            double pred_temp = calc_temperature(performanceCounters->getTemperatureOfCore(i),prediction[core_states[currentStatesIdx[i]]].temp,dvfsInterval,thermalInertia);
            while(pred_temp>=dtmCriticalTemperature+tolerance){
                std::cerr<<"Lowering core " <<i<<" freq from " <<core_states[currentStatesIdx[i]] << " GHz to " << core_states[currentStatesIdx[i]-1] << " Ghz" << std::endl;
                currentStatesIdx[i]-=1;
                float new_state = core_states[currentStatesIdx[i]];
                pred_temp = calc_temperature(performanceCounters->getTemperatureOfCore(i),prediction[new_state].temp,dvfsInterval,thermalInertia);
            }
        }
    }
}


double DynThreadMapping::getMeasuredIPSBillions(unsigned int coreId) {
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

bool DynThreadMapping::checkConstraints(const std::vector<NeighborPrediction::PredictionMap>& predictions, 
                                             const std::vector<int>& currentStateIdx, 
                                             const std::vector<bool>& activeCores){
    for (unsigned int coreCounter = 0; coreCounter < coreRows * coreColumns; coreCounter++) {
        if(activeCores.at(coreCounter)){
            auto prediction = predictions[coreCounter];
            float new_clock_speed = core_states[currentStateIdx[coreCounter]];
            float cur_temp = performanceCounters->getTemperatureOfCore(coreCounter);
            float nn_pred_temp = prediction.at(new_clock_speed).temp;
            double temp= calc_temperature(cur_temp,nn_pred_temp,dvfsInterval,thermalInertia);
            std::cout << "[Scheduler][DynThreadMapping]: core "<< coreCounter << ": freq "<<new_clock_speed
                      <<" temp "<<cur_temp<<" -> "<<nn_pred_temp<<" = "<<temp<< std::endl;
            if(temp>=dtmCriticalTemperature+tolerance){
                std::cout << "[Scheduler][DynThreadMapping]: Predicted Temp for core "<<coreCounter<<" ("<<temp<<"C) is greater than target" << std::endl;
                return false;
            }
        }
    }
    return true;
}


bool DynThreadMapping::throttle() {
    if (performanceCounters->getPeakTemperature() > dtmCriticalTemperature) {
        if (!in_throttle_mode) {
            std::cout << "[Scheduler][DynThreadMapping]: detected thermal violation" <<std::endl;
        }
        in_throttle_mode = true;
    } else if (performanceCounters->getPeakTemperature() < dtmRecoveredTemperature) {
        if (in_throttle_mode) {
            std::cout << "[Scheduler][DynThreadMapping]: thermal violation ended" << std::endl;
        }
        in_throttle_mode = false;
    }
    return in_throttle_mode;
}

double DynThreadMapping::calc_temperature(double current_temp_c, double equilibrium_temp_c, double interval_ms, double tau_ms){
    // Exponential decay factor
    double alpha = 1.0 - std::exp(-interval_ms / tau_ms);
    // First-order thermal response
    return current_temp_c + alpha * (equilibrium_temp_c - current_temp_c);
}