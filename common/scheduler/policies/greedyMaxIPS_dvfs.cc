#include "greedyMaxIPS.h"
#include <iomanip>
#include <iostream>
#include <queue>
#include <algorithm>

GreedyMaxIPS::GreedyMaxIPS(const PerformanceCounters *performanceCounters, 
                                   int coreRows, 
                                   int coreColumns, 
                                   std::string thermal_model_path,
                                   bool dvfs_when_migration,
                                   float tolerance_,
                                   float dvfs_interval,
                                   std::vector<float> core_states_,
                                   float dtmCriticalTemperature, 
                                   float dtmRecoveredTemperature): 
performanceCounters(performanceCounters),
coreRows(coreRows),
coreColumns(coreColumns),
core_states(core_states_),
dtmCriticalTemperature(dtmCriticalTemperature),
dtmRecoveredTemperature(dtmRecoveredTemperature),
tolerance(tolerance_),
dvfsInterval(dvfs_interval),
thermal_model(thermal_model_path,true),
dvfs_when_migration(dvfs_when_migration){
    std::string s_core_states = "[";
    for(float f:core_states){
        s_core_states+=std::to_string(f)+", ";
    }
    std::cout << "[Scheduler][GreedyMaxIPS]: Initializing with: "
    << "\n Thermal model: " << thermal_model_path
    << "\n critical temp: " << dtmCriticalTemperature
    << "\n recovery temp: " << dtmRecoveredTemperature
    << "\n tolerance: " << tolerance
    << "\n dvfs interval: " << dvfsInterval
    << "\n states: " << s_core_states << "]"
    << std::endl;
    
}

std::vector<int> GreedyMaxIPS::getFrequencies(const std::vector<int> &oldFrequencies, const std::vector<bool> &activeCores) {
    if (throttle()) {
        std::vector<int> minFrequencies(coreRows * coreColumns, core_states[0]*1000);
        std::cout << "[Scheduler][GreedyMaxIPS]: in throttle mode -> return min. frequencies" << std::endl;
        return minFrequencies;
    } else if(!migrationOccured || dvfs_when_migration){
        std::cout << "[Scheduler][GreedyMaxIPS]: Starting DVFS"<< std::endl;
        std::cout << "[Scheduler][GreedyMaxIPS]: Active Cores: ";
        for(bool active:activeCores){
            std::cout<<active<<", ";
        }
        std::cout<<std::endl;

        // Non-Parallel Region
        if(std::count(activeCores.begin(), activeCores.end(), true)==1){
            std::cout << "[Scheduler][GreedyMaxIPS]: Non-Parallel region, setting inactive cores to lowest frequency and active core to 2.0 Ghz" << std::endl;
            std::vector<int> newFrequencies(coreRows * coreColumns,core_states[0]);
            newFrequencies[std::distance(std::begin(activeCores), std::find(activeCores.begin(), activeCores.end(),true))]=2000; // Set only active core to 2 GHz
            return newFrequencies;
        }

        // Start all frequencies at the highest (max IPS)
        std::vector<int> newFrequencies(coreRows * coreColumns,core_states.back()*1000);

        // Tracks the index into core_state, used to increment between states
        std::vector<int> currentStatesIdx(coreRows * coreColumns,core_states.size()-1); // Initialization: start with highest possible frequency, last index in the core states
        
        // Find best freq
        get_max_freq(activeCores,currentStatesIdx);
        std::cout << "[Scheduler][GreedyMaxIPS]: Final States "<< std::endl;

        // Update core frequency
        for(size_t i=0;i<(coreRows * coreColumns);i++){
            newFrequencies[i]=int(1000*core_states[currentStatesIdx[i]]);
            double cur_temp = performanceCounters->getTemperatureOfCore(i);
            std::cout << "[Scheduler][GreedyMaxIPS]: core "<< i << ": freq "<<core_states[currentStatesIdx[i]] 
                      <<" GHz, current temp "<<cur_temp<< std::endl;
        }
        return newFrequencies;
    } else{
        migrationOccured = false;
        return {oldFrequencies};
    }
}

GreedyMaxIPS::Move GreedyMaxIPS::get_best_move(const std::vector<NeighborPrediction::PredictionMap>& predictions,
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

void GreedyMaxIPS::get_max_freq(const std::vector<bool>& activeCores,
                                       std::vector<int>& currentStatesIdx){
    if(!thermal_model.isLoaded()){
        std::cerr<<"Model not loaded, using lowest freq" << std::endl;
        std::fill(currentStatesIdx.begin(),currentStatesIdx.end(),1);
    }
    const size_t numCores = coreRows * coreColumns;
    std::vector<double> currentTemps(numCores, 0.0);
    std::vector<double> currentFreqs(numCores, 0.0);
    std::vector<double> ips(numCores, 0.0);
    std::vector<double> utilizations(numCores, 0.0);
    std::vector<double> cpis(numCores, 0.0);
    std::vector<double> relNucaCpis(numCores, 0.0);
    std::vector<double> powers(numCores, 0.0);

    for(size_t core = 0; core < numCores; core++){
        currentTemps[core] = performanceCounters->getTemperatureOfCore(core);
        powers[core] = performanceCounters->getPowerOfCore(core);
        utilizations[core] = performanceCounters->getUtilizationOfCore(core);
        cpis[core] = performanceCounters->getCPIOfCore(core);
        relNucaCpis[core] = performanceCounters->getRelNUCACPIOfCore(core);
        ips[core] = performanceCounters->getIPSOfCore(core);
        double old_freq_mhz = performanceCounters->getFreqOfCore(core);
        currentFreqs[core] = old_freq_mhz > 0.0 ? old_freq_mhz / 1000.0 : 0.0;
    }
    std::vector<double> expectedIps = ips;
    std::vector<double> expectedCpis = cpis;
    std::vector<double> expectedPowers = powers;
    std::vector<double> expectedUtilizations = utilizations;
    //expectedIps[i] = scaledForCandidateFrequency(ips[i], currentFreqs[i], state);
    //expectedCpis[i] = cpis[i];
    //expectedPowers[i] = scaledForCandidateFrequency(powers[i], currentFreqs[i], state);
    for(size_t i = 0;i<currentStatesIdx.size(); i++){
        if(activeCores.at(i)){
            // Cant go any lower than this state
            if(!currentStatesIdx[i]){
                continue;
            }

            // Ignore low IPS cores
            if(ips[i]<=(0.1*1e9)){
                std::cout << "[Scheduler][GreedyMaxIPS][DVFS]: Warning IPS of core "<<i<<" is near zero, ignoring for predictions"<< std::endl;
                currentStatesIdx[i]=0;
                continue;
            }

            // Update candidate frequencies to match previous core changes (I.e < i)
            std::vector<double> candidateFreq = currentFreqs;
            for(size_t j = 0;j<=i;j++){
                candidateFreq[j]=core_states[currentStatesIdx[j]];
            }

            double pred_temp = thermal_model.predictNextTemp(i,
                            currentTemps,
                            currentFreqs,
                            candidateFreq,
                            ips,
                            utilizations,
                            cpis,
                            relNucaCpis,
                            powers,
                            activeCores,
	                            true,
	                            &expectedIps,
	                            &expectedCpis,
	                            &expectedPowers,
	                            &expectedUtilizations,
	                            &activeCores);
             
            // Keep lowering frequencies until we get a predicted temp lower than critical temp
            while(pred_temp>=dtmCriticalTemperature+tolerance){
                if(!currentStatesIdx[i]){ // Cant go lower than 0
                    break;
                }
                std::cerr<<"[Scheduler][GreedyMaxIPS][DVFS]: Predicted temp: "<<pred_temp<<" C. Lowering core " <<i<<" freq from " <<core_states[currentStatesIdx[i]] << " GHz to " << core_states[currentStatesIdx[i]-1] << " Ghz" << std::endl;
                currentStatesIdx[i]-=1;
                double new_state = core_states[currentStatesIdx[i]];
                candidateFreq[i]=new_state;
                pred_temp = thermal_model.predictNextTemp(i,
                            currentTemps,
                            currentFreqs,
                            candidateFreq,
                            ips,
                            utilizations,
                            cpis,
                            relNucaCpis,
                            powers,
                            activeCores,
	                            true,
	                            &expectedIps,
	                            &expectedCpis,
	                            &expectedPowers,
	                            &expectedUtilizations,
	                            &activeCores);
            }
            currentTemps[i]=pred_temp;
            ips[i]=expectedIps[i];
            cpis[i]=expectedCpis[i];
            powers[i]=expectedPowers[i];
            utilizations[i]=expectedUtilizations[i];

            std::cerr<<"[Scheduler][GreedyMaxIPS][DVFS]: Core "<<i<<" final predicted temp: "<< pred_temp << std::endl;

        }else{
            currentStatesIdx[i]=0; // Inactive core
        }
    }
}


double GreedyMaxIPS::getMeasuredIPSBillions(unsigned int coreId) {
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

// Old Not used
bool GreedyMaxIPS::checkConstraints(const std::vector<NeighborPrediction::PredictionMap>& predictions, 
                                             const std::vector<int>& currentStateIdx, 
                                             const std::vector<bool>& activeCores){
    for (unsigned int coreCounter = 0; coreCounter < coreRows * coreColumns; coreCounter++) {
        if(activeCores.at(coreCounter)){
            auto prediction = predictions[coreCounter];
            float new_clock_speed = core_states[currentStateIdx[coreCounter]];
            float cur_temp = performanceCounters->getTemperatureOfCore(coreCounter);
            float nn_pred_temp = prediction.at(new_clock_speed).temp;
            double temp= calc_temperature(cur_temp,nn_pred_temp,dvfsInterval,thermalInertia);
            std::cout << "[Scheduler][GreedyMaxIPS]: core "<< coreCounter << ": freq "<<new_clock_speed
                      <<" temp "<<cur_temp<<" -> "<<nn_pred_temp<<" = "<<temp<< std::endl;
            if(temp>=dtmCriticalTemperature+tolerance){
                std::cout << "[Scheduler][GreedyMaxIPS]: Predicted Temp for core "<<coreCounter<<" ("<<temp<<"C) is greater than target" << std::endl;
                return false;
            }
        }
    }
    return true;
}


bool GreedyMaxIPS::throttle() {
    if (performanceCounters->getPeakTemperature() > dtmCriticalTemperature) {
        if (!in_throttle_mode) {
            std::cout << "[Scheduler][GreedyMaxIPS]: detected thermal violation" <<std::endl;
        }
        in_throttle_mode = true;
    } else if (performanceCounters->getPeakTemperature() < dtmRecoveredTemperature) {
        if (in_throttle_mode) {
            std::cout << "[Scheduler][GreedyMaxIPS]: thermal violation ended" << std::endl;
        }
        in_throttle_mode = false;
    }
    return in_throttle_mode;
}

// Old Not used, used to predict temps with NN
double GreedyMaxIPS::calc_temperature(double current_temp_c, double equilibrium_temp_c, double interval_ms, double tau_ms){
    // Exponential decay factor
    double alpha = 1.0 - std::exp(-interval_ms / tau_ms);
    // First-order thermal response
    return current_temp_c + alpha * (equilibrium_temp_c - current_temp_c);
}