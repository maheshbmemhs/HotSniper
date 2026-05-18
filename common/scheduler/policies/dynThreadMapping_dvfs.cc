#include "dynThreadMapping.h"
#include <cmath>
#include <cctype>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <algorithm>
#include <sstream>

namespace {

double readCounterValue(const std::function<double()> &reader, double fallback)
{
    try {
        double value = reader();
        if (std::isfinite(value) && value >= 0.0) {
            return value;
        }
    } catch (...) {
    }
    return fallback;
}

double scaledForCandidateFrequency(double value, double source_freq_ghz, double target_freq_ghz)
{
    if (!std::isfinite(value) || value < 0.0) {
        value = 0.0;
    }
    if (!std::isfinite(source_freq_ghz) || source_freq_ghz <= 0.0 ||
        !std::isfinite(target_freq_ghz) || target_freq_ghz < 0.0) {
        return value;
    }
    const double ratio = std::min(4.0, std::max(0.0, target_freq_ghz / source_freq_ghz));
    return value * ratio;
}

std::string trim(const std::string &value)
{
    const std::string whitespace = " \t\n\r";
    const size_t begin = value.find_first_not_of(whitespace);
    if (begin == std::string::npos) {
        return "";
    }
    const size_t end = value.find_last_not_of(whitespace);
    return value.substr(begin, end - begin + 1);
}

std::string normalizeThermalModelKey(const std::string &value)
{
    std::string key = trim(value);
    std::transform(key.begin(), key.end(), key.begin(), ::tolower);
    return key;
}

bool hasThermalModelMappingSyntax(const std::string &value)
{
    return value.find('=') != std::string::npos;
}

std::string benchmarkKeyFromTaskName(const std::string &taskName)
{
    const size_t firstDash = taskName.find('-');
    if (firstDash == std::string::npos) {
        return normalizeThermalModelKey(taskName);
    }

    const size_t secondDash = taskName.find('-', firstDash + 1);
    if (secondDash == std::string::npos) {
        return normalizeThermalModelKey(taskName.substr(firstDash + 1));
    }

    return normalizeThermalModelKey(taskName.substr(firstDash + 1, secondDash - firstDash - 1));
}

std::string suiteBenchmarkKeyFromTaskName(const std::string &taskName)
{
    const size_t firstDash = taskName.find('-');
    if (firstDash == std::string::npos) {
        return normalizeThermalModelKey(taskName);
    }

    const size_t secondDash = taskName.find('-', firstDash + 1);
    if (secondDash == std::string::npos) {
        return normalizeThermalModelKey(taskName);
    }

    return normalizeThermalModelKey(taskName.substr(0, secondDash));
}

}

DynThreadMapping::DynThreadMapping(const PerformanceCounters *performanceCounters, 
                                   int coreRows, 
                                   int coreColumns, 
                                   std::string profile_path,
                                   std::string thermal_model_path,
                                   bool thermal_model_debug,
                                   float predictionTemperatureBar_,
                                   float predictionSafetyMargin_,
                                   float migrationUtilizationDeltaThreshold_,
                                   float masterMigrationTemperatureDeltaThreshold_,
                                   unsigned long long masterMigrationCooldownNs_,
                                   float tolerance_,
                                   float dvfs_interval,
                                   std::vector<float> core_states_,
                                   float dtmCriticalTemperature, 
                                   float dtmRecoveredTemperature,
                                   bool sampleExplorationEnabled_,
                                   float sampleTargetMinTemperature_,
                                   float sampleTargetMaxTemperature_,
                                   float sampleMigrationProbability_,
                                   unsigned int sampleRandomSeed_): 
performanceCounters(performanceCounters),
coreRows(coreRows),
coreColumns(coreColumns),
core_states(core_states_),
pred(profile_path),
thermal_model(),
dtmCriticalTemperature(dtmCriticalTemperature),
dtmRecoveredTemperature(dtmRecoveredTemperature),
predictionTemperatureBar(predictionTemperatureBar_),
predictionSafetyMargin(std::max(0.0f, predictionSafetyMargin_)),
migrationUtilizationDeltaThreshold(std::max(0.0f, migrationUtilizationDeltaThreshold_)),
masterMigrationTemperatureDeltaThreshold(std::max(0.0f, masterMigrationTemperatureDeltaThreshold_)),
masterMigrationCooldownNs(masterMigrationCooldownNs_),
sampleExplorationEnabled(sampleExplorationEnabled_),
sampleTargetMinTemperature(std::min(sampleTargetMinTemperature_, sampleTargetMaxTemperature_)),
sampleTargetMaxTemperature(std::max(sampleTargetMinTemperature_, sampleTargetMaxTemperature_)),
sampleMigrationProbability(std::min(1.0f, std::max(0.0f, sampleMigrationProbability_))),
sampleRandomGenerator(sampleRandomSeed_),
tolerance(tolerance_),
dvfsInterval(dvfs_interval) {
    loadThermalModels(thermal_model_path, thermal_model_debug);
    std::string s_core_states = "[";
    for(float f:core_states){
        s_core_states+=std::to_string(f)+", ";
    }
    std::cout << "[Scheduler][DynThreadMapping]: Initializing with: "
    << "\n profile path: " << profile_path
    << "\n thermal model path: " << thermal_model_path
    << "\n critical temp: " << dtmCriticalTemperature
    << "\n recovery temp: " << dtmRecoveredTemperature
    << "\n prediction temp bar: " << predictionTemperatureBar
    << "\n prediction safety margin: " << predictionSafetyMargin
    << "\n effective prediction temp bar: " << effectivePredictionTemperatureBar()
    << "\n migration util delta threshold: " << migrationUtilizationDeltaThreshold
    << "\n master migration temp delta threshold: " << masterMigrationTemperatureDeltaThreshold
    << "\n master migration cooldown ns: " << masterMigrationCooldownNs
    << "\n sample exploration: " << (sampleExplorationEnabled ? "on" : "off")
    << "\n sample target temp range: [" << sampleTargetMinTemperature << ", " << sampleTargetMaxTemperature << "]"
    << "\n tolerance: " << tolerance
    << "\n dvfs interval: " << dvfsInterval
    << "\n states: " << s_core_states << "]"
	    << std::endl;
	}

DynThreadMapping::~DynThreadMapping()
{
}

void DynThreadMapping::setTaskNames(const std::vector<std::string> &taskNames_)
{
    taskNames = taskNames_;
}

void DynThreadMapping::setCurrentTaskIds(const std::vector<int> &taskIds)
{
    currentTaskIds = taskIds;
}

void DynThreadMapping::loadThermalModels(const std::string &thermal_model_path, bool thermal_model_debug)
{
    thermal_models.clear();
    thermal_model = MLTemperaturePredictor();

    const std::string configuredPath = trim(thermal_model_path);
    if (configuredPath.empty()) {
        return;
    }

    if (!hasThermalModelMappingSyntax(configuredPath)) {
        thermal_model = MLTemperaturePredictor(configuredPath, thermal_model_debug);
        return;
    }

    std::stringstream entries(configuredPath);
    std::string entry;
    while (std::getline(entries, entry, ',')) {
        entry = trim(entry);
        if (entry.empty()) {
            continue;
        }

        const size_t separator = entry.find('=');
        if (separator == std::string::npos) {
            std::cerr << "[Scheduler][DynThreadMapping]: ignoring malformed thermal model mapping entry: "
                      << entry << std::endl;
            continue;
        }

        const std::string key = normalizeThermalModelKey(entry.substr(0, separator));
        const std::string path = trim(entry.substr(separator + 1));
        if (key.empty() || path.empty()) {
            std::cerr << "[Scheduler][DynThreadMapping]: ignoring malformed thermal model mapping entry: "
                      << entry << std::endl;
            continue;
        }

        thermal_models[key] = MLTemperaturePredictor(path, thermal_model_debug);
        std::cout << "[Scheduler][DynThreadMapping]: thermal model mapping "
                  << key << " -> " << path << std::endl;
    }
}

bool DynThreadMapping::hasLoadedThermalModel() const
{
    if (thermal_model.isLoaded()) {
        return true;
    }

    for (std::map<std::string, MLTemperaturePredictor>::const_iterator it = thermal_models.begin();
         it != thermal_models.end();
         ++it) {
        if (it->second.isLoaded()) {
            return true;
        }
    }

    return false;
}

const MLTemperaturePredictor &DynThreadMapping::getThermalModelForTaskId(int taskId) const
{
    if (taskId >= 0 && taskId < static_cast<int>(taskNames.size())) {
        const std::string taskName = taskNames[taskId];
        std::vector<std::string> keys;
        keys.push_back(normalizeThermalModelKey(taskName));
        keys.push_back(suiteBenchmarkKeyFromTaskName(taskName));
        keys.push_back(benchmarkKeyFromTaskName(taskName));

        for (size_t i = 0; i < keys.size(); ++i) {
            std::map<std::string, MLTemperaturePredictor>::const_iterator it = thermal_models.find(keys[i]);
            if (it != thermal_models.end()) {
                return it->second;
            }
        }
    }

    return thermal_model;
}

const MLTemperaturePredictor &DynThreadMapping::getThermalModelForCore(unsigned int coreId) const
{
    if (coreId < currentTaskIds.size()) {
        return getThermalModelForTaskId(currentTaskIds[coreId]);
    }

    return thermal_model;
}

void DynThreadMapping::clearLastTemperaturePrediction()
{
    hasLastPredictedTemperatures = false;
    lastPredictedTemperatures.clear();
    lastPredictedFrequenciesMhz.clear();
}

void DynThreadMapping::setLastTemperaturePrediction(const std::vector<double> &predictedTemps,
                                                    const std::vector<int> &frequenciesMhz)
{
    hasLastPredictedTemperatures = !predictedTemps.empty();
    lastPredictedTemperatures = predictedTemps;
    lastPredictedFrequenciesMhz = frequenciesMhz;
}

float DynThreadMapping::effectivePredictionTemperatureBar() const
{
    return std::max(0.0f, predictionTemperatureBar - predictionSafetyMargin);
}
	
	std::vector<int> DynThreadMapping::getFrequencies(const std::vector<int> &oldFrequencies, const std::vector<bool> &activeCores) {
	    if (throttle()) {
	        clearLastTemperaturePrediction();
	        hasPendingCombinedFrequencies = false;
	        pendingCombinedFrequencies.clear();
        std::vector<int> minFrequencies(coreRows * coreColumns, core_states[0]*1000);
        std::cout << "[Scheduler][DynThreadMapping]: in throttle mode -> return min. frequencies" << std::endl;
        return minFrequencies;
    } else if(hasPendingCombinedFrequencies){
        hasPendingCombinedFrequencies = false;
        migrationOccured = false;
        std::cout << "[Scheduler][DynThreadMapping][Combined]: applying combined migration+DVFS frequencies" << std::endl;
        return pendingCombinedFrequencies;
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
	            clearLastTemperaturePrediction();
	            return newFrequencies;
	        }

        // Start all frequencies at the highest (max IPS)
        std::vector<int> newFrequencies(coreRows * coreColumns,core_states.back()*1000);

        // Tracks the index into core_state, used to increment between states
        std::vector<int> currentStatesIdx(coreRows * coreColumns,core_states.size()-1); // Initialization: start with highest possible frequency, last index in the core states
        
        // Get Predictions for each of the active cores
        std::vector<NeighborPrediction::PredictionMap> predictions;

        const size_t numCores = coreRows * coreColumns;
        std::vector<double> currentTemps(numCores, 0.0);
        std::vector<double> currentFreqs(numCores, 0.0);
        std::vector<double> ips(numCores, 0.0);
        std::vector<double> utilizations(numCores, 0.0);
        std::vector<double> cpis(numCores, 0.0);
        std::vector<double> relNucaCpis(numCores, 0.0);
        std::vector<double> powers(numCores, 0.0);

        for(size_t core = 0; core < numCores; core++){
            currentTemps[core] = readCounterValue([&] { return performanceCounters->getTemperatureOfCore(core); }, 0.0);
            powers[core] = readCounterValue([&] { return performanceCounters->getPowerOfCore(core); }, 0.0);
            utilizations[core] = readCounterValue([&] { return performanceCounters->getUtilizationOfCore(core); }, 0.0);
            cpis[core] = readCounterValue([&] { return performanceCounters->getCPIOfCore(core); }, 0.0);
            relNucaCpis[core] = readCounterValue([&] { return performanceCounters->getRelNUCACPIOfCore(core); }, 0.0);
            ips[core] = readCounterValue([&] { return performanceCounters->getIPSOfCore(core); }, 0.0);

            double old_freq_mhz = core < oldFrequencies.size() ? oldFrequencies[core] : -1.0;
            if (!std::isfinite(old_freq_mhz) || old_freq_mhz <= 0.0) {
                old_freq_mhz = readCounterValue([&] { return performanceCounters->getFreqOfCore(core); }, 0.0);
            }
            currentFreqs[core] = old_freq_mhz > 0.0 ? old_freq_mhz / 1000.0 : 0.0;
        }

        for(size_t i =0;i<(coreRows * coreColumns);i++){
            if(activeCores.at(i)){
                float current_ips = getMeasuredIPSBillions(i);
                NeighborPrediction::PredictionMap pm;
                // Thread is doing next to nothing, give it predictions that make it ignored in any moves
                if(current_ips <= 0.1){
                    std::cout << "[Scheduler][DynThreadMapping]: Warning IPS of core "<<i<<" is near zero, using ignored predictions"<< std::endl;
                    for(float state: core_states){
                        pm.emplace(state,NeighborPrediction::core_status{"none",0.0f,0.0f,0.0f,static_cast<float>(currentTemps[i]),0.0f});
                    }
                    currentStatesIdx[i]=0; // If we cant predict states accurately set core to lowest freq to be safe
                } else{
                    float current_state = float(oldFrequencies[i])/1000.0f; // Mhz to GHz
                    pm = pred.getNearestBenchmark(current_state,current_ips);
                }

                const MLTemperaturePredictor &thermalPredictor = getThermalModelForCore(i);
                if(thermalPredictor.isLoaded()){
                    for(float state: core_states){
                        auto status = pm.find(state);
                        if(status == pm.end()){
                            continue;
                        }

                        std::vector<double> candidateFreqs = currentFreqs;
                        if(i < candidateFreqs.size()){
                            candidateFreqs[i] = state;
                        }
	                        std::vector<double> expectedIps = ips;
	                        std::vector<double> expectedCpis = cpis;
	                        std::vector<double> expectedPowers = powers;
	                        std::vector<double> expectedUtilizations = utilizations;
	                        expectedIps[i] = scaledForCandidateFrequency(ips[i], currentFreqs[i], state);
	                        expectedCpis[i] = cpis[i];
	                        expectedPowers[i] = scaledForCandidateFrequency(powers[i], currentFreqs[i], state);

                        const double ml_temp = thermalPredictor.predictNextTemp(
                            i,
                            currentTemps,
                            currentFreqs,
                            candidateFreqs,
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

                        if(std::isfinite(ml_temp)){
                            status->second.temp = ml_temp;
                            static bool logged_ml_prediction = false;
                            if(!logged_ml_prediction){
                                std::cout << "[MLTemperaturePredictor] first prediction core=" << i
                                          << " cand_freq=" << state
                                          << " pred_temp=" << ml_temp << std::endl;
                                logged_ml_prediction = true;
                            }
                        } else {
                            std::cerr << "[Scheduler][DynThreadMapping]: Warning ML temperature prediction failed for core "
                                      << i << " state " << state << "GHz; using NeighborPrediction temperature" << std::endl;
                        }
                    }
                }
                predictions.push_back(pm);
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
	        std::vector<double> selectedPredictedTemps(coreRows * coreColumns, std::numeric_limits<double>::quiet_NaN());
	        for(size_t i=0;i<(coreRows * coreColumns);i++){
	            newFrequencies[i]=int(1000*core_states[currentStatesIdx[i]]);
	            double cur_temp = performanceCounters->getTemperatureOfCore(i);
	            double nn_pred_temp = std::numeric_limits<double>::quiet_NaN();
	            auto status = predictions[i].find(core_states[currentStatesIdx[i]]);
	            if(status != predictions[i].end()){
	                nn_pred_temp = status->second.temp;
	            }
	            selectedPredictedTemps[i] = nn_pred_temp;
	            std::cout << "[Scheduler][DynThreadMapping]: core "<< i << ": freq "<<core_states[currentStatesIdx[i]] 
	                      <<" temp "<<cur_temp<<" -> "<<nn_pred_temp<< std::endl;
	
	        }
	        setLastTemperaturePrediction(selectedPredictedTemps, newFrequencies);
	        return newFrequencies;
	    } else{
	        clearLastTemperaturePrediction();
	        migrationOccured = false;
	        return oldFrequencies;
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
            float temp_before = prediction[current_state].temp;

            // Potential IPS and Temp values of next state
            float new_state = core_states[currentStatesIdx[i]-1]; // next state is just current for core i minus 1
            float ips_after = prediction[new_state].ips;   
            float temp_after = prediction[new_state].temp;
            
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

    const float effectivePredictionBar = effectivePredictionTemperatureBar();
    for(size_t i = 0;i<currentStatesIdx.size(); i++){
        if(activeCores.at(i)){
            // Cant go any lower than this state
            if(!currentStatesIdx[i]){
                continue;
            }
            auto prediction = predictions[i];
            double pred_temp = prediction[core_states[currentStatesIdx[i]]].temp;
            while(currentStatesIdx[i] > 0 && pred_temp>=effectivePredictionBar){
                std::cerr<<"Lowering core " <<i<<" freq from " <<core_states[currentStatesIdx[i]]
                         << " GHz to " << core_states[currentStatesIdx[i]-1]
                         << " GHz because predicted temp " << pred_temp
                         << "C exceeds effective prediction bar " << effectivePredictionBar << "C" << std::endl;
                currentStatesIdx[i]-=1;
                float new_state = core_states[currentStatesIdx[i]];
                pred_temp = prediction[new_state].temp;
            }
            if(pred_temp>=effectivePredictionBar){
                std::cerr << "[Scheduler][DynThreadMapping]: Warning lowest frequency for core " << i
                          << " still predicts " << pred_temp
                          << "C, above effective prediction bar " << effectivePredictionBar << "C" << std::endl;
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
    const float effectivePredictionBar = effectivePredictionTemperatureBar();
    for (unsigned int coreCounter = 0; coreCounter < coreRows * coreColumns; coreCounter++) {
        if(activeCores.at(coreCounter)){
            auto prediction = predictions[coreCounter];
            float new_clock_speed = core_states[currentStateIdx[coreCounter]];
            float cur_temp = performanceCounters->getTemperatureOfCore(coreCounter);
            float nn_pred_temp = prediction.at(new_clock_speed).temp;
            double temp= nn_pred_temp;
            std::cout << "[Scheduler][DynThreadMapping]: core "<< coreCounter << ": freq "<<new_clock_speed
                      <<" temp "<<cur_temp<<" -> "<<temp<< std::endl;
            if(temp>=effectivePredictionBar){
                std::cout << "[Scheduler][DynThreadMapping]: Predicted Temp for core "<<coreCounter
                          <<" ("<<temp<<"C) is greater than effective prediction bar "
                          << effectivePredictionBar << "C" << std::endl;
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
