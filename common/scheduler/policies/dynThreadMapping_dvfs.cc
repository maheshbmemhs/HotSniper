#include "dynThreadMapping_dvfs.h"
#include "config.hpp"
#include "simulator.h"
#include <algorithm>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <sstream>

DynThreadMapping_dvfs::DynThreadMapping_dvfs(const PerformanceCounters *performanceCounters, 
                                   int coreRows, 
                                   int coreColumns, 
                                   std::string profile_path,
                                   std::string thermal_model_path,
                                   float temperature_constraint_,
                                   std::vector<float> core_states_,
                                   float dtmCriticalTemperature, 
                                   float dtmRecoveredTemperature):
performanceCounters(performanceCounters),
coreRows(coreRows),
coreColumns(coreColumns),
temperature_constraint(temperature_constraint_),
core_states(core_states_),
pred(profile_path),
thermal_model(thermal_model_path),
dtmCriticalTemperature(dtmCriticalTemperature),
dtmRecoveredTemperature(dtmRecoveredTemperature) {
    std::string output_dir = "benchmarks";
    experiment_name = "unknown";
    const char *benchmarks_root = std::getenv("BENCHMARKS_ROOT");
    const char *sniper_root = std::getenv("SNIPER_ROOT");
    if (benchmarks_root != NULL) {
        output_dir = std::string(benchmarks_root);
    } else if (sniper_root != NULL) {
        output_dir = std::string(sniper_root) + "/benchmarks";
    }

    try {
        experiment_name = std::string(Sim()->getCfg()->getString("traceinput/benchmarks").c_str());
    } catch (...) {
        std::cerr << "[Scheduler][DynThreadMapping_dvfs]: Could not read experiment name from config" << std::endl;
    }

    bool debug_thermal_model = false;
    try {
        const String debug_key = "scheduler/open/dvfs/DynThreadMapping_dvfs/debug";
        if (Sim()->getCfg()->hasKey(debug_key)) {
            debug_thermal_model = Sim()->getCfg()->getBool(debug_key);
        }
    } catch (...) {
    }
    thermal_model.setDebug(debug_thermal_model);

    std::string sample_log_path = output_dir;
    if (!sample_log_path.empty() && sample_log_path[sample_log_path.size() - 1] != '/') {
        sample_log_path += "/";
    }
    sample_log_path += "DynThreadMappingRandomSample.csv";
    std::ifstream existing_sample_log(sample_log_path.c_str(), std::ios::in | std::ios::ate);
    bool write_header = !existing_sample_log.good() || existing_sample_log.tellg() == std::streampos(0);

    sample_log.open(sample_log_path.c_str(), std::ios::out | std::ios::app);
    if (sample_log) {
        if (write_header) {
            sample_log << "experiment,cycle,"
                       << "start_peak_temp_c,end_peak_temp_c,"
                       << "old_frequencies_mhz,new_frequencies_mhz,active_cores,"
                       << "start_core_temps_c,end_core_temps_c,"
                       << "start_core_powers_w,end_core_powers_w,"
                       << "start_core_utilizations,end_core_utilizations,"
                       << "start_core_cpis,end_core_cpis,"
                       << "start_core_rel_nuca_cpis,end_core_rel_nuca_cpis,"
                       << "start_core_ips,end_core_ips\n";
        }
    } else {
        std::cerr << "[Scheduler][DynThreadMapping_dvfs]: Failed to open random sample log: "
                  << sample_log_path << std::endl;
    }

    std::string s_core_states = "[";
    for(float f:core_states){
        s_core_states+=std::to_string(f)+", ";
    }
    std::cout << "[Scheduler][DynThreadMapping_dvfs]: Initializing with: "
    << "predicted temperature constraint: "<< temperature_constraint_ << " C"
    << "\n profile path: " << profile_path
    << "\n thermal model path: " << thermal_model_path
    << "\n real critical temp: " << dtmCriticalTemperature
    << "\n real recovery temp: " << dtmRecoveredTemperature
    << "\n states: " << s_core_states << "]"
    << "\n random sample log: " << sample_log_path
    << std::endl;
}

std::vector<int> DynThreadMapping_dvfs::getFrequencies(const std::vector<int> &oldFrequencies, const std::vector<bool> &activeCores) {
    double current_peak_temp = performanceCounters ? performanceCounters->getPeakTemperature() : -1.0;
    std::vector<double> current_core_temps = getCoreTemperatures();
    std::vector<double> current_core_powers = getCorePowers();
    std::vector<double> current_core_utils = getCoreUtilizations();
    std::vector<double> current_core_cpis = getCoreCpis();
    std::vector<double> current_core_rel_nuca_cpis = getCoreRelNucaCpis();
    std::vector<double> current_core_ips = getCoreIps();
    ThermalModelSnapshot thermalSnapshot;
    thermalSnapshot.peakTemperature = current_peak_temp;
    thermalSnapshot.coreTemperatures = current_core_temps;
    thermalSnapshot.corePowers = current_core_powers;
    thermalSnapshot.coreUtilizations = current_core_utils;
    thermalSnapshot.coreCpis = current_core_cpis;
    thermalSnapshot.coreRelNucaCpis = current_core_rel_nuca_cpis;
    thermalSnapshot.coreIps = current_core_ips;

    std::vector<double> current_core_ips_gips;
    current_core_ips_gips.reserve(current_core_ips.size());
    for (double ips : current_core_ips) {
        current_core_ips_gips.push_back(ips > 0.0 ? ips / 1e9 : ips);
    }

    std::cout << "[Scheduler][DynThreadMapping_dvfs]: current core temperatures C: "
              << joinTemperatures(current_core_temps) << std::endl;
    std::cout << "[Scheduler][DynThreadMapping_dvfs]: current core IPS GIPS: "
              << joinDoubles(current_core_ips_gips) << std::endl;

    logCompletedCycle(
        current_peak_temp,
        current_core_temps,
        current_core_powers,
        current_core_utils,
        current_core_cpis,
        current_core_rel_nuca_cpis,
        current_core_ips);

    const unsigned int numCores = coreRows * coreColumns;

    if (throttle()) {
        std::vector<int> minFrequencies(numCores, static_cast<int>(std::round(core_states[0] * 1000.0f)));
        std::cout << "[Scheduler][DynThreadMapping_dvfs]: in throttle mode -> return min. frequencies" << std::endl;
        rememberCycle(
            current_peak_temp,
            current_core_temps,
            current_core_powers,
            current_core_utils,
            current_core_cpis,
            current_core_rel_nuca_cpis,
            current_core_ips,
            oldFrequencies,
            minFrequencies,
            activeCores);
        return minFrequencies;
    }

    const bool hasInvalidCurrentIps = hasInvalidIps(current_core_ips);
    const bool currentPhaseIsTransition = hasTransitionPhase(current_core_utils);

    if (hasInvalidCurrentIps || currentPhaseIsTransition) {
        std::vector<int> fixedFrequencies(numCores, 2000);
        std::cout << "[Scheduler][DynThreadMapping_dvfs]: "
                  << (hasInvalidCurrentIps ? "invalid current IPS detected" : "transition phase detected")
                  << " -> return fixed 2000 MHz for all cores" << std::endl;
        rememberCycle(
            current_peak_temp,
            current_core_temps,
            current_core_powers,
            current_core_utils,
            current_core_cpis,
            current_core_rel_nuca_cpis,
            current_core_ips,
            oldFrequencies,
            fixedFrequencies,
            activeCores);
        return fixedFrequencies;
    }

    if (thermal_model.getNumCores() != numCores) {
        std::cerr << "[Scheduler][DynThreadMapping_dvfs]: thermal model core count "
                  << thermal_model.getNumCores() << " does not match system core count "
                  << numCores << std::endl;
        return oldFrequencies;
    }

    const double minContributingIps = 1e-2;
    std::vector<NeighborPrediction::PredictionMap> predictions(numCores);
    std::vector<bool> ipsContributingCores(numCores, false);
    for (unsigned int core = 0; core < numCores; core++) {
        const float current_state = oldFrequencies.at(core) / 1000.0f;
        const float current_ips = getMeasuredIPSBillions(core);
        ipsContributingCores[core] = (core != 0 && current_ips >= minContributingIps);
        predictions[core] = pred.getNearestBenchmark(current_state, current_ips);
    }

    Candidate bestFeasible = findBestFeasibleCandidate(
        predictions,
        ipsContributingCores,
        thermalSnapshot,
        oldFrequencies,
        activeCores);

    Candidate chosen;
    if (bestFeasible.valid) {
        chosen = bestFeasible;
        std::cout << "[Scheduler][DynThreadMapping_dvfs]: selected max-IPS profile+ML frequencies under predicted temperature constraint"
                  << " predicted_ips=" << std::fixed << std::setprecision(3) << chosen.totalIps
                  << " predicted_peak_temp=" << std::fixed << std::setprecision(2) << chosen.peakTemperature
                  << " C predicted_temp_constraint=" << std::fixed << std::setprecision(2) << temperature_constraint
                  << " C predicted_power=" << std::fixed << std::setprecision(3) << chosen.totalPower
                  << " W" << std::endl;
    } else {
        chosen.valid = true;
        chosen.frequencies.assign(numCores, static_cast<int>(std::round(core_states[0] * 1000.0f)));
        std::cout << "[Scheduler][DynThreadMapping_dvfs]: no candidate satisfied predicted temperature constraint "
                  << temperature_constraint << " C; "
                  << "returning min. frequencies" << std::endl;
    }

    for(unsigned int i = 0; i < numCores; i++){
        std::cout << "[Scheduler][DynThreadMapping_dvfs]: core "<< i
                  << ": selected freq "<< chosen.frequencies[i] << " MHz" << std::endl;
    }

    rememberCycle(
        current_peak_temp,
        current_core_temps,
        current_core_powers,
        current_core_utils,
        current_core_cpis,
        current_core_rel_nuca_cpis,
        current_core_ips,
        oldFrequencies,
        chosen.frequencies,
        activeCores);
    return chosen.frequencies;
}

std::vector<double> DynThreadMapping_dvfs::getCoreTemperatures() const {
    std::vector<double> temperatures;
    for (unsigned int i = 0; i < coreRows * coreColumns; i++) {
        temperatures.push_back(performanceCounters ? performanceCounters->getTemperatureOfCore(i) : -1.0);
    }
    return temperatures;
}

std::vector<double> DynThreadMapping_dvfs::getCorePowers() const {
    std::vector<double> powers;
    for (unsigned int i = 0; i < coreRows * coreColumns; i++) {
        powers.push_back(performanceCounters ? performanceCounters->getPowerOfCore(i) : -1.0);
    }
    return powers;
}

std::vector<double> DynThreadMapping_dvfs::getCoreUtilizations() const {
    std::vector<double> utilizations;
    for (unsigned int i = 0; i < coreRows * coreColumns; i++) {
        utilizations.push_back(performanceCounters ? performanceCounters->getUtilizationOfCore(i) : -1.0);
    }
    return utilizations;
}

std::vector<double> DynThreadMapping_dvfs::getCoreCpis() const {
    std::vector<double> cpis;
    for (unsigned int i = 0; i < coreRows * coreColumns; i++) {
        cpis.push_back(performanceCounters ? performanceCounters->getCPIOfCore(i) : -1.0);
    }
    return cpis;
}

std::vector<double> DynThreadMapping_dvfs::getCoreRelNucaCpis() const {
    std::vector<double> rel_nuca_cpis;
    for (unsigned int i = 0; i < coreRows * coreColumns; i++) {
        rel_nuca_cpis.push_back(performanceCounters ? performanceCounters->getRelNUCACPIOfCore(i) : -1.0);
    }
    return rel_nuca_cpis;
}

std::vector<double> DynThreadMapping_dvfs::getCoreIps() const {
    std::vector<double> ips;
    for (unsigned int i = 0; i < coreRows * coreColumns; i++) {
        ips.push_back(performanceCounters ? performanceCounters->getIPSOfCore(i) : -1.0);
    }
    return ips;
}

void DynThreadMapping_dvfs::rememberCycle(
    double start_peak_temp,
    const std::vector<double>& start_core_temps,
    const std::vector<double>& start_core_powers,
    const std::vector<double>& start_core_utils,
    const std::vector<double>& start_core_cpis,
    const std::vector<double>& start_core_rel_nuca_cpis,
    const std::vector<double>& start_core_ips,
    const std::vector<int>& old_frequencies,
    const std::vector<int>& frequencies,
    const std::vector<bool>& active_cores) {
    pending_cycle_id = next_cycle_id++;
    pending_start_peak_temp = start_peak_temp;
    pending_start_core_temps = start_core_temps;
    pending_start_core_powers = start_core_powers;
    pending_start_core_utils = start_core_utils;
    pending_start_core_cpis = start_core_cpis;
    pending_start_core_rel_nuca_cpis = start_core_rel_nuca_cpis;
    pending_start_core_ips = start_core_ips;
    pending_old_frequencies = old_frequencies;
    pending_frequencies = frequencies;
    pending_active_cores = active_cores;
    has_pending_cycle = true;
}

void DynThreadMapping_dvfs::logCompletedCycle(
    double end_peak_temp,
    const std::vector<double>& end_core_temps,
    const std::vector<double>& end_core_powers,
    const std::vector<double>& end_core_utils,
    const std::vector<double>& end_core_cpis,
    const std::vector<double>& end_core_rel_nuca_cpis,
    const std::vector<double>& end_core_ips) {
    if (!has_pending_cycle || !sample_log) {
        return;
    }

    sample_log << csvEscape(experiment_name) << ","
               << pending_cycle_id << ","
               << std::fixed << std::setprecision(3) << pending_start_peak_temp << ","
               << std::fixed << std::setprecision(3) << end_peak_temp << ","
               << csvEscape(joinFrequencies(pending_old_frequencies)) << ","
               << csvEscape(joinFrequencies(pending_frequencies)) << ","
               << csvEscape(joinBools(pending_active_cores)) << ","
               << csvEscape(joinTemperatures(pending_start_core_temps)) << ","
               << csvEscape(joinTemperatures(end_core_temps)) << ","
               << csvEscape(joinDoubles(pending_start_core_powers)) << ","
               << csvEscape(joinDoubles(end_core_powers)) << ","
               << csvEscape(joinDoubles(pending_start_core_utils)) << ","
               << csvEscape(joinDoubles(end_core_utils)) << ","
               << csvEscape(joinDoubles(pending_start_core_cpis)) << ","
               << csvEscape(joinDoubles(end_core_cpis)) << ","
               << csvEscape(joinDoubles(pending_start_core_rel_nuca_cpis)) << ","
               << csvEscape(joinDoubles(end_core_rel_nuca_cpis)) << ","
               << csvEscape(joinDoubles(pending_start_core_ips)) << ","
               << csvEscape(joinDoubles(end_core_ips)) << "\n";
    sample_log.flush();
    has_pending_cycle = false;
}

std::string DynThreadMapping_dvfs::joinFrequencies(const std::vector<int>& values) const {
    std::ostringstream out;
    for (size_t i = 0; i < values.size(); i++) {
        if (i != 0) {
            out << ";";
        }
        out << values[i];
    }
    return out.str();
}

std::string DynThreadMapping_dvfs::joinBools(const std::vector<bool>& values) const {
    std::ostringstream out;
    for (size_t i = 0; i < values.size(); i++) {
        if (i != 0) {
            out << ";";
        }
        out << (values[i] ? 1 : 0);
    }
    return out.str();
}

std::string DynThreadMapping_dvfs::joinTemperatures(const std::vector<double>& values) const {
    return joinDoubles(values);
}

std::string DynThreadMapping_dvfs::joinDoubles(const std::vector<double>& values) const {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < values.size(); i++) {
        if (i != 0) {
            out << ";";
        }
        out << values[i];
    }
    return out.str();
}

std::string DynThreadMapping_dvfs::csvEscape(const std::string& value) const {
    std::string escaped = "\"";
    for (char c : value) {
        if (c == '"') {
            escaped += "\"\"";
        } else {
            escaped += c;
        }
    }
    escaped += "\"";
    return escaped;
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

bool DynThreadMapping_dvfs::hasInvalidIps(const std::vector<double>& ips) const {
    for (double value : ips) {
        if (value < 0.0) {
            return true;
        }
    }
    return false;
}

bool DynThreadMapping_dvfs::hasTransitionPhase(const std::vector<double>& utilizations) const {
    for (double util : utilizations) {
        if (util >= 0.01 && util <= 0.30) {
            return true;
        }
    }
    return false;
}

DynThreadMapping_dvfs::Candidate DynThreadMapping_dvfs::findBestFeasibleCandidate(
    const std::vector<NeighborPrediction::PredictionMap>& predictions,
    const std::vector<bool>& ips_contributing_cores,
    const ThermalModelSnapshot& snapshot,
    const std::vector<int>& old_frequencies,
    const std::vector<bool>& active_cores) {
    Candidate bestFeasible;
    std::vector<int> stateIndices(coreRows * coreColumns, 0);

    enumerateCandidateStates(
        0,
        stateIndices,
        predictions,
        ips_contributing_cores,
        snapshot,
        old_frequencies,
        active_cores,
        bestFeasible);

    return bestFeasible;
}

void DynThreadMapping_dvfs::enumerateCandidateStates(
    unsigned int core,
    std::vector<int>& state_indices,
    const std::vector<NeighborPrediction::PredictionMap>& predictions,
    const std::vector<bool>& ips_contributing_cores,
    const ThermalModelSnapshot& snapshot,
    const std::vector<int>& old_frequencies,
    const std::vector<bool>& active_cores,
    Candidate& best_feasible) {
    const unsigned int numCores = coreRows * coreColumns;
    if (core == numCores) {
        const Candidate candidate = evaluateCandidate(
            state_indices,
            predictions,
            ips_contributing_cores,
            snapshot,
            old_frequencies,
            active_cores);
        if (!candidate.valid) {
            return;
        }

        const bool predictedTempSafe = candidate.peakTemperature <= temperature_constraint;
        if (predictedTempSafe && isBetterCandidate(candidate, best_feasible)) {
            best_feasible = candidate;
        }
        return;
    }

    for (unsigned int state = 0; state < core_states.size(); state++) {
        state_indices[core] = state;
        enumerateCandidateStates(
            core + 1,
            state_indices,
            predictions,
            ips_contributing_cores,
            snapshot,
            old_frequencies,
            active_cores,
            best_feasible);
    }
}

DynThreadMapping_dvfs::Candidate DynThreadMapping_dvfs::evaluateCandidate(
    const std::vector<int>& state_indices,
    const std::vector<NeighborPrediction::PredictionMap>& predictions,
    const std::vector<bool>& ips_contributing_cores,
    const ThermalModelSnapshot& snapshot,
    const std::vector<int>& old_frequencies,
    const std::vector<bool>& active_cores) {
    const unsigned int numCores = coreRows * coreColumns;
    Candidate candidate;
    candidate.valid = true;
    candidate.frequencies.assign(numCores, static_cast<int>(std::round(core_states[0] * 1000.0f)));

    for (unsigned int core = 0; core < numCores; core++) {
        const int stateIndex = state_indices.at(core);
        const float freqGhz = core_states.at(stateIndex);
        candidate.frequencies[core] = static_cast<int>(std::round(freqGhz * 1000.0f));

        auto status = predictions[core].find(freqGhz);
        if (status == predictions[core].end()) {
            candidate.valid = false;
            return candidate;
        }

        if (ips_contributing_cores.at(core)) {
            candidate.totalIps += status->second.ips;
        }
        candidate.totalPower += status->second.power;
    }

    if (!predictCandidatePeakTemperature(
            snapshot,
            old_frequencies,
            candidate.frequencies,
            active_cores,
            candidate.peakTemperature)) {
        candidate.valid = false;
        return candidate;
    }
    return candidate;
}

bool DynThreadMapping_dvfs::isBetterCandidate(const Candidate& candidate, const Candidate& best) const {
    if (!best.valid) {
        return true;
    }
    const bool candidateHasNoIps = candidate.totalIps <= 0.0;
    const bool bestHasNoIps = best.totalIps <= 0.0;
    if (candidateHasNoIps && bestHasNoIps) {
        if (candidate.frequencies.at(0) != best.frequencies.at(0)) {
            return candidate.frequencies.at(0) > best.frequencies.at(0);
        }
    }
    if (candidate.totalIps != best.totalIps) {
        return candidate.totalIps > best.totalIps;
    }
    if (candidate.peakTemperature != best.peakTemperature) {
        return candidate.peakTemperature < best.peakTemperature;
    }
    return candidate.totalPower < best.totalPower;
}

ThermalRegressionModel::Inputs DynThreadMapping_dvfs::buildThermalModelInputs(
    const ThermalModelSnapshot& snapshot,
    const std::vector<int>& old_frequencies,
    const std::vector<int>& candidate_frequencies,
    const std::vector<bool>& active_cores) const {
    const unsigned int numCores = coreRows * coreColumns;
    ThermalRegressionModel::Inputs inputs;
    inputs.oldFrequenciesMhz.reserve(numCores);
    inputs.newFrequenciesMhz.reserve(numCores);
    inputs.activeCores.reserve(numCores);

    for (unsigned int core = 0; core < numCores; core++) {
        inputs.oldFrequenciesMhz.push_back(old_frequencies.at(core));
        inputs.newFrequenciesMhz.push_back(candidate_frequencies.at(core));
        inputs.activeCores.push_back(active_cores.at(core) ? 1.0 : 0.0);
    }

    inputs.startTempsC = snapshot.coreTemperatures;
    inputs.startPowersW = snapshot.corePowers;
    inputs.startUtilizations = snapshot.coreUtilizations;
    inputs.startCpis = snapshot.coreCpis;
    inputs.startRelNucaCpis = snapshot.coreRelNucaCpis;
    inputs.startIps = snapshot.coreIps;
    inputs.startPeakTempC = snapshot.peakTemperature;
    return inputs;
}

bool DynThreadMapping_dvfs::predictCandidatePeakTemperature(
    const ThermalModelSnapshot& snapshot,
    const std::vector<int>& old_frequencies,
    const std::vector<int>& candidate_frequencies,
    const std::vector<bool>& active_cores,
    double& predicted_peak_temperature) {
    const unsigned int numCores = coreRows * coreColumns;
    const ThermalRegressionModel::Inputs inputs = buildThermalModelInputs(
        snapshot,
        old_frequencies,
        candidate_frequencies,
        active_cores);
    const std::vector<double> predictedTemps = thermal_model.predictEndTemperatures(inputs);
    if (!thermal_model.isEnabled() || predictedTemps.size() != numCores) {
        return false;
    }

    predicted_peak_temperature = *std::max_element(predictedTemps.begin(), predictedTemps.end());
    return true;
}

bool DynThreadMapping_dvfs::throttle() {
    if (performanceCounters == NULL) {
        return false;
    }

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
