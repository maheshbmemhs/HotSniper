#include "dynThreadMapping_dvfs.h"
#include "config.hpp"
#include "simulator.h"
#include <algorithm>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <functional>
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
    std::string output_dir = ".";
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
        std::vector<int> minFrequencies(numCores, core_states[0]*1000);
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

    const bool hasInvalidCurrentIps = std::any_of(
        current_core_ips.begin(),
        current_core_ips.end(),
        [](double ips) { return ips < 0.0; });
    const bool isTransitionPhase = std::any_of(
        current_core_utils.begin(),
        current_core_utils.end(),
        [](double util) { return util >= 0.01 && util <= 0.30; });

    if (hasInvalidCurrentIps || isTransitionPhase) {
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

    struct Candidate {
        bool valid;
        std::vector<int> frequencies;
        double total_ips;
        double total_power;
        double peak_temperature;

        Candidate() : valid(false), total_ips(0.0), total_power(0.0), peak_temperature(0.0) {}
    };

    auto buildInputs = [&](const std::vector<int>& frequencies) -> ThermalRegressionModel::Inputs {
        ThermalRegressionModel::Inputs inputs;
        inputs.oldFrequenciesMhz.reserve(numCores);
        inputs.newFrequenciesMhz.reserve(numCores);
        inputs.activeCores.reserve(numCores);

        for (unsigned int core = 0; core < numCores; core++) {
            inputs.oldFrequenciesMhz.push_back(oldFrequencies.at(core));
            inputs.newFrequenciesMhz.push_back(frequencies.at(core));
            inputs.activeCores.push_back(activeCores.at(core) ? 1.0 : 0.0);
        }

        inputs.startTempsC = current_core_temps;
        inputs.startPowersW = current_core_powers;
        inputs.startUtilizations = current_core_utils;
        inputs.startCpis = current_core_cpis;
        inputs.startRelNucaCpis = current_core_rel_nuca_cpis;
        inputs.startIps = current_core_ips;
        inputs.startPeakTempC = current_peak_temp;
        return inputs;
    };

    auto evaluateCandidate = [&](const std::vector<int>& stateIndices) -> Candidate {
        Candidate candidate;
        candidate.valid = true;
        candidate.frequencies.assign(numCores, static_cast<int>(std::round(core_states[0] * 1000.0f)));

        for (unsigned int core = 0; core < numCores; core++) {
            const int stateIndex = stateIndices.at(core);
            const float freqGhz = core_states.at(stateIndex);
            candidate.frequencies[core] = static_cast<int>(std::round(freqGhz * 1000.0f));

            auto status = predictions[core].find(freqGhz);
            if (status == predictions[core].end()) {
                candidate.valid = false;
                return candidate;
            }

            if (ipsContributingCores.at(core)) {
                candidate.total_ips += status->second.ips;
            }
            candidate.total_power += status->second.power;
        }

        const std::vector<double> predictedTemps = thermal_model.predictEndTemperatures(buildInputs(candidate.frequencies));
        if (!thermal_model.isEnabled() || predictedTemps.size() != numCores) {
            candidate.valid = false;
            return candidate;
        }
        candidate.peak_temperature = *std::max_element(predictedTemps.begin(), predictedTemps.end());
        return candidate;
    };

    auto betterThroughput = [](const Candidate& candidate, const Candidate& best) -> bool {
        if (!best.valid) {
            return true;
        }
        const bool candidateHasNoIps = candidate.total_ips <= 0.0;
        const bool bestHasNoIps = best.total_ips <= 0.0;
        if (candidateHasNoIps && bestHasNoIps) {
            if (candidate.frequencies.at(0) != best.frequencies.at(0)) {
                return candidate.frequencies.at(0) > best.frequencies.at(0);
            }
        }
        if (candidate.total_ips != best.total_ips) {
            return candidate.total_ips > best.total_ips;
        }
        if (candidate.peak_temperature != best.peak_temperature) {
            return candidate.peak_temperature < best.peak_temperature;
        }
        return candidate.total_power < best.total_power;
    };

    Candidate bestFeasible;
    std::vector<int> stateIndices(numCores, 0);

    std::function<void(unsigned int)> enumerate = [&](unsigned int core) {
        if (core == numCores) {
            const Candidate candidate = evaluateCandidate(stateIndices);
            if (!candidate.valid) {
                return;
            }

            const bool predictedTempSafe = candidate.peak_temperature <= temperature_constraint;
            if (predictedTempSafe && betterThroughput(candidate, bestFeasible)) {
                bestFeasible = candidate;
            }
            return;
        }

        for (unsigned int state = 0; state < core_states.size(); state++) {
            stateIndices[core] = state;
            enumerate(core + 1);
        }
    };

    enumerate(0);

    Candidate chosen;
    if (bestFeasible.valid) {
        chosen = bestFeasible;
        std::cout << "[Scheduler][DynThreadMapping_dvfs]: selected max-IPS profile+ML frequencies under predicted temperature constraint"
                  << " predicted_ips=" << std::fixed << std::setprecision(3) << chosen.total_ips
                  << " predicted_peak_temp=" << std::fixed << std::setprecision(2) << chosen.peak_temperature
                  << " C predicted_temp_constraint=" << std::fixed << std::setprecision(2) << temperature_constraint
                  << " C predicted_power=" << std::fixed << std::setprecision(3) << chosen.total_power
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
