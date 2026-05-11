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
core_states(core_states_),
dtmCriticalTemperature(dtmCriticalTemperature),
dtmRecoveredTemperature(dtmRecoveredTemperature) {
    std::random_device rd;
    random_engine.seed(rd());

    for (float state : core_states) {
        const int frequency_mhz = static_cast<int>(std::round(state * 1000.0f));
        if (frequency_mhz >= 1000 && frequency_mhz <= 4000) {
            random_frequency_choices.push_back(frequency_mhz);
        }
    }
    std::sort(random_frequency_choices.begin(), random_frequency_choices.end());
    random_frequency_choices.erase(
        std::unique(random_frequency_choices.begin(), random_frequency_choices.end()),
        random_frequency_choices.end());
    if (random_frequency_choices.empty()) {
        for (int frequency_mhz = 1000; frequency_mhz <= 4000; frequency_mhz += 500) {
            random_frequency_choices.push_back(frequency_mhz);
        }
    }

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
    << "\n random frequency choices MHz: [" << joinFrequencies(random_frequency_choices) << "]"
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
        const int minFrequency = random_frequency_choices.empty() ? 1000 : random_frequency_choices.front();
        std::vector<int> minFrequencies(numCores, minFrequency);
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

    std::cout << "[Scheduler][DynThreadMapping_dvfs]: Random DVFS sample at peak T="
              << std::fixed << std::setprecision(2) << current_peak_temp << " C" << std::endl;
    std::vector<int> sampledFrequencies(numCores, random_frequency_choices.front());
    std::uniform_int_distribution<size_t> pick(0, random_frequency_choices.size() - 1);
    for(unsigned int i = 0; i < numCores; i++){
        sampledFrequencies[i] = random_frequency_choices[pick(random_engine)];
        std::cout << "[Scheduler][DynThreadMapping_dvfs]: core "<< i
                  << ": random freq "<< sampledFrequencies[i] << " MHz" << std::endl;
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
        sampledFrequencies,
        activeCores);
    return sampledFrequencies;
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
