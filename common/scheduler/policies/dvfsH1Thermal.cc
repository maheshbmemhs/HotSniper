#include "dvfsH1Thermal.h"

#include "clock_skew_minimization_object.h"
#include "fixed_types.h"
#include "simulator.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

using namespace std;

DVFSH1Thermal::DVFSH1Thermal(const PerformanceCounters *performanceCounters,
                             int numberOfCores,
                             const vector<double> &enabledStates,
                             const vector<int> &frequencies,
	                             double targetIPS,
	                             const string &objective,
	                             double maxTemp,
	                             double thermalMargin,
	                             double powerBudget,
	                             double powerBudgetMargin,
	                             double perCorePowerGuard,
	                             const string &profileFile,
	                             const string &benchmarkHint,
	                             bool freezeMaster,
	                             bool debug)
    : performanceCounters(performanceCounters)
    , numberOfCores(numberOfCores)
    , enabledStates(enabledStates)
    , frequencies(frequencies)
	    , targetIPS(targetIPS)
	    , objective(objective)
	    , maxTemp(maxTemp)
	    , thermalMargin(thermalMargin)
	    , powerBudget(powerBudget)
	    , powerBudgetMargin(powerBudgetMargin)
	    , perCorePowerGuard(perCorePowerGuard)
	    , profileFile(profileFile)
	    , benchmarkHint(benchmarkHint)
	    , freezeMaster(freezeMaster)
	    , debug(debug)
	    , warnedFallback(false)
	    , warnedInvalidPowerBudget(false)
	{
	    if (this->frequencies.size() != this->enabledStates.size()) {
	        cout << "[DVFSH1Thermal][Warning]: frequency count does not match enabled state count." << endl;
	    }
	    if (objective != "thermal_max_ips" && objective != "power_budget_max_ips" && objective != "target_ips_min_power") {
	        cout << "[DVFSH1Thermal][Warning]: objective=" << objective
	             << " is unsupported for this policy; using thermal_max_ips behavior." << endl;
	    }
	
	    loadProfile();
	}

DVFSH1Thermal::Candidate::Candidate()
    : totalIPS(0.0)
    , totalPower(0.0)
    , maxPredTemp(0.0)
    , changedCores(0)
    , feasible(false)
    , valid(false)
{
}

int DVFSH1Thermal::stateKey(double state) const
{
    return (int)floor(state * 1000.0 + 0.5);
}

void DVFSH1Thermal::loadProfile()
{
    if (profileFile == "") {
        cout << "[DVFSH1Thermal][Warning]: no profile file configured; using measured fallback predictions." << endl;
        return;
    }

    ifstream file(profileFile.c_str());
    if (!file.good()) {
        cout << "[DVFSH1Thermal][Warning]: failed to open profile file " << profileFile
             << "; using measured fallback predictions." << endl;
        return;
    }

    string line;
    int rows = 0;
    while (getline(file, line)) {
        if (line.find_first_not_of(" \t\r\n") == string::npos) {
            continue;
        }
        if (line.at(0) == '#') {
            continue;
        }

        istringstream iss(line);
        string name;
        ProfileEntry entry;
        if (!(iss >> name >> entry.state >> entry.ips >> entry.cpi >> entry.temp >> entry.power)) {
            continue;
        }

        profile[name][stateKey(entry.state)] = entry;
        rows++;
    }

    int validBenchmarks = 0;
    for (ProfileMap::const_iterator it = profile.begin(); it != profile.end(); ++it) {
        if (hasAllEnabledStates(it->first)) {
            validBenchmarks++;
        }
    }

    cout << "[DVFSH1Thermal] loaded " << rows << " profile rows from " << profileFile
         << " (" << validBenchmarks << " benchmarks cover all enabled states)" << endl;
}

bool DVFSH1Thermal::hasAllEnabledStates(const string &benchmarkName) const
{
    ProfileMap::const_iterator benchmark = profile.find(benchmarkName);
    if (benchmark == profile.end()) {
        return false;
    }

    for (unsigned int state = 0; state < enabledStates.size(); state++) {
        if (benchmark->second.find(stateKey(enabledStates.at(state))) == benchmark->second.end()) {
            return false;
        }
    }

    return true;
}

string DVFSH1Thermal::findNearestBenchmark(double currentStateValue, double measuredIPS, double measuredPower) const
{
    if (benchmarkHint != "") {
        if (hasAllEnabledStates(benchmarkHint)) {
            return benchmarkHint;
        }

        size_t dash = benchmarkHint.find('-');
        if (dash != string::npos && dash + 1 < benchmarkHint.size()) {
            string withoutSuite = benchmarkHint.substr(dash + 1);
            if (hasAllEnabledStates(withoutSuite)) {
                return withoutSuite;
            }
        }
    }

    string nearestBenchmark;
    double nearestDistance = numeric_limits<double>::max();
    int currentStateKey = stateKey(currentStateValue);

    for (ProfileMap::const_iterator benchmark = profile.begin(); benchmark != profile.end(); ++benchmark) {
        if (!hasAllEnabledStates(benchmark->first)) {
            continue;
        }

        StateProfile::const_iterator currentState = benchmark->second.find(currentStateKey);
        if (currentState == benchmark->second.end()) {
            continue;
        }

        double ipsScale = max(max(fabs(currentState->second.ips), fabs(measuredIPS)), 1e-9);
        double distance = fabs(currentState->second.ips - measuredIPS) / ipsScale;
        if (measuredPower > 0.0 && currentState->second.power > 0.0) {
            double powerScale = max(max(fabs(currentState->second.power), fabs(measuredPower)), 1e-9);
            distance += fabs(currentState->second.power - measuredPower) / powerScale;
        }
        if (distance < nearestDistance) {
            nearestDistance = distance;
            nearestBenchmark = benchmark->first;
        }
    }

    return nearestBenchmark;
}

int DVFSH1Thermal::getStateForFrequency(int frequency) const
{
    if (frequencies.size() == 0) {
        return -1;
    }

    int bestState = 0;
    int bestDistance = abs(frequency - frequencies.at(0));
    for (unsigned int state = 1; state < frequencies.size(); state++) {
        int distance = abs(frequency - frequencies.at(state));
        if (distance < bestDistance) {
            bestState = state;
            bestDistance = distance;
        }
    }

    return bestState;
}

double DVFSH1Thermal::getMeasuredIPSBillions(unsigned int coreId, double currentStateValue) const
{
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

    try {
        double cpi = performanceCounters->getCPIOfCore(coreId);
        if (cpi > 0.0) {
            return currentStateValue / cpi;
        }
    } catch (...) {
    }

    return 0.0;
}

double DVFSH1Thermal::getMeasuredPower(unsigned int coreId) const
{
    if (performanceCounters == NULL) {
        return 0.0;
    }

    try {
        double power = performanceCounters->getPowerOfCore(coreId);
        if (power > 0.0) {
            return power;
        }
    } catch (...) {
    }

    return 0.0;
}

double DVFSH1Thermal::getMeasuredTemperature(unsigned int coreId) const
{
    if (performanceCounters == NULL) {
        return 0.0;
    }

    try {
        double temperature = performanceCounters->getTemperatureOfCore(coreId);
        if (temperature > 0.0) {
            return temperature;
        }
    } catch (...) {
    }

    return 0.0;
}

bool DVFSH1Thermal::isPowerBudgetObjective() const
{
    return objective == "power_budget_max_ips";
}

bool DVFSH1Thermal::isTargetIPSMinPowerObjective() const
{
    return objective == "target_ips_min_power";
}

bool DVFSH1Thermal::isMasterCore(const vector<int> &taskIds, const vector<int> &threadIds, unsigned int coreId) const
{
    if (!freezeMaster) {
        return false;
    }
    if (coreId >= taskIds.size() || coreId >= threadIds.size()) {
        return false;
    }

    return taskIds.at(coreId) != -1 && threadIds.at(coreId) == 0;
}

double DVFSH1Thermal::tempLimit() const
{
    return maxTemp - thermalMargin;
}

double DVFSH1Thermal::effectivePowerBudget() const
{
    return powerBudget * powerBudgetMargin;
}

void DVFSH1Thermal::buildPrediction(unsigned int coreId,
                                    int currentStateIndex,
                                    double measuredIPS,
                                    string &nearestBenchmark,
                                    vector<double> &predIPS,
                                    vector<double> &predPower,
                                    vector<double> &predTemp)
{
    predIPS.assign(enabledStates.size(), 0.0);
    predPower.assign(enabledStates.size(), 0.0);
    predTemp.assign(enabledStates.size(), 0.0);

    double currentStateValue = enabledStates.at(currentStateIndex);
    double measuredPower = getMeasuredPower(coreId);
    double measuredTemp = getMeasuredTemperature(coreId);

    nearestBenchmark = findNearestBenchmark(currentStateValue, measuredIPS, measuredPower);
    if (nearestBenchmark == "") {
        if (!warnedFallback) {
            cout << "[DVFSH1Thermal][Warning]: no valid nearest benchmark with all enabled states found; using conservative measured fallback predictions." << endl;
            warnedFallback = true;
        }

        for (unsigned int state = 0; state < enabledStates.size(); state++) {
            double ratio = currentStateValue > 0.0 ? enabledStates.at(state) / currentStateValue : 1.0;
            predIPS.at(state) = measuredIPS * ratio;
            predPower.at(state) = measuredPower > 0.0 ? measuredPower * ratio * ratio : 0.0;
            if ((int)state <= currentStateIndex) {
                predTemp.at(state) = measuredTemp > 0.0 ? measuredTemp : tempLimit();
            } else {
                predTemp.at(state) = tempLimit() + 10.0 * ((int)state - currentStateIndex);
            }
        }
        return;
    }

    ProfileMap::const_iterator benchmark = profile.find(nearestBenchmark);
    StateProfile::const_iterator currentEntry = benchmark->second.find(stateKey(currentStateValue));
    double baseIPS = currentEntry == benchmark->second.end() ? 0.0 : currentEntry->second.ips;
    double basePower = currentEntry == benchmark->second.end() ? 0.0 : currentEntry->second.power;
    double baseTemp = currentEntry == benchmark->second.end() ? 0.0 : currentEntry->second.temp;
    for (unsigned int state = 0; state < enabledStates.size(); state++) {
        StateProfile::const_iterator entry = benchmark->second.find(stateKey(enabledStates.at(state)));
        if (entry == benchmark->second.end()) {
            predIPS.at(state) = measuredIPS;
            predPower.at(state) = measuredPower;
            predTemp.at(state) = measuredTemp > 0.0 ? measuredTemp : tempLimit();
        } else {
            predIPS.at(state) = (measuredIPS > 0.0 && baseIPS > 0.0)
                ? measuredIPS * entry->second.ips / baseIPS
                : entry->second.ips;
            predPower.at(state) = (measuredPower > 0.0 && basePower > 0.0)
                ? measuredPower * entry->second.power / basePower
                : entry->second.power;
            predTemp.at(state) = (measuredTemp > 0.0 && baseTemp > 0.0)
                ? max(0.0, measuredTemp + entry->second.temp - baseTemp)
                : entry->second.temp;
        }
    }
}

int DVFSH1Thermal::selectState(const vector<double> &predIPS,
                               const vector<double> &predPower,
                               const vector<double> &predTemp) const
{
    int bestFeasible = -1;
    int safest = -1;
    const double eps = 1e-9;

    for (unsigned int state = 0; state < predIPS.size(); state++) {
        if (predTemp.at(state) <= tempLimit()) {
            if (bestFeasible == -1
                || predIPS.at(state) > predIPS.at(bestFeasible) + eps
                || (fabs(predIPS.at(state) - predIPS.at(bestFeasible)) <= eps && predPower.at(state) < predPower.at(bestFeasible) - eps)
                || (fabs(predIPS.at(state) - predIPS.at(bestFeasible)) <= eps
                    && fabs(predPower.at(state) - predPower.at(bestFeasible)) <= eps
                    && predTemp.at(state) < predTemp.at(bestFeasible) - eps)) {
                bestFeasible = state;
            }
        }

        if (safest == -1
            || predTemp.at(state) < predTemp.at(safest) - eps
            || (fabs(predTemp.at(state) - predTemp.at(safest)) <= eps && predPower.at(state) < predPower.at(safest) - eps)
            || (fabs(predTemp.at(state) - predTemp.at(safest)) <= eps
                && fabs(predPower.at(state) - predPower.at(safest)) <= eps
                && predIPS.at(state) > predIPS.at(safest) + eps)) {
            safest = state;
        }
    }

    return bestFeasible != -1 ? bestFeasible : safest;
}

DVFSH1Thermal::Candidate DVFSH1Thermal::evaluatePowerBudgetCandidate(const vector<int> &chosenStates,
                                                                     const vector<int> &currentStates,
                                                                     const vector<vector<double> > &predIPS,
                                                                     const vector<vector<double> > &predPower,
                                                                     const vector<vector<double> > &predTemp) const
{
    Candidate candidate;
    candidate.states = chosenStates;
    candidate.valid = chosenStates.size() == currentStates.size();
    candidate.feasible = true;
    candidate.maxPredTemp = -numeric_limits<double>::max();

    if (!candidate.valid) {
        candidate.feasible = false;
        return candidate;
    }

    for (unsigned int item = 0; item < chosenStates.size(); item++) {
        int state = chosenStates.at(item);
        if (state < 0 || state >= (int)enabledStates.size()) {
            candidate.valid = false;
            candidate.feasible = false;
            return candidate;
        }

        double itemPower = predPower.at(item).at(state);
        candidate.totalIPS += predIPS.at(item).at(state);
        candidate.totalPower += itemPower;
        candidate.maxPredTemp = max(candidate.maxPredTemp, predTemp.at(item).at(state));
        if (perCorePowerGuard > 0.0 && itemPower > perCorePowerGuard) {
            candidate.feasible = false;
        }
        if (state != currentStates.at(item)) {
            candidate.changedCores++;
        }
    }

    if (chosenStates.size() == 0) {
        candidate.maxPredTemp = 0.0;
    }

    if (candidate.totalPower > effectivePowerBudget() + 1e-9) {
        candidate.feasible = false;
    }

    return candidate;
}

DVFSH1Thermal::Candidate DVFSH1Thermal::evaluateTargetIPSCandidate(const vector<int> &chosenStates,
                                                                   const vector<int> &currentStates,
                                                                   const vector<vector<double> > &predIPS,
                                                                   const vector<vector<double> > &predPower,
                                                                   const vector<vector<double> > &predTemp) const
{
    Candidate candidate;
    candidate.states = chosenStates;
    candidate.valid = chosenStates.size() == currentStates.size();
    candidate.feasible = false;
    candidate.maxPredTemp = -numeric_limits<double>::max();

    if (!candidate.valid) {
        return candidate;
    }

    for (unsigned int item = 0; item < chosenStates.size(); item++) {
        int state = chosenStates.at(item);
        if (state < 0 || state >= (int)enabledStates.size()) {
            candidate.valid = false;
            return candidate;
        }

        candidate.totalIPS += predIPS.at(item).at(state);
        candidate.totalPower += predPower.at(item).at(state);
        candidate.maxPredTemp = max(candidate.maxPredTemp, predTemp.at(item).at(state));
        if (state != currentStates.at(item)) {
            candidate.changedCores++;
        }
    }

    if (chosenStates.size() == 0) {
        candidate.maxPredTemp = 0.0;
    }

    candidate.feasible = candidate.totalIPS + 1e-9 >= targetIPS;
    return candidate;
}

bool DVFSH1Thermal::isBetterPowerBudgetCandidate(const Candidate &candidate,
                                                 const Candidate &best,
                                                 bool requireFeasible) const
{
    const double eps = 1e-9;
    if (!candidate.valid) {
        return false;
    }
    if (requireFeasible && !candidate.feasible) {
        return false;
    }
    if (!best.valid) {
        return true;
    }

    if (requireFeasible) {
        if (candidate.totalIPS > best.totalIPS + eps) {
            return true;
        }
        if (candidate.totalIPS < best.totalIPS - eps) {
            return false;
        }
        if (candidate.totalPower < best.totalPower - eps) {
            return true;
        }
        if (candidate.totalPower > best.totalPower + eps) {
            return false;
        }
        if (candidate.maxPredTemp < best.maxPredTemp - eps) {
            return true;
        }
        if (candidate.maxPredTemp > best.maxPredTemp + eps) {
            return false;
        }
        return candidate.changedCores < best.changedCores;
    }

    if (candidate.totalPower < best.totalPower - eps) {
        return true;
    }
    if (candidate.totalPower > best.totalPower + eps) {
        return false;
    }
    if (candidate.maxPredTemp < best.maxPredTemp - eps) {
        return true;
    }
    if (candidate.maxPredTemp > best.maxPredTemp + eps) {
        return false;
    }
    if (candidate.totalIPS > best.totalIPS + eps) {
        return true;
    }
    if (candidate.totalIPS < best.totalIPS - eps) {
        return false;
    }
    return candidate.changedCores < best.changedCores;
}

bool DVFSH1Thermal::isBetterTargetIPSCandidate(const Candidate &candidate,
                                               const Candidate &best,
                                               bool requireFeasible) const
{
    const double eps = 1e-9;
    if (!candidate.valid) {
        return false;
    }
    if (requireFeasible && !candidate.feasible) {
        return false;
    }
    if (!best.valid) {
        return true;
    }

    if (requireFeasible) {
        if (candidate.totalPower < best.totalPower - eps) {
            return true;
        }
        if (candidate.totalPower > best.totalPower + eps) {
            return false;
        }
        if (candidate.totalIPS > best.totalIPS + eps) {
            return true;
        }
        if (candidate.totalIPS < best.totalIPS - eps) {
            return false;
        }
        if (candidate.maxPredTemp < best.maxPredTemp - eps) {
            return true;
        }
        if (candidate.maxPredTemp > best.maxPredTemp + eps) {
            return false;
        }
        return candidate.changedCores < best.changedCores;
    }

    if (candidate.totalIPS > best.totalIPS + eps) {
        return true;
    }
    if (candidate.totalIPS < best.totalIPS - eps) {
        return false;
    }
    if (candidate.totalPower < best.totalPower - eps) {
        return true;
    }
    if (candidate.totalPower > best.totalPower + eps) {
        return false;
    }
    if (candidate.maxPredTemp < best.maxPredTemp - eps) {
        return true;
    }
    if (candidate.maxPredTemp > best.maxPredTemp + eps) {
        return false;
    }
    return candidate.changedCores < best.changedCores;
}

vector<int> DVFSH1Thermal::selectPowerBudgetStates(const vector<int> &currentStates,
                                                   const vector<vector<double> > &predIPS,
                                                   const vector<vector<double> > &predPower,
                                                   const vector<vector<double> > &predTemp,
                                                   Candidate &selected) const
{
    Candidate bestFeasible;
    Candidate safest;
    vector<int> chosen(currentStates.size(), 0);

    function<void(unsigned int)> enumerate = [&](unsigned int item) {
        if (item == currentStates.size()) {
            Candidate candidate = evaluatePowerBudgetCandidate(chosen, currentStates, predIPS, predPower, predTemp);
            if (isBetterPowerBudgetCandidate(candidate, bestFeasible, true)) {
                bestFeasible = candidate;
            }
            if (isBetterPowerBudgetCandidate(candidate, safest, false)) {
                safest = candidate;
            }
            return;
        }

        for (unsigned int state = 0; state < enabledStates.size(); state++) {
            chosen.at(item) = state;
            enumerate(item + 1);
        }
    };

    enumerate(0);

    selected = bestFeasible.valid ? bestFeasible : safest;
    return selected.states;
}

vector<int> DVFSH1Thermal::selectTargetIPSMinPowerStates(const vector<int> &currentStates,
                                                         const vector<vector<double> > &predIPS,
                                                         const vector<vector<double> > &predPower,
                                                         const vector<vector<double> > &predTemp,
                                                         Candidate &selected) const
{
    Candidate bestFeasible;
    Candidate bestFallback;
    vector<int> chosen(currentStates.size(), 0);

    function<void(unsigned int)> enumerate = [&](unsigned int item) {
        if (item == currentStates.size()) {
            Candidate candidate = evaluateTargetIPSCandidate(chosen, currentStates, predIPS, predPower, predTemp);
            if (isBetterTargetIPSCandidate(candidate, bestFeasible, true)) {
                bestFeasible = candidate;
            }
            if (isBetterTargetIPSCandidate(candidate, bestFallback, false)) {
                bestFallback = candidate;
            }
            return;
        }

        for (unsigned int state = 0; state < enabledStates.size(); state++) {
            chosen.at(item) = state;
            enumerate(item + 1);
        }
    };

    enumerate(0);

    selected = bestFeasible.valid ? bestFeasible : bestFallback;
    return selected.states;
}

void DVFSH1Thermal::logPrediction(unsigned int coreId,
                                  double measuredIPS,
                                  double currentStateValue,
                                  const string &benchmarkName,
                                  const vector<double> &ips,
                                  const vector<double> &power,
                                  const vector<double> &temp) const
{
    cout << "[DVFSH1Thermal][debug] core=" << coreId
         << " measuredIPS=" << fixed << setprecision(4) << measuredIPS
         << " currentState=" << currentStateValue
         << " nearestBenchmark=" << (benchmarkName == "" ? "fallback" : benchmarkName)
         << " predIPS=[";
    for (unsigned int state = 0; state < ips.size(); state++) {
        if (state > 0) {
            cout << ",";
        }
        cout << fixed << setprecision(4) << ips.at(state);
    }
    cout << "] predPower=[";
    for (unsigned int state = 0; state < power.size(); state++) {
        if (state > 0) {
            cout << ",";
        }
        cout << fixed << setprecision(4) << power.at(state);
    }
    cout << "] predTemp=[";
    for (unsigned int state = 0; state < temp.size(); state++) {
        if (state > 0) {
            cout << ",";
        }
        cout << fixed << setprecision(4) << temp.at(state);
    }
    cout << "]" << endl;
}

vector<int> DVFSH1Thermal::getFrequencies(const vector<int> &oldFrequencies, const vector<bool> &activeCores)
{
    vector<int> taskIds;
    vector<int> threadIds;
    return getFrequencies(oldFrequencies, taskIds, threadIds, activeCores);
}

vector<int> DVFSH1Thermal::getFrequencies(const vector<int> &oldFrequencies, const vector<int> &taskIds, const vector<int> &threadIds, const vector<bool> &activeCores)
{
    vector<int> result(numberOfCores);
    int lowFrequency = frequencies.size() > 0 ? frequencies.at(0) : 0;
    UInt64 time = Sim()->getClockSkewMinimizationServer()->getGlobalTime().getNS();

    for (int core = 0; core < numberOfCores; core++) {
        int oldFrequency = core < (int)oldFrequencies.size() ? oldFrequencies.at(core) : lowFrequency;
        if (isMasterCore(taskIds, threadIds, core)) {
            result.at(core) = oldFrequency;
            cout << "[DVFSH1Thermal] freeze master thread task=" << taskIds.at(core)
                 << " thread=" << threadIds.at(core)
                 << " core=" << core
                 << " frequency=" << oldFrequency << endl;
            continue;
        }

        bool isActive = core < (int)activeCores.size() && activeCores.at(core);
        if (!isActive) {
            result.at(core) = lowFrequency;
        } else {
            result.at(core) = oldFrequency;
        }
    }

    if (isPowerBudgetObjective() || isTargetIPSMinPowerObjective()) {
        vector<unsigned int> activeCoreIds;
        vector<int> currentStates;
        vector<vector<double> > predIPS;
        vector<vector<double> > predPower;
        vector<vector<double> > predTemp;

        for (int core = 0; core < numberOfCores; core++) {
            bool isActive = core < (int)activeCores.size() && activeCores.at(core);
            if (!isActive) {
                continue;
            }
            if (isMasterCore(taskIds, threadIds, core)) {
                continue;
            }

            int currentFrequency = result.at(core);
            int currentStateIndex = getStateForFrequency(currentFrequency);
            if (currentStateIndex < 0 || currentStateIndex >= (int)enabledStates.size()) {
                continue;
            }

            double currentStateValue = enabledStates.at(currentStateIndex);
            double measuredIPS = getMeasuredIPSBillions(core, currentStateValue);
            string nearestBenchmark;
            vector<double> itemPredIPS;
            vector<double> itemPredPower;
            vector<double> itemPredTemp;
            buildPrediction(core, currentStateIndex, measuredIPS, nearestBenchmark, itemPredIPS, itemPredPower, itemPredTemp);

            activeCoreIds.push_back(core);
            currentStates.push_back(currentStateIndex);
            predIPS.push_back(itemPredIPS);
            predPower.push_back(itemPredPower);
            predTemp.push_back(itemPredTemp);

            if (debug) {
                logPrediction(core, measuredIPS, currentStateValue, nearestBenchmark, itemPredIPS, itemPredPower, itemPredTemp);
            }
        }

        if (activeCoreIds.size() == 0) {
            return result;
        }

        if (isTargetIPSMinPowerObjective()) {
            Candidate current = evaluateTargetIPSCandidate(currentStates, currentStates, predIPS, predPower, predTemp);
            Candidate selected;
            vector<int> selectedStates = selectTargetIPSMinPowerStates(currentStates, predIPS, predPower, predTemp, selected);
            for (unsigned int item = 0; item < activeCoreIds.size() && item < selectedStates.size(); item++) {
                int selectedState = selectedStates.at(item);
                if (selectedState >= 0 && selectedState < (int)frequencies.size()) {
                    result.at(activeCoreIds.at(item)) = frequencies.at(selectedState);
                }
            }

            cout << "[DVFSH1Thermal] objective=target_ips_min_power"
                 << " time=" << time
                 << " active=" << activeCoreIds.size()
                 << " target_ips=" << fixed << setprecision(4) << targetIPS
                 << " currentTotalPredIPS=" << fixed << setprecision(4) << current.totalIPS
                 << " selectedTotalPredIPS=" << fixed << setprecision(4) << selected.totalIPS
                 << " currentTotalPredPower=" << fixed << setprecision(4) << current.totalPower
                 << " selectedTotalPredPower=" << fixed << setprecision(4) << selected.totalPower
                 << " selectedMaxProfileTemp=" << fixed << setprecision(4) << selected.maxPredTemp
                 << " feasible=" << (selected.feasible ? "true" : "false")
                 << " dvfs_changes=" << selected.changedCores
                 << (selected.feasible ? "" : " reason=no_feasible_target_ips_frequency_combination")
                 << " selectedStates=[";
            for (unsigned int item = 0; item < selectedStates.size(); item++) {
                if (item > 0) {
                    cout << ",";
                }
                cout << fixed << setprecision(4) << enabledStates.at(selectedStates.at(item));
            }
            cout << "] selectedFrequencies=[";
            for (unsigned int item = 0; item < selectedStates.size(); item++) {
                if (item > 0) {
                    cout << ",";
                }
                int selectedState = selectedStates.at(item);
                cout << (selectedState >= 0 && selectedState < (int)frequencies.size() ? frequencies.at(selectedState) : 0);
            }
            cout << "]" << endl;

            return result;
        }

        Candidate current = evaluatePowerBudgetCandidate(currentStates, currentStates, predIPS, predPower, predTemp);
        if (powerBudget <= 0.0) {
            if (!warnedInvalidPowerBudget) {
                cout << "[DVFSH1Thermal][Warning]: objective=power_budget_max_ips but power_budget="
                     << fixed << setprecision(4) << powerBudget
                     << "; H1 per-core DVFS optimization is disabled for this epoch." << endl;
                warnedInvalidPowerBudget = true;
            }

            cout << "[DVFSH1Thermal] objective=power_budget_max_ips"
                 << " time=" << time
                 << " active=" << activeCoreIds.size()
                 << " power_budget=" << fixed << setprecision(4) << powerBudget
                 << " power_budget_margin=" << fixed << setprecision(4) << powerBudgetMargin
                 << " effective_power_budget=" << fixed << setprecision(4) << effectivePowerBudget()
                 << " per_core_power_guard=" << fixed << setprecision(4) << perCorePowerGuard
                 << " max_temp=" << fixed << setprecision(4) << maxTemp
                 << " currentTotalPredIPS=" << fixed << setprecision(4) << current.totalIPS
                 << " selectedTotalPredIPS=" << fixed << setprecision(4) << current.totalIPS
                 << " currentTotalPredPower=" << fixed << setprecision(4) << current.totalPower
                 << " selectedTotalPredPower=" << fixed << setprecision(4) << current.totalPower
                 << " selectedMaxProfileTemp=" << fixed << setprecision(4) << current.maxPredTemp
                 << " feasible=false"
                 << " dvfs_changes=0"
                 << " reason=invalid_power_budget"
                 << endl;

            return result;
        }

        Candidate selected;
        vector<int> selectedStates = selectPowerBudgetStates(currentStates, predIPS, predPower, predTemp, selected);
        for (unsigned int item = 0; item < activeCoreIds.size() && item < selectedStates.size(); item++) {
            int selectedState = selectedStates.at(item);
            if (selectedState >= 0 && selectedState < (int)frequencies.size()) {
                result.at(activeCoreIds.at(item)) = frequencies.at(selectedState);
            }
        }

        cout << "[DVFSH1Thermal] objective=power_budget_max_ips"
             << " time=" << time
             << " active=" << activeCoreIds.size()
             << " power_budget=" << fixed << setprecision(4) << powerBudget
             << " power_budget_margin=" << fixed << setprecision(4) << powerBudgetMargin
             << " effective_power_budget=" << fixed << setprecision(4) << effectivePowerBudget()
             << " per_core_power_guard=" << fixed << setprecision(4) << perCorePowerGuard
             << " max_temp=" << fixed << setprecision(4) << maxTemp
             << " currentTotalPredIPS=" << fixed << setprecision(4) << current.totalIPS
             << " selectedTotalPredIPS=" << fixed << setprecision(4) << selected.totalIPS
             << " currentTotalPredPower=" << fixed << setprecision(4) << current.totalPower
             << " selectedTotalPredPower=" << fixed << setprecision(4) << selected.totalPower
             << " selectedMaxProfileTemp=" << fixed << setprecision(4) << selected.maxPredTemp
             << " feasible=" << (selected.feasible ? "true" : "false")
             << " dvfs_changes=" << selected.changedCores
             << (selected.feasible ? "" : " reason=no_feasible_power_budget_frequency_combination")
             << " selectedStates=[";
        for (unsigned int item = 0; item < selectedStates.size(); item++) {
            if (item > 0) {
                cout << ",";
            }
            cout << fixed << setprecision(4) << enabledStates.at(selectedStates.at(item));
        }
        cout << "] selectedFrequencies=[";
        for (unsigned int item = 0; item < selectedStates.size(); item++) {
            if (item > 0) {
                cout << ",";
            }
            int selectedState = selectedStates.at(item);
            cout << (selectedState >= 0 && selectedState < (int)frequencies.size() ? frequencies.at(selectedState) : 0);
        }
        cout << "]" << endl;

        return result;
    }

    for (int core = 0; core < numberOfCores; core++) {
        bool isActive = core < (int)activeCores.size() && activeCores.at(core);
        if (!isActive) {
            continue;
        }
        if (isMasterCore(taskIds, threadIds, core)) {
            continue;
        }

        int currentFrequency = result.at(core);
        int currentStateIndex = getStateForFrequency(currentFrequency);
        if (currentStateIndex < 0 || currentStateIndex >= (int)enabledStates.size()) {
            continue;
        }

        double currentStateValue = enabledStates.at(currentStateIndex);
        double measuredIPS = getMeasuredIPSBillions(core, currentStateValue);
        string nearestBenchmark;
        vector<double> predIPS;
        vector<double> predPower;
        vector<double> predTemp;
        buildPrediction(core, currentStateIndex, measuredIPS, nearestBenchmark, predIPS, predPower, predTemp);

        int selectedState = selectState(predIPS, predPower, predTemp);
        if (selectedState < 0 || selectedState >= (int)frequencies.size()) {
            result.at(core) = currentFrequency;
            continue;
        }

        result.at(core) = frequencies.at(selectedState);

        cout << "[DVFSH1Thermal] objective=thermal_max_ips"
             << " time=" << time
             << " core=" << core
             << " measuredIPS=" << fixed << setprecision(4) << measuredIPS
             << " currentFrequency=" << currentFrequency
             << " currentState=" << fixed << setprecision(4) << currentStateValue
             << " nearestBenchmark=" << (nearestBenchmark == "" ? "fallback" : nearestBenchmark)
             << " selectedFrequency=" << result.at(core)
             << " selectedState=" << fixed << setprecision(4) << enabledStates.at(selectedState)
             << " selectedPredIPS=" << fixed << setprecision(4) << predIPS.at(selectedState)
             << " selectedPredTemp=" << fixed << setprecision(4) << predTemp.at(selectedState)
             << " selectedPredPower=" << fixed << setprecision(4) << predPower.at(selectedState)
             << " max_temp=" << fixed << setprecision(4) << maxTemp
             << " thermal_margin=" << fixed << setprecision(4) << thermalMargin
             << endl;

        if (debug) {
            logPrediction(core, measuredIPS, currentStateValue, nearestBenchmark, predIPS, predPower, predTemp);
        }
    }

    return result;
}
