#include "migration_h1.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>

using namespace std;

MigrationH1::MigrationH1(const PerformanceCounters *performanceCounters,
                         int numberOfCores,
                         const vector<int> &coreToState,
                         const vector<double> &enabledStates,
                         double targetIPS,
                         const string &objective,
                         double maxTemp,
                         double thermalMargin,
                         double powerBudget,
                         double powerBudgetMargin,
                         double perCorePowerGuard,
                         const string &profileFile,
                         bool freezeMaster,
                         bool debug)
    : performanceCounters(performanceCounters)
    , numberOfCores(numberOfCores)
    , coreToState(coreToState)
    , enabledStates(enabledStates)
    , targetIPS(targetIPS)
    , objective(objective)
    , maxTemp(maxTemp)
    , thermalMargin(thermalMargin)
    , powerBudget(powerBudget)
    , powerBudgetMargin(powerBudgetMargin)
    , perCorePowerGuard(perCorePowerGuard)
    , profileFile(profileFile)
    , freezeMaster(freezeMaster)
    , debug(debug)
    , warnedFallback(false)
    , warnedInvalidPowerBudget(false)
{
    if ((int)this->coreToState.size() < numberOfCores) {
        cout << "[MigrationH1][Warning]: core_state has fewer entries than cores; missing cores will be ignored." << endl;
    }

    for (int core = 0; core < numberOfCores && core < (int)this->coreToState.size(); core++) {
        int state = this->coreToState.at(core);
        if (state < 0 || state >= (int)this->enabledStates.size()) {
            cout << "[MigrationH1][Warning]: core " << core << " has invalid state index " << state << endl;
        }
    }

    for (unsigned int state = 1; state < this->enabledStates.size(); state++) {
        if (this->enabledStates.at(state) < this->enabledStates.at(state - 1)) {
            cout << "[MigrationH1][Warning]: enabled states are not sorted by frequency; H1 will use frequency order internally." << endl;
            break;
        }
    }

    loadProfile();
}

MigrationH1::ThermalCandidate::ThermalCandidate()
    : totalIPS(0.0)
    , totalPower(0.0)
    , maxPredTemp(0.0)
    , movedItems(0)
    , feasible(false)
    , valid(false)
{
}

int MigrationH1::stateKey(double state) const
{
    return (int)floor(state * 1000.0 + 0.5);
}

void MigrationH1::loadProfile()
{
    if (profileFile == "") {
        cout << "[MigrationH1][Warning]: no profile file configured; using measured fallback predictions." << endl;
        return;
    }

    ifstream file(profileFile.c_str());
    if (!file.good()) {
        cout << "[MigrationH1][Warning]: failed to open profile file " << profileFile << "; using measured fallback predictions." << endl;
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

    cout << "[MigrationH1] loaded " << rows << " profile rows from " << profileFile
         << " (" << validBenchmarks << " benchmarks cover all enabled states)" << endl;
}

bool MigrationH1::hasAllEnabledStates(const string &benchmarkName) const
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

string MigrationH1::findNearestBenchmark(double currentStateValue, double measuredIPS) const
{
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

        double distance = fabs(currentState->second.ips - measuredIPS);
        if (distance < nearestDistance) {
            nearestDistance = distance;
            nearestBenchmark = benchmark->first;
        }
    }

    return nearestBenchmark;
}

bool MigrationH1::isMasterCore(const vector<int> &taskIds, const vector<int> &threadIds, unsigned int coreId) const
{
    if (!freezeMaster) {
        return false;
    }
    if (coreId >= taskIds.size() || coreId >= threadIds.size()) {
        return false;
    }

    return taskIds.at(coreId) != -1 && threadIds.at(coreId) == 0;
}

vector<unsigned int> MigrationH1::getActiveCoreIds(const vector<int> &taskIds, const vector<int> &threadIds, const vector<bool> &activeCores) const
{
    vector<unsigned int> activeCoreIds;
    int coreCount = min(numberOfCores, (int)activeCores.size());

    for (int core = 0; core < coreCount; core++) {
        if (!activeCores.at(core)) {
            continue;
        }
        if (core >= (int)taskIds.size() || taskIds.at(core) == -1) {
            if (debug) {
                cout << "[MigrationH1][debug]: skipping active core " << core << " with no assigned task" << endl;
            }
            continue;
        }
        if (isMasterCore(taskIds, threadIds, core)) {
            cout << "[MigrationH1] skip master thread task=" << taskIds.at(core)
                 << " thread=" << threadIds.at(core)
                 << " core=" << core << endl;
            continue;
        }
        if (getStateForCore(core) < 0) {
            continue;
        }
        activeCoreIds.push_back(core);
    }

    if ((int)activeCoreIds.size() > numberOfCores) {
        cout << "[MigrationH1][Warning]: more active items than cores; truncating to one item per core." << endl;
        activeCoreIds.resize(numberOfCores);
    }

    return activeCoreIds;
}

int MigrationH1::getStateForCore(unsigned int coreId) const
{
    if (coreId >= (unsigned int)numberOfCores || coreId >= coreToState.size()) {
        return -1;
    }

    int state = coreToState.at(coreId);
    if (state < 0 || state >= (int)enabledStates.size()) {
        return -1;
    }

    return state;
}

double MigrationH1::getMeasuredIPSBillions(unsigned int coreId, double currentStateValue) const
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

double MigrationH1::getMeasuredPower(unsigned int coreId) const
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

double MigrationH1::getMeasuredTemperature(unsigned int coreId) const
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

bool MigrationH1::isThermalObjective() const
{
    return objective == "thermal_max_ips";
}

bool MigrationH1::isPowerBudgetObjective() const
{
    return objective == "power_budget_max_ips";
}

bool MigrationH1::isTargetIPSMinPowerObjective() const
{
    return objective == "target_ips_min_power";
}

bool MigrationH1::isTargetIPSObjective() const
{
    return objective == "target_ips";
}

double MigrationH1::tempLimit() const
{
    return maxTemp - thermalMargin;
}

double MigrationH1::effectivePowerBudget() const
{
    return powerBudget * powerBudgetMargin;
}

void MigrationH1::buildPredictions(const vector<unsigned int> &activeCoreIds,
                                   vector<vector<double> > &predIPS,
                                   vector<vector<double> > &predPower,
                                   vector<vector<double> > &predTemp)
{
    predIPS.assign(activeCoreIds.size(), vector<double>(enabledStates.size(), 0.0));
    predPower.assign(activeCoreIds.size(), vector<double>(enabledStates.size(), 0.0));
    predTemp.assign(activeCoreIds.size(), vector<double>(enabledStates.size(), 0.0));

    for (unsigned int item = 0; item < activeCoreIds.size(); item++) {
        unsigned int coreId = activeCoreIds.at(item);
        int currentStateIndex = getStateForCore(coreId);
        double currentStateValue = enabledStates.at(currentStateIndex);
        double measuredIPS = getMeasuredIPSBillions(coreId, currentStateValue);
        double measuredPower = getMeasuredPower(coreId);
        double measuredTemp = getMeasuredTemperature(coreId);

        string nearestBenchmark = findNearestBenchmark(currentStateValue, measuredIPS);
        if (nearestBenchmark == "") {
            if (!warnedFallback) {
                cout << "[MigrationH1][Warning]: no valid nearest benchmark with all enabled states found; using conservative measured fallback predictions." << endl;
                warnedFallback = true;
            }

            for (unsigned int state = 0; state < enabledStates.size(); state++) {
                double ratio = currentStateValue > 0.0 ? enabledStates.at(state) / currentStateValue : 1.0;
                predIPS.at(item).at(state) = measuredIPS * ratio;
                predPower.at(item).at(state) = measuredPower > 0.0 ? measuredPower * ratio * ratio : 0.0;
                if ((int)state <= currentStateIndex) {
                    predTemp.at(item).at(state) = measuredTemp > 0.0 ? measuredTemp : tempLimit();
                } else {
                    predTemp.at(item).at(state) = tempLimit() + 10.0 * ((int)state - currentStateIndex);
                }
            }
        } else {
            ProfileMap::const_iterator benchmark = profile.find(nearestBenchmark);
            for (unsigned int state = 0; state < enabledStates.size(); state++) {
                StateProfile::const_iterator entry = benchmark->second.find(stateKey(enabledStates.at(state)));
                if (entry == benchmark->second.end()) {
                    predIPS.at(item).at(state) = measuredIPS;
                    predPower.at(item).at(state) = measuredPower;
                    predTemp.at(item).at(state) = measuredTemp > 0.0 ? measuredTemp : tempLimit();
                } else {
                    predIPS.at(item).at(state) = entry->second.ips;
                    predPower.at(item).at(state) = entry->second.power;
                    predTemp.at(item).at(state) = entry->second.temp;
                }
            }
        }

        if (debug) {
            logPrediction(coreId, measuredIPS, currentStateValue, nearestBenchmark, predIPS.at(item), predPower.at(item), predTemp.at(item));
        }
    }
}

vector<int> MigrationH1::runH1(const vector<unsigned int> &activeCoreIds,
                               const vector<int> &taskIds,
                               const vector<int> &threadIds,
                               const vector<vector<double> > &predIPS,
                               const vector<vector<double> > &predPower)
{
    vector<int> desiredState(activeCoreIds.size(), -1);
    if (enabledStates.size() == 0 || activeCoreIds.size() == 0) {
        return desiredState;
    }

    vector<int> capacity(enabledStates.size(), 0);
    for (int core = 0; core < numberOfCores; core++) {
        if (isMasterCore(taskIds, threadIds, core)) {
            continue;
        }
        int state = getStateForCore(core);
        if (state >= 0) {
            capacity.at(state)++;
        }
    }

    int totalCapacity = accumulate(capacity.begin(), capacity.end(), 0);
    if ((int)activeCoreIds.size() > totalCapacity) {
        cout << "[MigrationH1][Warning]: active item count exceeds configured state capacity; some items will keep their current state." << endl;
    }

    vector<int> stateOrder(enabledStates.size());
    for (unsigned int state = 0; state < enabledStates.size(); state++) {
        stateOrder.at(state) = state;
    }
    sort(stateOrder.begin(), stateOrder.end(), [this](int a, int b) {
        return enabledStates.at(a) < enabledStates.at(b);
    });

    vector<bool> assigned(activeCoreIds.size(), false);
    for (vector<int>::reverse_iterator stateIt = stateOrder.rbegin(); stateIt != stateOrder.rend(); ++stateIt) {
        int state = *stateIt;
        vector<int> candidates;
        for (unsigned int item = 0; item < activeCoreIds.size(); item++) {
            if (!assigned.at(item)) {
                candidates.push_back(item);
            }
        }

        sort(candidates.begin(), candidates.end(), [&predPower, state](int a, int b) {
            return predPower.at(a).at(state) < predPower.at(b).at(state);
        });

        int assignedToState = 0;
        for (unsigned int candidate = 0; candidate < candidates.size() && assignedToState < capacity.at(state); candidate++) {
            int item = candidates.at(candidate);
            desiredState.at(item) = state;
            assigned.at(item) = true;
            assignedToState++;
        }
    }

    for (unsigned int item = 0; item < activeCoreIds.size(); item++) {
        if (desiredState.at(item) == -1) {
            int currentState = getStateForCore(activeCoreIds.at(item));
            desiredState.at(item) = currentState >= 0 ? currentState : stateOrder.front();
        }
    }

    double totalIPS = predictedTotalIPS(predIPS, desiredState);

    while (totalIPS < targetIPS) {
        bool foundMove = false;
        int bestLowItem = -1;
        int bestHighItem = -1;
        int bestLowState = -1;
        int bestHighState = -1;
        double bestPriority = -numeric_limits<double>::max();
        double bestDeltaIPS = 0.0;

        for (unsigned int orderIndex = 0; orderIndex + 1 < stateOrder.size(); orderIndex++) {
            int lowState = stateOrder.at(orderIndex);
            int highState = stateOrder.at(orderIndex + 1);

            for (unsigned int lowItem = 0; lowItem < desiredState.size(); lowItem++) {
                if (desiredState.at(lowItem) != lowState) {
                    continue;
                }

                for (unsigned int highItem = 0; highItem < desiredState.size(); highItem++) {
                    if (desiredState.at(highItem) != highState) {
                        continue;
                    }

                    double oldIPS = predIPS.at(lowItem).at(lowState) + predIPS.at(highItem).at(highState);
                    double newIPS = predIPS.at(lowItem).at(highState) + predIPS.at(highItem).at(lowState);
                    double deltaIPS = newIPS - oldIPS;
                    if (deltaIPS <= 0.0) {
                        continue;
                    }

                    double oldPower = predPower.at(lowItem).at(lowState) + predPower.at(highItem).at(highState);
                    double newPower = predPower.at(lowItem).at(highState) + predPower.at(highItem).at(lowState);
                    double deltaPower = newPower - oldPower;
                    double priority = deltaPower <= 0.0 ? 1e100 : deltaIPS / deltaPower;

                    if (!foundMove || priority > bestPriority || (priority == bestPriority && deltaIPS > bestDeltaIPS)) {
                        foundMove = true;
                        bestLowItem = lowItem;
                        bestHighItem = highItem;
                        bestLowState = lowState;
                        bestHighState = highState;
                        bestPriority = priority;
                        bestDeltaIPS = deltaIPS;
                    }
                }
            }
        }

        if (!foundMove) {
            break;
        }

        if (debug) {
            cout << "[MigrationH1][debug] exchange core " << activeCoreIds.at(bestLowItem)
                 << " state " << enabledStates.at(bestLowState)
                 << " <-> core " << activeCoreIds.at(bestHighItem)
                 << " state " << enabledStates.at(bestHighState) << endl;
        }

        swap(desiredState.at(bestLowItem), desiredState.at(bestHighItem));
        totalIPS += bestDeltaIPS;
    }

    return desiredState;
}

vector<int> MigrationH1::getAvailableFixedCores(const vector<unsigned int> &activeCoreIds,
                                                const vector<int> &taskIds,
                                                const vector<int> &threadIds) const
{
    vector<int> availableCores;
    for (int core = 0; core < numberOfCores; core++) {
        if (getStateForCore(core) < 0) {
            continue;
        }
        if (isMasterCore(taskIds, threadIds, core)) {
            continue;
        }

        bool isActiveItemCore = find(activeCoreIds.begin(), activeCoreIds.end(), (unsigned int)core) != activeCoreIds.end();
        bool isUnassigned = core >= (int)taskIds.size() || taskIds.at(core) == -1;
        if (isActiveItemCore || isUnassigned) {
            availableCores.push_back(core);
        }
    }

    return availableCores;
}

MigrationH1::ThermalCandidate MigrationH1::evaluateThermalCandidate(const vector<unsigned int> &activeCoreIds,
                                                                    const vector<int> &desiredCore,
                                                                    const vector<vector<double> > &predIPS,
                                                                    const vector<vector<double> > &predPower,
                                                                    const vector<vector<double> > &predTemp) const
{
    ThermalCandidate candidate;
    candidate.desiredCore = desiredCore;
    candidate.valid = desiredCore.size() == activeCoreIds.size();
    candidate.feasible = true;
    candidate.maxPredTemp = -numeric_limits<double>::max();

    if (!candidate.valid) {
        candidate.feasible = false;
        return candidate;
    }

    for (unsigned int item = 0; item < desiredCore.size(); item++) {
        int state = getStateForCore(desiredCore.at(item));
        if (state < 0) {
            candidate.valid = false;
            candidate.feasible = false;
            return candidate;
        }

        double itemTemp = predTemp.at(item).at(state);
        candidate.totalIPS += predIPS.at(item).at(state);
        candidate.totalPower += predPower.at(item).at(state);
        candidate.maxPredTemp = max(candidate.maxPredTemp, itemTemp);
        if (itemTemp > tempLimit()) {
            candidate.feasible = false;
        }
        if ((int)activeCoreIds.at(item) != desiredCore.at(item)) {
            candidate.movedItems++;
        }
    }

    if (desiredCore.size() == 0) {
        candidate.maxPredTemp = 0.0;
    }

    return candidate;
}

MigrationH1::ThermalCandidate MigrationH1::evaluatePowerBudgetCandidate(const vector<unsigned int> &activeCoreIds,
                                                                        const vector<int> &desiredCore,
                                                                        const vector<vector<double> > &predIPS,
                                                                        const vector<vector<double> > &predPower,
                                                                        const vector<vector<double> > &predTemp) const
{
    ThermalCandidate candidate;
    candidate.desiredCore = desiredCore;
    candidate.valid = desiredCore.size() == activeCoreIds.size();
    candidate.feasible = true;
    candidate.maxPredTemp = -numeric_limits<double>::max();

    if (!candidate.valid) {
        candidate.feasible = false;
        return candidate;
    }

    for (unsigned int item = 0; item < desiredCore.size(); item++) {
        int state = getStateForCore(desiredCore.at(item));
        if (state < 0) {
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
        if ((int)activeCoreIds.at(item) != desiredCore.at(item)) {
            candidate.movedItems++;
        }
    }

    if (desiredCore.size() == 0) {
        candidate.maxPredTemp = 0.0;
    }

    if (candidate.totalPower > effectivePowerBudget() + 1e-9) {
        candidate.feasible = false;
    }

    return candidate;
}

MigrationH1::ThermalCandidate MigrationH1::evaluateTargetIPSCandidate(const vector<unsigned int> &activeCoreIds,
                                                                      const vector<int> &desiredCore,
                                                                      const vector<vector<double> > &predIPS,
                                                                      const vector<vector<double> > &predPower,
                                                                      const vector<vector<double> > &predTemp) const
{
    ThermalCandidate candidate;
    candidate.desiredCore = desiredCore;
    candidate.valid = desiredCore.size() == activeCoreIds.size();
    candidate.feasible = false;
    candidate.maxPredTemp = -numeric_limits<double>::max();

    if (!candidate.valid) {
        return candidate;
    }

    for (unsigned int item = 0; item < desiredCore.size(); item++) {
        int state = getStateForCore(desiredCore.at(item));
        if (state < 0) {
            candidate.valid = false;
            return candidate;
        }

        candidate.totalIPS += predIPS.at(item).at(state);
        candidate.totalPower += predPower.at(item).at(state);
        candidate.maxPredTemp = max(candidate.maxPredTemp, predTemp.at(item).at(state));
        if ((int)activeCoreIds.at(item) != desiredCore.at(item)) {
            candidate.movedItems++;
        }
    }

    if (desiredCore.size() == 0) {
        candidate.maxPredTemp = 0.0;
    }

    candidate.feasible = candidate.totalIPS + 1e-9 >= targetIPS;
    return candidate;
}

bool MigrationH1::isBetterThermalCandidate(const ThermalCandidate &candidate,
                                           const ThermalCandidate &best,
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
        return candidate.movedItems < best.movedItems;
    }

    if (candidate.maxPredTemp < best.maxPredTemp - eps) {
        return true;
    }
    if (candidate.maxPredTemp > best.maxPredTemp + eps) {
        return false;
    }
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
    return candidate.movedItems < best.movedItems;
}

bool MigrationH1::isBetterPowerBudgetCandidate(const ThermalCandidate &candidate,
                                               const ThermalCandidate &best,
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
        return candidate.movedItems < best.movedItems;
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
    return candidate.movedItems < best.movedItems;
}

bool MigrationH1::isBetterTargetIPSCandidate(const ThermalCandidate &candidate,
                                             const ThermalCandidate &best,
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
        return candidate.movedItems < best.movedItems;
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
    return candidate.movedItems < best.movedItems;
}

vector<int> MigrationH1::runThermalMaxIPS(const vector<unsigned int> &activeCoreIds,
                                          const vector<int> &taskIds,
                                          const vector<int> &threadIds,
                                          const vector<vector<double> > &predIPS,
                                          const vector<vector<double> > &predPower,
                                          const vector<vector<double> > &predTemp,
                                          ThermalCandidate &selected) const
{
    vector<int> availableCores = getAvailableFixedCores(activeCoreIds, taskIds, threadIds);
    ThermalCandidate bestFeasible;
    ThermalCandidate safest;

    if (availableCores.size() < activeCoreIds.size()) {
        cout << "[MigrationH1][Warning]: active item count exceeds available fixed-core slots; keeping current mapping." << endl;
        vector<int> currentCore(activeCoreIds.begin(), activeCoreIds.end());
        selected = evaluateThermalCandidate(activeCoreIds, currentCore, predIPS, predPower, predTemp);
        return currentCore;
    }

    vector<int> desiredCore(activeCoreIds.size(), -1);
    vector<bool> used(availableCores.size(), false);

    function<void(unsigned int)> enumerate = [&](unsigned int item) {
        if (item == activeCoreIds.size()) {
            ThermalCandidate candidate = evaluateThermalCandidate(activeCoreIds, desiredCore, predIPS, predPower, predTemp);
            if (isBetterThermalCandidate(candidate, bestFeasible, true)) {
                bestFeasible = candidate;
            }
            if (isBetterThermalCandidate(candidate, safest, false)) {
                safest = candidate;
            }
            return;
        }

        for (unsigned int slot = 0; slot < availableCores.size(); slot++) {
            if (used.at(slot)) {
                continue;
            }
            used.at(slot) = true;
            desiredCore.at(item) = availableCores.at(slot);
            enumerate(item + 1);
            desiredCore.at(item) = -1;
            used.at(slot) = false;
        }
    };

    enumerate(0);

    if (bestFeasible.valid) {
        selected = bestFeasible;
    } else {
        selected = safest;
    }

    return selected.desiredCore;
}

vector<int> MigrationH1::runPowerBudgetMaxIPS(const vector<unsigned int> &activeCoreIds,
                                              const vector<int> &taskIds,
                                              const vector<int> &threadIds,
                                              const vector<vector<double> > &predIPS,
                                              const vector<vector<double> > &predPower,
                                              const vector<vector<double> > &predTemp,
                                              ThermalCandidate &selected) const
{
    vector<int> availableCores = getAvailableFixedCores(activeCoreIds, taskIds, threadIds);
    ThermalCandidate bestFeasible;
    ThermalCandidate safest;

    if (availableCores.size() < activeCoreIds.size()) {
        cout << "[MigrationH1][Warning]: active item count exceeds available fixed-core slots; keeping current mapping." << endl;
        vector<int> currentCore(activeCoreIds.begin(), activeCoreIds.end());
        selected = evaluatePowerBudgetCandidate(activeCoreIds, currentCore, predIPS, predPower, predTemp);
        return currentCore;
    }

    vector<int> desiredCore(activeCoreIds.size(), -1);
    vector<bool> used(availableCores.size(), false);

    function<void(unsigned int)> enumerate = [&](unsigned int item) {
        if (item == activeCoreIds.size()) {
            ThermalCandidate candidate = evaluatePowerBudgetCandidate(activeCoreIds, desiredCore, predIPS, predPower, predTemp);
            if (isBetterPowerBudgetCandidate(candidate, bestFeasible, true)) {
                bestFeasible = candidate;
            }
            if (isBetterPowerBudgetCandidate(candidate, safest, false)) {
                safest = candidate;
            }
            return;
        }

        for (unsigned int slot = 0; slot < availableCores.size(); slot++) {
            if (used.at(slot)) {
                continue;
            }
            used.at(slot) = true;
            desiredCore.at(item) = availableCores.at(slot);
            enumerate(item + 1);
            desiredCore.at(item) = -1;
            used.at(slot) = false;
        }
    };

    enumerate(0);

    if (bestFeasible.valid) {
        selected = bestFeasible;
    } else {
        selected = safest;
    }

    return selected.desiredCore;
}

vector<int> MigrationH1::runTargetIPSMinPower(const vector<unsigned int> &activeCoreIds,
                                              const vector<int> &taskIds,
                                              const vector<int> &threadIds,
                                              const vector<vector<double> > &predIPS,
                                              const vector<vector<double> > &predPower,
                                              const vector<vector<double> > &predTemp,
                                              ThermalCandidate &selected) const
{
    vector<int> availableCores = getAvailableFixedCores(activeCoreIds, taskIds, threadIds);
    ThermalCandidate bestFeasible;
    ThermalCandidate bestFallback;

    if (availableCores.size() < activeCoreIds.size()) {
        cout << "[MigrationH1][Warning]: active item count exceeds available fixed-core slots; keeping current mapping." << endl;
        vector<int> currentCore(activeCoreIds.begin(), activeCoreIds.end());
        selected = evaluateTargetIPSCandidate(activeCoreIds, currentCore, predIPS, predPower, predTemp);
        return currentCore;
    }

    vector<int> desiredCore(activeCoreIds.size(), -1);
    vector<bool> used(availableCores.size(), false);

    function<void(unsigned int)> enumerate = [&](unsigned int item) {
        if (item == activeCoreIds.size()) {
            ThermalCandidate candidate = evaluateTargetIPSCandidate(activeCoreIds, desiredCore, predIPS, predPower, predTemp);
            if (isBetterTargetIPSCandidate(candidate, bestFeasible, true)) {
                bestFeasible = candidate;
            }
            if (isBetterTargetIPSCandidate(candidate, bestFallback, false)) {
                bestFallback = candidate;
            }
            return;
        }

        for (unsigned int slot = 0; slot < availableCores.size(); slot++) {
            if (used.at(slot)) {
                continue;
            }
            used.at(slot) = true;
            desiredCore.at(item) = availableCores.at(slot);
            enumerate(item + 1);
            desiredCore.at(item) = -1;
            used.at(slot) = false;
        }
    };

    enumerate(0);

    selected = bestFeasible.valid ? bestFeasible : bestFallback;
    return selected.desiredCore;
}

vector<migration> MigrationH1::convertDesiredStatesToMigrations(const vector<unsigned int> &activeCoreIds,
                                                                 const vector<int> &desiredState,
                                                                 const vector<int> &taskIds,
                                                                 const vector<int> &threadIds) const
{
    vector<int> desiredCore(activeCoreIds.size(), -1);
    vector<bool> reserved(numberOfCores, false);

    // Keep already-correct items in place first. This preserves the old H1
    // behavior when the target state is already occupied by the same item.
    for (unsigned int item = 0; item < activeCoreIds.size(); item++) {
        int core = activeCoreIds.at(item);
        if (getStateForCore(core) == desiredState.at(item)) {
            desiredCore.at(item) = core;
            reserved.at(core) = true;
        }
    }

    for (unsigned int item = 0; item < activeCoreIds.size(); item++) {
        if (desiredCore.at(item) != -1) {
            continue;
        }

        int wantedState = desiredState.at(item);
        for (int pass = 0; pass < 3 && desiredCore.at(item) == -1; pass++) {
            for (int core = 0; core < numberOfCores; core++) {
                if (reserved.at(core) || getStateForCore(core) != wantedState) {
                    continue;
                }
                if (isMasterCore(taskIds, threadIds, core)) {
                    continue;
                }

                bool isIdle = core >= (int)taskIds.size() || taskIds.at(core) == -1;
                bool isActiveItemCore = find(activeCoreIds.begin(), activeCoreIds.end(), (unsigned int)core) != activeCoreIds.end();
                if ((pass == 0 && !isIdle) || (pass == 1 && !isActiveItemCore)) {
                    continue;
                }

                desiredCore.at(item) = core;
                reserved.at(core) = true;
                break;
            }
        }

        if (desiredCore.at(item) == -1) {
            desiredCore.at(item) = activeCoreIds.at(item);
            if (debug) {
                cout << "[MigrationH1][debug]: no core found for desired state "
                     << desiredState.at(item) << "; keeping core " << activeCoreIds.at(item) << endl;
            }
        }
    }

    return convertDesiredCoresToMigrations(activeCoreIds, desiredCore, taskIds, threadIds);
}

vector<migration> MigrationH1::convertDesiredCoresToMigrations(const vector<unsigned int> &activeCoreIds,
                                                               const vector<int> &desiredCore,
                                                               const vector<int> &taskIds,
                                                               const vector<int> &threadIds) const
{
    vector<migration> migrations;
    vector<int> itemAtCore(numberOfCores, -1);
    vector<int> coreOfItem(activeCoreIds.begin(), activeCoreIds.end());
    vector<bool> occupied(numberOfCores, false);

    for (int core = 0; core < numberOfCores; core++) {
        occupied.at(core) = core < (int)taskIds.size() && taskIds.at(core) != -1;
    }
    for (unsigned int item = 0; item < activeCoreIds.size(); item++) {
        if (activeCoreIds.at(item) < (unsigned int)numberOfCores) {
            itemAtCore.at(activeCoreIds.at(item)) = item;
            occupied.at(activeCoreIds.at(item)) = true;
        }
    }

    for (unsigned int item = 0; item < desiredCore.size(); item++) {
        while (coreOfItem.at(item) != desiredCore.at(item)) {
            int fromCore = coreOfItem.at(item);
            int toCore = desiredCore.at(item);
            if (toCore < 0 || toCore >= numberOfCores) {
                cout << "[MigrationH1][Warning]: invalid desired core " << toCore << "; skipping migration for item " << item << endl;
                break;
            }
            if (isMasterCore(taskIds, threadIds, fromCore) || isMasterCore(taskIds, threadIds, toCore)) {
                cout << "[MigrationH1][Warning]: requested migration touches frozen master core; skipping from core "
                     << fromCore << " to core " << toCore << endl;
                break;
            }

            int blockingItem = itemAtCore.at(toCore);
            migration move;
            move.fromCore = fromCore;
            move.toCore = toCore;
            move.swap = blockingItem != -1;

            if (move.swap) {
                migrations.push_back(move);

                cout << "[MigrationH1] swap core " << fromCore << " state " << enabledStates.at(getStateForCore(fromCore))
                     << " <-> core " << toCore << " state " << enabledStates.at(getStateForCore(toCore)) << endl;

                itemAtCore.at(toCore) = item;
                itemAtCore.at(fromCore) = blockingItem;
                coreOfItem.at(item) = toCore;
                coreOfItem.at(blockingItem) = fromCore;
            } else {
                if (occupied.at(toCore)) {
                    cout << "[MigrationH1][Warning]: desired core " << toCore
                         << " is occupied by an inactive task; skipping migration for item " << item << endl;
                    break;
                }

                migrations.push_back(move);

                cout << "[MigrationH1] move core " << fromCore << " state " << enabledStates.at(getStateForCore(fromCore))
                     << " -> idle core " << toCore << " state " << enabledStates.at(getStateForCore(toCore)) << endl;

                itemAtCore.at(fromCore) = -1;
                itemAtCore.at(toCore) = item;
                occupied.at(fromCore) = false;
                occupied.at(toCore) = true;
                coreOfItem.at(item) = toCore;
            }
        }
    }

    return migrations;
}

double MigrationH1::predictedTotalIPS(const vector<vector<double> > &predIPS, const vector<int> &states) const
{
    double total = 0.0;
    for (unsigned int item = 0; item < states.size(); item++) {
        if (states.at(item) >= 0) {
            total += predIPS.at(item).at(states.at(item));
        }
    }
    return total;
}

double MigrationH1::predictedCurrentIPS(const vector<unsigned int> &activeCoreIds, const vector<vector<double> > &predIPS) const
{
    double total = 0.0;
    for (unsigned int item = 0; item < activeCoreIds.size(); item++) {
        int state = getStateForCore(activeCoreIds.at(item));
        if (state >= 0) {
            total += predIPS.at(item).at(state);
        }
    }
    return total;
}

void MigrationH1::logPrediction(unsigned int coreId,
                                double measuredIPS,
                                double currentStateValue,
                                const string &benchmarkName,
                                const vector<double> &ips,
                                const vector<double> &power,
                                const vector<double> &temp) const
{
    cout << "[MigrationH1][debug] core=" << coreId
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

vector<migration> MigrationH1::migrate(SubsecondTime time, const vector<int> &taskIds, const vector<bool> &activeCores)
{
    vector<int> threadIds;
    return migrate(time, taskIds, threadIds, activeCores);
}

vector<migration> MigrationH1::migrate(SubsecondTime time, const vector<int> &taskIds, const vector<int> &threadIds, const vector<bool> &activeCores)
{
    vector<migration> empty;

    vector<unsigned int> activeCoreIds = getActiveCoreIds(taskIds, threadIds, activeCores);
    if (activeCoreIds.size() == 0 || enabledStates.size() == 0) {
        return empty;
    }

    if (isTargetIPSObjective() && activeCoreIds.size() < 2) {
        if (debug) {
            cout << "[MigrationH1][debug] skip migration at " << time.getNS()
                 << " ns because active cores < 2" << endl;
        }
        return empty;
    }

    vector<unsigned int> filteredActiveCoreIds;
    const double MIN_ACTIVE_IPS = 0.05; // 0.05 billion IPS = 50 MIPS

    for (unsigned int i = 0; i < activeCoreIds.size(); i++) {
        unsigned int core = activeCoreIds.at(i);
        int stateIndex = getStateForCore(core);

        if (stateIndex < 0) {
            continue;
        }

        double currentStateValue = enabledStates.at(stateIndex);
        double measuredIPS = getMeasuredIPSBillions(core, currentStateValue);

        if (isTargetIPSObjective() && measuredIPS <= MIN_ACTIVE_IPS) {
            if (debug) {
                cout << "[MigrationH1][debug] skip core " << core
                     << " because measuredIPS=" << fixed << setprecision(4)
                     << measuredIPS << endl;
            }
            continue;
        }

        filteredActiveCoreIds.push_back(core);
    }

    activeCoreIds = filteredActiveCoreIds;

    if (activeCoreIds.size() == 0 || (isTargetIPSObjective() && activeCoreIds.size() < 2)) {
        if (debug) {
            cout << "[MigrationH1][debug] skip migration at " << time.getNS()
                 << " ns after filtering inactive cores" << endl;
        }
        return empty;
    }

    vector<vector<double> > predIPS;
    vector<vector<double> > predPower;
    vector<vector<double> > predTemp;
    buildPredictions(activeCoreIds, predIPS, predPower, predTemp);

    double totalPredIPSBefore = predictedCurrentIPS(activeCoreIds, predIPS);

	    if (isThermalObjective()) {
	        ThermalCandidate selected;
	        vector<int> desiredCore = runThermalMaxIPS(activeCoreIds, taskIds, threadIds, predIPS, predPower, predTemp, selected);
	        vector<migration> migrations = convertDesiredCoresToMigrations(activeCoreIds, desiredCore, taskIds, threadIds);

        cout << "[MigrationH1] objective=thermal_max_ips"
             << " time=" << time.getNS()
             << " active=" << activeCoreIds.size()
             << " currentTotalPredIPS=" << fixed << setprecision(4) << totalPredIPSBefore
             << " selectedTotalPredIPS=" << fixed << setprecision(4) << selected.totalIPS
             << " selectedTotalPredPower=" << fixed << setprecision(4) << selected.totalPower
             << " selectedMaxPredTemp=" << fixed << setprecision(4) << selected.maxPredTemp
             << " max_temp=" << fixed << setprecision(4) << maxTemp
             << " thermal_margin=" << fixed << setprecision(4) << thermalMargin
             << " feasible=" << (selected.feasible ? "true" : "false")
             << " migrations=" << migrations.size()
             << endl;
	
	        return migrations;
	    }

	    if (isPowerBudgetObjective()) {
	        vector<int> currentCore(activeCoreIds.begin(), activeCoreIds.end());
	        ThermalCandidate current = evaluatePowerBudgetCandidate(activeCoreIds, currentCore, predIPS, predPower, predTemp);

	        if (powerBudget <= 0.0) {
	            if (!warnedInvalidPowerBudget) {
	                cout << "[MigrationH1][Warning]: objective=power_budget_max_ips but power_budget="
	                     << fixed << setprecision(4) << powerBudget
	                     << "; H1 fixed-VF migration optimization is disabled for this epoch." << endl;
	                warnedInvalidPowerBudget = true;
	            }

	            cout << "[MigrationH1] objective=power_budget_max_ips"
	                 << " time=" << time.getNS()
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
	                 << " migrations=0"
	                 << " reason=invalid_power_budget"
	                 << endl;

	            return empty;
	        }

	        ThermalCandidate selected;
	        vector<int> desiredCore = runPowerBudgetMaxIPS(activeCoreIds, taskIds, threadIds, predIPS, predPower, predTemp, selected);
	        vector<migration> migrations = convertDesiredCoresToMigrations(activeCoreIds, desiredCore, taskIds, threadIds);

	        cout << "[MigrationH1] objective=power_budget_max_ips"
	             << " time=" << time.getNS()
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
	             << " migrations=" << migrations.size()
	             << (selected.feasible ? "" : " reason=no_feasible_power_budget_mapping")
	             << endl;

	        return migrations;
	    }
	
	    if (isTargetIPSMinPowerObjective()) {
	        vector<int> currentCore(activeCoreIds.begin(), activeCoreIds.end());
	        ThermalCandidate current = evaluateTargetIPSCandidate(activeCoreIds, currentCore, predIPS, predPower, predTemp);
	        ThermalCandidate selected;
	        vector<int> desiredCore = runTargetIPSMinPower(activeCoreIds, taskIds, threadIds, predIPS, predPower, predTemp, selected);
	        vector<migration> migrations = convertDesiredCoresToMigrations(activeCoreIds, desiredCore, taskIds, threadIds);

	        cout << "[MigrationH1] objective=target_ips_min_power"
	             << " time=" << time.getNS()
	             << " active=" << activeCoreIds.size()
	             << " target_ips=" << fixed << setprecision(4) << targetIPS
	             << " currentTotalPredIPS=" << fixed << setprecision(4) << current.totalIPS
	             << " selectedTotalPredIPS=" << fixed << setprecision(4) << selected.totalIPS
	             << " currentTotalPredPower=" << fixed << setprecision(4) << current.totalPower
	             << " selectedTotalPredPower=" << fixed << setprecision(4) << selected.totalPower
	             << " selectedMaxProfileTemp=" << fixed << setprecision(4) << selected.maxPredTemp
	             << " feasible=" << (selected.feasible ? "true" : "false")
	             << " migrations=" << migrations.size()
	             << (selected.feasible ? "" : " reason=no_feasible_target_ips_mapping")
	             << endl;

	        return migrations;
	    }
	
	    if (totalPredIPSBefore >= targetIPS) {
        cout << "[MigrationH1] time=" << time.getNS()
             << " active=" << activeCoreIds.size()
             << " totalPredIPSBefore=" << fixed << setprecision(4) << totalPredIPSBefore
             << " target=" << fixed << setprecision(4) << targetIPS
             << " migrations=0 reason=target_already_met"
             << endl;

        return empty;
    }

    vector<int> desiredState = runH1(activeCoreIds, taskIds, threadIds, predIPS, predPower);
    double totalPredIPSAfter = predictedTotalIPS(predIPS, desiredState);

    if (totalPredIPSAfter <= totalPredIPSBefore + 1e-6) {
        cout << "[MigrationH1] time=" << time.getNS()
             << " active=" << activeCoreIds.size()
             << " totalPredIPSBefore=" << fixed << setprecision(4) << totalPredIPSBefore
             << " totalPredIPSAfter=" << fixed << setprecision(4) << totalPredIPSAfter
             << " target=" << fixed << setprecision(4) << targetIPS
             << " migrations=0 reason=no_predicted_improvement"
             << endl;

        return empty;
    }

    vector<migration> migrations = convertDesiredStatesToMigrations(activeCoreIds, desiredState, taskIds, threadIds);

    cout << "[MigrationH1] time=" << time.getNS()
         << " active=" << activeCoreIds.size()
         << " totalPredIPSBefore=" << fixed << setprecision(4) << totalPredIPSBefore
         << " totalPredIPSAfter=" << fixed << setprecision(4) << totalPredIPSAfter
         << " target=" << fixed << setprecision(4) << targetIPS
         << " migrations=" << migrations.size()
         << endl;

    return migrations;
}
