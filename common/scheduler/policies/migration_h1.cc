#include "migration_h1.h"

#include <algorithm>
#include <cmath>
#include <fstream>
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
                         const string &profileFile,
                         bool debug)
    : performanceCounters(performanceCounters)
    , numberOfCores(numberOfCores)
    , coreToState(coreToState)
    , enabledStates(enabledStates)
    , targetIPS(targetIPS)
    , profileFile(profileFile)
    , debug(debug)
    , warnedFallback(false)
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

vector<unsigned int> MigrationH1::getActiveCoreIds(const vector<int> &taskIds, const vector<bool> &activeCores) const
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

void MigrationH1::buildPredictions(const vector<unsigned int> &activeCoreIds,
                                   vector<vector<double> > &predIPS,
                                   vector<vector<double> > &predPower)
{
    predIPS.assign(activeCoreIds.size(), vector<double>(enabledStates.size(), 0.0));
    predPower.assign(activeCoreIds.size(), vector<double>(enabledStates.size(), 0.0));

    for (unsigned int item = 0; item < activeCoreIds.size(); item++) {
        unsigned int coreId = activeCoreIds.at(item);
        int currentStateIndex = getStateForCore(coreId);
        double currentStateValue = enabledStates.at(currentStateIndex);
        double measuredIPS = getMeasuredIPSBillions(coreId, currentStateValue);
        double measuredPower = getMeasuredPower(coreId);

        string nearestBenchmark = findNearestBenchmark(currentStateValue, measuredIPS);
        if (nearestBenchmark == "") {
            if (!warnedFallback) {
                cout << "[MigrationH1][Warning]: no valid nearest benchmark with all enabled states found; using measured fallback predictions." << endl;
                warnedFallback = true;
            }

            for (unsigned int state = 0; state < enabledStates.size(); state++) {
                double ratio = currentStateValue > 0.0 ? enabledStates.at(state) / currentStateValue : 1.0;
                predIPS.at(item).at(state) = measuredIPS * ratio;
                predPower.at(item).at(state) = measuredPower;
            }
        } else {
            ProfileMap::const_iterator benchmark = profile.find(nearestBenchmark);
            for (unsigned int state = 0; state < enabledStates.size(); state++) {
                StateProfile::const_iterator entry = benchmark->second.find(stateKey(enabledStates.at(state)));
                if (entry == benchmark->second.end()) {
                    predIPS.at(item).at(state) = measuredIPS;
                    predPower.at(item).at(state) = measuredPower;
                } else {
                    predIPS.at(item).at(state) = entry->second.ips;
                    predPower.at(item).at(state) = entry->second.power;
                }
            }
        }

        if (debug) {
            logPrediction(coreId, measuredIPS, currentStateValue, nearestBenchmark, predIPS.at(item), predPower.at(item));
        }
    }
}

vector<int> MigrationH1::runH1(const vector<unsigned int> &activeCoreIds,
                               const vector<vector<double> > &predIPS,
                               const vector<vector<double> > &predPower)
{
    vector<int> desiredState(activeCoreIds.size(), -1);
    if (enabledStates.size() == 0 || activeCoreIds.size() == 0) {
        return desiredState;
    }

    vector<int> capacity(enabledStates.size(), 0);
    for (int core = 0; core < numberOfCores; core++) {
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

vector<migration> MigrationH1::convertDesiredStatesToMigrations(const vector<unsigned int> &activeCoreIds,
                                                                 const vector<int> &desiredState,
                                                                 const vector<bool> &activeCores) const
{
    vector<migration> migrations;
    vector<bool> handled(activeCoreIds.size(), false);

    vector<bool> coreOccupied(numberOfCores, false);
    for (int core = 0; core < numberOfCores && core < (int)activeCores.size(); core++) {
        coreOccupied.at(core) = activeCores.at(core);
    }

    for (unsigned int item = 0; item < activeCoreIds.size(); item++) {
        if (handled.at(item)) {
            continue;
        }

        unsigned int fromCore = activeCoreIds.at(item);
        int currentState = getStateForCore(fromCore);
        int wantedState = desiredState.at(item);
        if (currentState == wantedState) {
            handled.at(item) = true;
            continue;
        }

        for (unsigned int other = item + 1; other < activeCoreIds.size(); other++) {
            if (handled.at(other)) {
                continue;
            }

            unsigned int toCore = activeCoreIds.at(other);
            int otherCurrentState = getStateForCore(toCore);
            int otherWantedState = desiredState.at(other);
            if (otherCurrentState == wantedState && otherWantedState == currentState) {
                migration move;
                move.fromCore = fromCore;
                move.toCore = toCore;
                move.swap = true;
                migrations.push_back(move);

                cout << "[MigrationH1] swap core " << fromCore << " state " << enabledStates.at(currentState)
                     << " <-> core " << toCore << " state " << enabledStates.at(otherCurrentState) << endl;

                handled.at(item) = true;
                handled.at(other) = true;
                break;
            }
        }
    }

    bool progress = true;
    while (progress) {
        progress = false;
        for (unsigned int item = 0; item < activeCoreIds.size(); item++) {
            if (handled.at(item)) {
                continue;
            }

            unsigned int fromCore = activeCoreIds.at(item);
            int currentState = getStateForCore(fromCore);
            int wantedState = desiredState.at(item);
            if (currentState == wantedState) {
                handled.at(item) = true;
                progress = true;
                continue;
            }

            for (int targetCore = 0; targetCore < numberOfCores; targetCore++) {
                if (getStateForCore(targetCore) != wantedState || coreOccupied.at(targetCore)) {
                    continue;
                }

                migration move;
                move.fromCore = fromCore;
                move.toCore = targetCore;
                move.swap = false;
                migrations.push_back(move);

                cout << "[MigrationH1] move core " << fromCore << " state " << enabledStates.at(currentState)
                     << " -> idle core " << targetCore << " state " << enabledStates.at(wantedState) << endl;

                coreOccupied.at(fromCore) = false;
                coreOccupied.at(targetCore) = true;
                handled.at(item) = true;
                progress = true;
                break;
            }
        }
    }

    if (debug) {
        for (unsigned int item = 0; item < activeCoreIds.size(); item++) {
            if (!handled.at(item)) {
                unsigned int core = activeCoreIds.at(item);
                cout << "[MigrationH1][debug]: skipped non-reciprocal migration from core " << core
                     << " state " << enabledStates.at(getStateForCore(core))
                     << " to state " << enabledStates.at(desiredState.at(item)) << endl;
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
                                const vector<double> &power) const
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
    cout << "]" << endl;
}

vector<migration> MigrationH1::migrate(SubsecondTime time, const vector<int> &taskIds, const vector<bool> &activeCores)
{
    vector<unsigned int> activeCoreIds = getActiveCoreIds(taskIds, activeCores);
    if (activeCoreIds.size() == 0 || enabledStates.size() == 0) {
        vector<migration> empty;
        return empty;
    }

    vector<vector<double> > predIPS;
    vector<vector<double> > predPower;
    buildPredictions(activeCoreIds, predIPS, predPower);

    double totalPredIPSBefore = predictedCurrentIPS(activeCoreIds, predIPS);
    vector<int> desiredState = runH1(activeCoreIds, predIPS, predPower);
    double totalPredIPSAfter = predictedTotalIPS(predIPS, desiredState);
    vector<migration> migrations = convertDesiredStatesToMigrations(activeCoreIds, desiredState, activeCores);

    cout << "[MigrationH1] time=" << time.getNS()
         << " active=" << activeCoreIds.size()
         << " totalPredIPSBefore=" << fixed << setprecision(4) << totalPredIPSBefore
         << " totalPredIPSAfter=" << fixed << setprecision(4) << totalPredIPSAfter
         << " target=" << fixed << setprecision(4) << targetIPS
         << " migrations=" << migrations.size() << endl;

    return migrations;
}
