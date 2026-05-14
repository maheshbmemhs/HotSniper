#include "lpThermalRounding.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

using namespace std;

namespace {
struct Candidate {
    double score;
    int thread;
    int target;
};

bool candidateGreater(const Candidate &a, const Candidate &b) {
    return a.score > b.score;
}

double clamp01(double value) {
    if (value < 0.0) {
        return 0.0;
    }
    if (value > 1.0) {
        return 1.0;
    }
    return value;
}

vector<vector<double> > normalizeMatrix(const vector<vector<double> > &values) {
    double minValue = numeric_limits<double>::max();
    double maxValue = -numeric_limits<double>::max();

    for (unsigned int i = 0; i < values.size(); i++) {
        for (unsigned int j = 0; j < values[i].size(); j++) {
            minValue = min(minValue, values[i][j]);
            maxValue = max(maxValue, values[i][j]);
        }
    }

    vector<vector<double> > normalized(values.size());
    if (fabs(maxValue - minValue) < 1e-12) {
        for (unsigned int i = 0; i < values.size(); i++) {
            normalized[i].assign(values[i].size(), 0.0);
        }
        return normalized;
    }

    for (unsigned int i = 0; i < values.size(); i++) {
        normalized[i].resize(values[i].size());
        for (unsigned int j = 0; j < values[i].size(); j++) {
            normalized[i][j] = (values[i][j] - minValue) / (maxValue - minValue);
        }
    }

    return normalized;
}

vector<double> normalizeVector(const vector<double> &values) {
    double minValue = numeric_limits<double>::max();
    double maxValue = -numeric_limits<double>::max();

    for (unsigned int i = 0; i < values.size(); i++) {
        minValue = min(minValue, values[i]);
        maxValue = max(maxValue, values[i]);
    }

    vector<double> normalized(values.size(), 0.0);
    if (fabs(maxValue - minValue) < 1e-12) {
        return normalized;
    }

    for (unsigned int i = 0; i < values.size(); i++) {
        normalized[i] = (values[i] - minValue) / (maxValue - minValue);
    }

    return normalized;
}
}

LpThermalRounding::LpThermalRounding(
    const PerformanceCounters *performanceCounters,
    int coreRows,
    int coreColumns,
    double targetIps,
    double alpha,
    double beta,
    double gamma,
    double lambdaTemp,
    double criticalTemperature,
    double minTemperatureDelta)
    : performanceCounters(performanceCounters),
      coreRows(coreRows),
      coreColumns(coreColumns),
      targetIps(targetIps),
      alpha(alpha),
      beta(beta),
      gamma(gamma),
      lambdaTemp(lambdaTemp),
      criticalTemperature(criticalTemperature),
      minTemperatureDelta(minTemperatureDelta) {
    cout << "[Scheduler][LpThermalRounding]: Initializing with target_ips="
         << targetIps << " alpha=" << alpha << " beta=" << beta
         << " gamma=" << gamma << " lambda_temp=" << lambdaTemp
         << " critical_temperature=" << criticalTemperature
         << " min_temperature_delta=" << minTemperatureDelta << endl;
}

vector<migration> LpThermalRounding::migrate(
    SubsecondTime time,
    const vector<int> &taskIds,
    const vector<bool> &activeCores) {

    const unsigned int numberOfCores = coreRows * coreColumns;
    vector<int> sourceCores;
    vector<int> targetCores;
    vector<double> temperatures(numberOfCores, 0.0);

    double minTemperature = numeric_limits<double>::max();
    double maxTemperature = -numeric_limits<double>::max();

    for (unsigned int core = 0; core < numberOfCores; core++) {
        targetCores.push_back(core);
        temperatures[core] = measuredTemperature(core);
        minTemperature = min(minTemperature, temperatures[core]);
        maxTemperature = max(maxTemperature, temperatures[core]);

        if (taskIds.at(core) != -1) {
            sourceCores.push_back(core);
        }
    }

    if (sourceCores.size() < 2) {
        return vector<migration>();
    }

    if (criticalTemperature > 0.0 &&
        maxTemperature < criticalTemperature &&
        (maxTemperature - minTemperature) < minTemperatureDelta &&
        targetIps <= 0.0) {
        return vector<migration>();
    }

    vector<vector<double> > throughput(sourceCores.size());
    vector<vector<double> > power(sourceCores.size());
    double currentThroughput = 0.0;

    for (unsigned int i = 0; i < sourceCores.size(); i++) {
        const unsigned int fromCore = sourceCores[i];
        const double currentIps = measuredIpsBillions(fromCore);
        const double currentPower = measuredPower(fromCore);
        currentThroughput += currentIps;

        throughput[i].resize(targetCores.size());
        power[i].resize(targetCores.size());

        for (unsigned int j = 0; j < targetCores.size(); j++) {
            const unsigned int toCore = targetCores[j];
            const double scale = frequencyScale(fromCore, toCore);
            throughput[i][j] = currentIps * scale;
            power[i][j] = currentPower * scale;
        }
    }

    const double throughputRequirement =
        targetIps > 0.0 ? targetIps : currentThroughput;

    vector<vector<double> > aLp = solveLpRelaxation(
        throughput,
        power,
        throughputRequirement);

    vector<int> assignment = thermalAwareCapacityRounding(
        aLp,
        power,
        temperatures);

    repairAssignment(
        assignment,
        throughput,
        power,
        temperatures,
        throughputRequirement);

    cout << "[Scheduler][LpThermalRounding]: current_ips=" << fixed
         << setprecision(3) << currentThroughput
         << " target_ips=" << throughputRequirement
         << " final_ips=" << assignmentThroughput(assignment, throughput)
         << " final_power=" << assignmentPower(assignment, power) << endl;

    return createMigrations(sourceCores, targetCores, assignment);
}

vector<vector<double> > LpThermalRounding::solveLpRelaxation(
    const vector<vector<double> > &throughput,
    const vector<vector<double> > &power,
    double throughputRequirement) const {

    vector<vector<double> > lowPowerScore(power.size());
    vector<vector<double> > highThroughputScore(throughput.size());

    for (unsigned int i = 0; i < power.size(); i++) {
        lowPowerScore[i].resize(power[i].size());
        highThroughputScore[i].resize(throughput[i].size());
        for (unsigned int j = 0; j < power[i].size(); j++) {
            lowPowerScore[i][j] = -power[i][j];
            highThroughputScore[i][j] = throughput[i][j];
        }
    }

    vector<int> lowPowerAssignment = greedyAssignment(lowPowerScore);
    vector<int> highThroughputAssignment = greedyAssignment(highThroughputScore);

    const double lowPowerThroughput =
        assignmentThroughput(lowPowerAssignment, throughput);
    const double highThroughput =
        assignmentThroughput(highThroughputAssignment, throughput);

    vector<vector<double> > lowMatrix =
        assignmentToMatrix(lowPowerAssignment, power[0].size());
    vector<vector<double> > highMatrix =
        assignmentToMatrix(highThroughputAssignment, power[0].size());

    double mix = 0.0;
    if (lowPowerThroughput < throughputRequirement &&
        highThroughput > lowPowerThroughput) {
        mix = (throughputRequirement - lowPowerThroughput) /
              (highThroughput - lowPowerThroughput);
        mix = clamp01(mix);
    }

    vector<vector<double> > aLp(lowMatrix.size());
    for (unsigned int i = 0; i < lowMatrix.size(); i++) {
        aLp[i].resize(lowMatrix[i].size());
        for (unsigned int j = 0; j < lowMatrix[i].size(); j++) {
            aLp[i][j] = (1.0 - mix) * lowMatrix[i][j] + mix * highMatrix[i][j];
        }
    }

    cout << "[Scheduler][LpThermalRounding]: LP relaxation mix=" << fixed
         << setprecision(3) << mix
         << " low_power_ips=" << lowPowerThroughput
         << " high_ips=" << highThroughput << endl;

    return aLp;
}

vector<int> LpThermalRounding::thermalAwareCapacityRounding(
    const vector<vector<double> > &aLp,
    const vector<vector<double> > &power,
    const vector<double> &temperature) const {

    vector<vector<double> > normPower = normalizeMatrix(power);
    vector<double> normTemp = normalizeVector(temperature);
    vector<vector<double> > score(aLp.size());

    for (unsigned int i = 0; i < aLp.size(); i++) {
        score[i].resize(aLp[i].size());
        for (unsigned int j = 0; j < aLp[i].size(); j++) {
            const double thermalPressure =
                normTemp[j] * (0.5 + normPower[i][j]);
            score[i][j] = alpha * aLp[i][j] -
                          beta * normPower[i][j] -
                          gamma * thermalPressure;
        }
    }

    return greedyAssignment(score);
}

void LpThermalRounding::repairAssignment(
    vector<int> &assignment,
    const vector<vector<double> > &throughput,
    const vector<vector<double> > &power,
    const vector<double> &temperature,
    double throughputRequirement) const {

    const double epsilon = 1e-9;
    const unsigned int numThreads = assignment.size();
    const unsigned int numTargets = throughput[0].size();

    while (assignmentThroughput(assignment, throughput) < throughputRequirement) {
        double bestScore = -numeric_limits<double>::max();
        int bestThread = -1;
        int bestOtherThread = -1;
        int bestTarget = -1;
        bool bestIsSwap = false;

        vector<bool> used(numTargets, false);
        for (unsigned int i = 0; i < assignment.size(); i++) {
            used[assignment[i]] = true;
        }

        for (unsigned int i = 0; i < numThreads; i++) {
            const int oldTarget = assignment[i];
            for (unsigned int newTarget = 0; newTarget < numTargets; newTarget++) {
                if ((int)newTarget == oldTarget || used[newTarget]) {
                    continue;
                }

                const double deltaThroughput =
                    throughput[i][newTarget] - throughput[i][oldTarget];
                if (deltaThroughput <= 0.0) {
                    continue;
                }

                const double deltaPower =
                    power[i][newTarget] - power[i][oldTarget];
                const double deltaTemp =
                    max(0.0, temperature[newTarget] - temperature[oldTarget]);
                const double score =
                    deltaThroughput /
                    (max(deltaPower, 0.0) + lambdaTemp * deltaTemp + epsilon);

                if (score > bestScore) {
                    bestScore = score;
                    bestThread = i;
                    bestTarget = newTarget;
                    bestIsSwap = false;
                }
            }
        }

        for (unsigned int i = 0; i < numThreads; i++) {
            const int targetI = assignment[i];
            for (unsigned int k = i + 1; k < numThreads; k++) {
                const int targetK = assignment[k];
                if (targetI == targetK) {
                    continue;
                }

                const double deltaThroughput =
                    throughput[i][targetK] + throughput[k][targetI] -
                    throughput[i][targetI] - throughput[k][targetK];
                if (deltaThroughput <= 0.0) {
                    continue;
                }

                const double deltaPower =
                    power[i][targetK] + power[k][targetI] -
                    power[i][targetI] - power[k][targetK];
                const double deltaTemp =
                    fabs(temperature[targetK] - temperature[targetI]);
                const double score =
                    deltaThroughput /
                    (max(deltaPower, 0.0) + lambdaTemp * deltaTemp + epsilon);

                if (score > bestScore) {
                    bestScore = score;
                    bestThread = i;
                    bestOtherThread = k;
                    bestIsSwap = true;
                }
            }
        }

        if (bestThread == -1) {
            cout << "[Scheduler][LpThermalRounding]: repair could not meet "
                 << "target throughput; keeping best feasible assignment" << endl;
            break;
        }

        if (bestIsSwap) {
            swap(assignment[bestThread], assignment[bestOtherThread]);
        } else {
            assignment[bestThread] = bestTarget;
        }
    }
}

vector<migration> LpThermalRounding::createMigrations(
    const vector<int> &sourceCores,
    const vector<int> &targetCores,
    const vector<int> &assignment) const {

    const unsigned int numberOfCores = coreRows * coreColumns;
    vector<migration> migrations;
    vector<int> threadCore(sourceCores);
    vector<int> coreThread(numberOfCores, -1);
    vector<int> finalCore(sourceCores.size(), -1);

    for (unsigned int i = 0; i < sourceCores.size(); i++) {
        coreThread[sourceCores[i]] = i;
        finalCore[i] = targetCores[assignment[i]];
    }

    bool changed = true;
    while (changed) {
        changed = false;
        for (unsigned int thread = 0; thread < threadCore.size(); thread++) {
            const int fromCore = threadCore[thread];
            const int toCore = finalCore[thread];
            if (fromCore == toCore) {
                continue;
            }

            migration m;
            m.fromCore = fromCore;
            m.toCore = toCore;

            const int occupant = coreThread[toCore];
            if (occupant == -1) {
                m.swap = false;
                coreThread[fromCore] = -1;
                coreThread[toCore] = thread;
                threadCore[thread] = toCore;
            } else {
                m.swap = true;
                coreThread[fromCore] = occupant;
                coreThread[toCore] = thread;
                threadCore[occupant] = fromCore;
                threadCore[thread] = toCore;
            }

            migrations.push_back(m);
            changed = true;
        }
    }

    cout << "[Scheduler][LpThermalRounding]: migrations=" << migrations.size()
         << endl;
    for (unsigned int i = 0; i < migrations.size(); i++) {
        cout << "[Scheduler][LpThermalRounding]: migration " << i
             << " from core " << migrations[i].fromCore
             << " to core " << migrations[i].toCore
             << " swap=" << (migrations[i].swap ? "true" : "false")
             << endl;
    }

    return migrations;
}

vector<int> LpThermalRounding::greedyAssignment(
    const vector<vector<double> > &score) const {

    vector<Candidate> candidates;
    for (unsigned int i = 0; i < score.size(); i++) {
        for (unsigned int j = 0; j < score[i].size(); j++) {
            Candidate candidate;
            candidate.score = score[i][j];
            candidate.thread = i;
            candidate.target = j;
            candidates.push_back(candidate);
        }
    }

    sort(candidates.begin(), candidates.end(), candidateGreater);

    vector<int> assignment(score.size(), -1);
    vector<bool> targetUsed(score[0].size(), false);

    for (unsigned int c = 0; c < candidates.size(); c++) {
        const int thread = candidates[c].thread;
        const int target = candidates[c].target;
        if (assignment[thread] == -1 && !targetUsed[target]) {
            assignment[thread] = target;
            targetUsed[target] = true;
        }
    }

    for (unsigned int i = 0; i < assignment.size(); i++) {
        if (assignment[i] != -1) {
            continue;
        }

        for (unsigned int j = 0; j < targetUsed.size(); j++) {
            if (!targetUsed[j]) {
                assignment[i] = j;
                targetUsed[j] = true;
                break;
            }
        }
    }

    return assignment;
}

vector<vector<double> > LpThermalRounding::assignmentToMatrix(
    const vector<int> &assignment,
    unsigned int numTargets) const {

    vector<vector<double> > matrix(assignment.size());
    for (unsigned int i = 0; i < assignment.size(); i++) {
        matrix[i].assign(numTargets, 0.0);
        if (assignment[i] >= 0) {
            matrix[i][assignment[i]] = 1.0;
        }
    }
    return matrix;
}

double LpThermalRounding::assignmentThroughput(
    const vector<int> &assignment,
    const vector<vector<double> > &throughput) const {

    double total = 0.0;
    for (unsigned int i = 0; i < assignment.size(); i++) {
        total += throughput[i][assignment[i]];
    }
    return total;
}

double LpThermalRounding::assignmentPower(
    const vector<int> &assignment,
    const vector<vector<double> > &power) const {

    double total = 0.0;
    for (unsigned int i = 0; i < assignment.size(); i++) {
        total += power[i][assignment[i]];
    }
    return total;
}

double LpThermalRounding::measuredIpsBillions(unsigned int coreId) const {
    if (performanceCounters == NULL) {
        return 0.0;
    }

    try {
        const double rawIps = performanceCounters->getIPSOfCore(coreId);
        if (finiteAndPositive(rawIps)) {
            return rawIps / 1e9;
        }
    } catch (...) {
    }

    return 0.0;
}

double LpThermalRounding::measuredPower(unsigned int coreId) const {
    if (performanceCounters == NULL) {
        return 1.0;
    }

    try {
        const double power = performanceCounters->getPowerOfCore(coreId);
        if (finiteAndPositive(power)) {
            return power;
        }
    } catch (...) {
    }

    return 1.0;
}

double LpThermalRounding::measuredTemperature(unsigned int coreId) const {
    if (performanceCounters == NULL) {
        return 0.0;
    }

    try {
        const double temperature = performanceCounters->getTemperatureOfCore(coreId);
        if (temperature >= 0.0 && std::isfinite(temperature)) {
            return temperature;
        }
    } catch (...) {
    }

    return 0.0;
}

double LpThermalRounding::frequencyScale(
    unsigned int fromCore,
    unsigned int toCore) const {

    if (performanceCounters == NULL) {
        return 1.0;
    }

    try {
        const int fromFreq = performanceCounters->getFreqOfCore(fromCore);
        const int toFreq = performanceCounters->getFreqOfCore(toCore);
        if (fromFreq > 0 && toFreq > 0) {
            return (double)toFreq / (double)fromFreq;
        }
    } catch (...) {
    }

    return 1.0;
}

bool LpThermalRounding::finiteAndPositive(double value) const {
    return value > 0.0 && std::isfinite(value);
}
