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
    double minTemperatureDelta,
    UInt64 migrationCooldownNs,
    double hysteresisTemperatureDelta,
    double ambientTemperature,
    double thermalResistance,
    double thermalGuardBand)
    : performanceCounters(performanceCounters),
      coreRows(coreRows),
      coreColumns(coreColumns),
      targetIps(targetIps),
      alpha(alpha),
      beta(beta),
      gamma(gamma),
      lambdaTemp(lambdaTemp),
      criticalTemperature(criticalTemperature),
      minTemperatureDelta(minTemperatureDelta),
      migrationCooldownNs(migrationCooldownNs),
      hysteresisTemperatureDelta(hysteresisTemperatureDelta),
      ambientTemperature(ambientTemperature),
      thermalResistance(thermalResistance),
      thermalGuardBand(thermalGuardBand),
      lastMigrationNs(
          coreRows * coreColumns,
          numeric_limits<UInt64>::max()) {
    cout << "[Scheduler][LpThermalRounding]: Initializing with target_ips="
         << targetIps << " alpha=" << alpha << " beta=" << beta
         << " gamma=" << gamma << " lambda_temp=" << lambdaTemp
         << " critical_temperature=" << criticalTemperature
         << " min_temperature_delta=" << minTemperatureDelta
         << " migration_cooldown_ns=" << migrationCooldownNs
         << " hysteresis_temperature_delta=" << hysteresisTemperatureDelta
         << " ambient_temperature=" << ambientTemperature
         << " thermal_resistance=" << thermalResistance
         << " thermal_guard_band=" << thermalGuardBand
         << " effective_critical_temperature="
         << effectiveCriticalTemperature() << endl;
}

vector<migration> LpThermalRounding::migrate(
    SubsecondTime time,
    const vector<int> &taskIds,
    const vector<bool> &activeCores) {

    const unsigned int numberOfCores = coreRows * coreColumns;
    vector<int> sourceCores;
    vector<int> targetCores;
    vector<double> temperatures(numberOfCores, 0.0);
    vector<double> targetTemperatures;

    double minTemperature = numeric_limits<double>::max();
    double maxTemperature = -numeric_limits<double>::max();

    for (unsigned int core = 0; core < numberOfCores; core++) {
        temperatures[core] = measuredTemperature(core);
        minTemperature = min(minTemperature, temperatures[core]);
        maxTemperature = max(maxTemperature, temperatures[core]);

        const bool activeCore = activeCores.at(core);
        const bool freeCore = taskIds.at(core) == -1;

        if (activeCore || freeCore) {
            targetCores.push_back(core);
            targetTemperatures.push_back(temperatures[core]);
        }

        // Only cores with live threads are real migration sources. A core can
        // remain reserved for a task after its thread has exited.
        if (activeCore && !freeCore) {
            sourceCores.push_back(core);
        }
    }

    if (sourceCores.empty()) {
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

    double maxPredictedSourceTemperature = -numeric_limits<double>::max();
    for (unsigned int i = 0; i < sourceCores.size(); i++) {
        for (unsigned int j = 0; j < targetCores.size(); j++) {
            if (targetCores[j] == sourceCores[i]) {
                maxPredictedSourceTemperature = max(
                    maxPredictedSourceTemperature,
                    predictedTemperature(temperatures[sourceCores[i]],
                                         power[i][j]));
                break;
            }
        }
    }

    if (criticalTemperature > 0.0 &&
        maxTemperature < effectiveCriticalTemperature() &&
        maxPredictedSourceTemperature < effectiveCriticalTemperature() &&
        (maxTemperature - minTemperature) < minTemperatureDelta &&
        targetIps <= 0.0) {
        return vector<migration>();
    }

    if (sourceCores.size() == 1) {
        // Tail phase: there is no swap partner left, but inactive cores can
        // still be reserved for the same task after their threads exit.
        const int sourceCore = sourceCores[0];
        const int sourceTask = taskIds.at(sourceCore);
        vector<int> singleTargetCores;
        vector<double> singleTargetTemperatures;

        for (unsigned int core = 0; core < numberOfCores; core++) {
            const bool currentCore = (int)core == sourceCore;
            const bool freeCore = taskIds.at(core) == -1;
            const bool sameTaskReserved = taskIds.at(core) == sourceTask;

            if (currentCore || freeCore || sameTaskReserved) {
                singleTargetCores.push_back(core);
                singleTargetTemperatures.push_back(temperatures[core]);
            }
        }

        if (singleTargetCores.size() < 2) {
            return vector<migration>();
        }

        vector<vector<double> > singleThroughput(1);
        vector<vector<double> > singlePower(1);
        singleThroughput[0].resize(singleTargetCores.size());
        singlePower[0].resize(singleTargetCores.size());

        const double currentIps = measuredIpsBillions(sourceCore);
        const double currentPower = measuredPower(sourceCore);
        int sourceTarget = -1;
        for (unsigned int j = 0; j < singleTargetCores.size(); j++) {
            if (singleTargetCores[j] == sourceCore) {
                sourceTarget = j;
            }
            const double scale = frequencyScale(sourceCore, singleTargetCores[j]);
            singleThroughput[0][j] = currentIps * scale;
            singlePower[0][j] = currentPower * scale;
        }

        if (sourceTarget == -1) {
            return vector<migration>();
        }

        const double sourcePredictedTemperature = predictedTemperature(
            temperatures[sourceCore], singlePower[0][sourceTarget]);
        if (criticalTemperature <= 0.0 ||
            sourcePredictedTemperature < effectiveCriticalTemperature()) {
            return vector<migration>();
        }

        int bestTarget = -1;
        double bestTargetTemperature = sourcePredictedTemperature;
        double bestThroughput = -numeric_limits<double>::max();
        for (unsigned int j = 0; j < singleTargetCores.size(); j++) {
            if ((int)j == sourceTarget) {
                continue;
            }
            if (targetIps > 0.0 && singleThroughput[0][j] < targetIps) {
                continue;
            }

            const double targetPredictedTemperature = predictedTemperature(
                singleTargetTemperatures[j], singlePower[0][j]);
            if (sourcePredictedTemperature - targetPredictedTemperature <
                minTemperatureDelta) {
                continue;
            }

            if (bestTarget == -1 ||
                targetPredictedTemperature < bestTargetTemperature ||
                (fabs(targetPredictedTemperature - bestTargetTemperature) < 1e-12 &&
                 singleThroughput[0][j] > bestThroughput)) {
                bestTarget = j;
                bestTargetTemperature = targetPredictedTemperature;
                bestThroughput = singleThroughput[0][j];
            }
        }

        if (bestTarget == -1) {
            return vector<migration>();
        }

        cout << "[Scheduler][LpThermalRounding]: single-thread thermal escape "
             << "core " << sourceCore << " predicted " << fixed
             << setprecision(1) << sourcePredictedTemperature << " -> core "
             << singleTargetCores[bestTarget] << " predicted "
             << bestTargetTemperature << endl;

        vector<int> assignment(1, bestTarget);
        return createMigrations(
            sourceCores,
            singleTargetCores,
            assignment,
            temperatures,
            singlePower,
            time.getNS());
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
        targetTemperatures);

    if (criticalTemperature > 0.0) {
        vector<bool> usedTargets(targetCores.size(), false);
        for (unsigned int i = 0; i < assignment.size(); i++) {
            usedTargets[assignment[i]] = true;
        }

        const double effCritical = effectiveCriticalTemperature();
        for (unsigned int i = 0; i < assignment.size(); i++) {
            const int oldTarget = assignment[i];
            const double oldTemperature = predictedTemperature(
                targetTemperatures[oldTarget], power[i][oldTarget]);
            if (oldTemperature < effCritical) {
                continue;
            }

            int bestTarget = -1;
            double bestTemperature = oldTemperature;
            double bestThroughput = -numeric_limits<double>::max();
            for (unsigned int newTarget = 0; newTarget < targetCores.size(); newTarget++) {
                if (usedTargets[newTarget]) {
                    continue;
                }

                const double newTemperature = predictedTemperature(
                    targetTemperatures[newTarget], power[i][newTarget]);
                if (oldTemperature - newTemperature < minTemperatureDelta) {
                    continue;
                }

                const double newThroughput = throughput[i][newTarget];
                if (bestTarget == -1 ||
                    newTemperature < bestTemperature ||
                    (fabs(newTemperature - bestTemperature) < 1e-12 &&
                     newThroughput > bestThroughput)) {
                    bestTarget = newTarget;
                    bestTemperature = newTemperature;
                    bestThroughput = newThroughput;
                }
            }

            if (bestTarget != -1) {
                cout << "[Scheduler][LpThermalRounding]: escaping critical "
                     << "target core " << targetCores[oldTarget]
                     << " temp=" << fixed << setprecision(1) << oldTemperature
                     << " -> core " << targetCores[bestTarget]
                     << " temp=" << bestTemperature << endl;
                usedTargets[oldTarget] = false;
                usedTargets[bestTarget] = true;
                assignment[i] = bestTarget;
            }
        }
    }

    repairAssignment(
        assignment,
        throughput,
        power,
        targetTemperatures,
        throughputRequirement);

    // Unconditional thermal relief. Everything above is gated on throughput
    // (repair only fires on a deficit) or can only fill an unused target
    // (the escaping-critical block), so with target_ips == 0 the rounded
    // assignment stays identity and a hot core is never relieved. This pass
    // forces a swap purely on predicted temperature, independent of
    // throughput, so the thread pinned to an over-limit core actually moves.
    thermalSwap(assignment, power, targetTemperatures);

    cout << "[Scheduler][LpThermalRounding]: current_ips=" << fixed
         << setprecision(3) << currentThroughput
         << " target_ips=" << throughputRequirement
         << " final_ips=" << assignmentThroughput(assignment, throughput)
         << " final_power=" << assignmentPower(assignment, power) << endl;

    return createMigrations(
        sourceCores,
        targetCores,
        assignment,
        temperatures,
        power,
        time.getNS());
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

    const double effCritical = effectiveCriticalTemperature();
    const double rampBand = 10.0;
    const double rampStart = effCritical - rampBand;

    for (unsigned int i = 0; i < aLp.size(); i++) {
        score[i].resize(aLp[i].size());
        for (unsigned int j = 0; j < aLp[i].size(); j++) {
            const double thermalPressure =
                normTemp[j] * (0.5 + normPower[i][j]);
            score[i][j] = alpha * aLp[i][j] -
                          beta * normPower[i][j] -
                          gamma * thermalPressure;

            if (criticalTemperature > 0.0) {
                const double predTemp =
                    predictedTemperature(temperature[j], power[i][j]);
                if (predTemp >= effCritical) {
                    // Hard wall: never chosen unless no feasible alternative.
                    score[i][j] -= 1000.0 + (predTemp - effCritical);
                } else if (predTemp > rampStart) {
                    // Quadratic soft ramp so the policy steers away from a
                    // core well before it reaches the limit, while still
                    // tolerating a mildly warm core when nothing cooler can
                    // carry the load. Peak (~100) dominates alpha*aLp (~1).
                    const double frac = (predTemp - rampStart) / rampBand;
                    score[i][j] -= 100.0 * frac * frac;
                }
            }
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
    const double effCritical = effectiveCriticalTemperature();

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

                if (criticalTemperature > 0.0) {
                    // Feed-forward guard: never ratchet a thread back onto a
                    // core whose predicted steady-state temperature exceeds
                    // the effective limit, even if it currently reads cool.
                    // This is what breaks the throughput-driven ping-pong
                    // onto the fast/hot core.
                    const double predNew = predictedTemperature(
                        temperature[newTarget], power[i][newTarget]);
                    const double predOld = predictedTemperature(
                        temperature[oldTarget], power[i][oldTarget]);
                    if (predNew >= effCritical && predNew >= predOld) {
                        continue;
                    }
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

                if (criticalTemperature > 0.0) {
                    // After the swap thread i runs on targetK and thread k
                    // on targetI; forbid any swap that lands a thread on a
                    // core predicted to exceed the effective limit.
                    const double predIonK = predictedTemperature(
                        temperature[targetK], power[i][targetK]);
                    const double predKonI = predictedTemperature(
                        temperature[targetI], power[k][targetI]);
                    if (predIonK >= effCritical || predKonI >= effCritical) {
                        continue;
                    }
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
    const vector<int> &assignment,
    const vector<double> &temperature,
    const vector<vector<double> > &power,
    UInt64 nowNs) {

    const unsigned int numberOfCores = coreRows * coreColumns;
    vector<migration> migrations;
    vector<int> threadCore(sourceCores);
    vector<int> coreThread(numberOfCores, -1);
    vector<int> targetIndexByCore(numberOfCores, -1);
    vector<int> finalCore(sourceCores.size(), -1);

    for (unsigned int j = 0; j < targetCores.size(); j++) {
        if ((unsigned int)targetCores[j] < numberOfCores) {
            targetIndexByCore[targetCores[j]] = j;
        }
    }

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

            double sourcePredictedTemperature = temperature[fromCore];
            double targetPredictedTemperature = temperature[toCore];
            const int fromTarget = targetIndexByCore[fromCore];
            const int toTarget = assignment[thread];
            if (fromTarget >= 0 &&
                (unsigned int)fromTarget < power[thread].size()) {
                sourcePredictedTemperature = predictedTemperature(
                    temperature[fromCore], power[thread][fromTarget]);
            }
            if (toTarget >= 0 &&
                (unsigned int)toTarget < power[thread].size()) {
                targetPredictedTemperature = predictedTemperature(
                    temperature[toCore], power[thread][toTarget]);
            }

            const int occupant = coreThread[toCore];
            if (occupant == -1) {
                m.swap = false;
                if (!migrationPassesHysteresis(
                        m,
                        temperature,
                        sourcePredictedTemperature,
                        targetPredictedTemperature,
                        nowNs)) {
                    continue;
                }
                coreThread[fromCore] = -1;
                coreThread[toCore] = thread;
                threadCore[thread] = toCore;
            } else {
                m.swap = true;
                if (!migrationPassesHysteresis(
                        m,
                        temperature,
                        sourcePredictedTemperature,
                        targetPredictedTemperature,
                        nowNs)) {
                    continue;
                }
                coreThread[fromCore] = occupant;
                coreThread[toCore] = thread;
                threadCore[occupant] = fromCore;
                threadCore[thread] = toCore;
            }

            migrations.push_back(m);
            changed = true;
        }
    }

    recordMigrations(migrations, nowNs);

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

bool LpThermalRounding::migrationPassesHysteresis(
    const migration &migration,
    const vector<double> &temperature,
    double sourcePredictedTemperature,
    double targetPredictedTemperature,
    UInt64 nowNs) const {

    if (migration.fromCore >= temperature.size() ||
        migration.toCore >= temperature.size()) {
        return true;
    }

    const double sourceTemperature = temperature[migration.fromCore];
    const double targetTemperature = temperature[migration.toCore];
    const double sourceThermalPressure =
        max(sourceTemperature, sourcePredictedTemperature);
    const double targetThermalPressure =
        max(targetTemperature, targetPredictedTemperature);
    const bool sourceCritical =
        criticalTemperature > 0.0 &&
        sourceThermalPressure >= effectiveCriticalTemperature();

    // The destination cooldown is ALWAYS enforced, even when the source is
    // critical. Previously a critical source bypassed every cooldown and
    // could dump its thread onto a core that was migrated to one epoch ago
    // and is still heating, producing the core2<->core3 ping-pong. A hot
    // core may shed load, but never onto an unsettled core.
    if (coreCoolingDown(migration.toCore, nowNs)) {
        cout << "[Scheduler][LpThermalRounding]: hysteresis skipped migration "
             << migration.fromCore << " -> " << migration.toCore
             << " because destination is cooling down" << endl;
        return false;
    }

    // The source cooldown is waived only when the source is critical and
    // genuinely must escape; otherwise an unsettled source also blocks.
    if (!sourceCritical && coreCoolingDown(migration.fromCore, nowNs)) {
        cout << "[Scheduler][LpThermalRounding]: hysteresis skipped migration "
             << migration.fromCore << " -> " << migration.toCore
             << " because source is cooling down" << endl;
        return false;
    }

    const double requiredDelta =
        sourceCritical ? minTemperatureDelta : hysteresisTemperatureDelta;
    if (requiredDelta > 0.0 &&
        sourceThermalPressure - targetThermalPressure < requiredDelta) {
        cout << "[Scheduler][LpThermalRounding]: hysteresis skipped migration "
             << migration.fromCore << " -> " << migration.toCore
             << " source_temp=" << fixed << setprecision(1)
             << sourceTemperature << " target_temp=" << targetTemperature
             << " source_pred=" << sourceThermalPressure
             << " target_pred=" << targetThermalPressure
             << " required_delta=" << requiredDelta << endl;
        return false;
    }

    return true;
}

void LpThermalRounding::recordMigrations(
    const vector<migration> &migrations,
    UInt64 nowNs) {

    for (unsigned int i = 0; i < migrations.size(); i++) {
        const migration &m = migrations[i];
        if (m.fromCore < lastMigrationNs.size()) {
            lastMigrationNs[m.fromCore] = nowNs;
        }
        if (m.toCore < lastMigrationNs.size()) {
            lastMigrationNs[m.toCore] = nowNs;
        }
    }
}

bool LpThermalRounding::coreCoolingDown(
    unsigned int coreId,
    UInt64 nowNs) const {

    if (migrationCooldownNs == 0 ||
        coreId >= lastMigrationNs.size() ||
        lastMigrationNs[coreId] == numeric_limits<UInt64>::max()) {
        return false;
    }

    if (nowNs < lastMigrationNs[coreId]) {
        return true;
    }

    return nowNs - lastMigrationNs[coreId] < migrationCooldownNs;
}

bool LpThermalRounding::finiteAndPositive(double value) const {
    return value > 0.0 && std::isfinite(value);
}

double LpThermalRounding::effectiveCriticalTemperature() const {
    if (criticalTemperature <= 0.0) {
        return criticalTemperature;
    }

    double effective = criticalTemperature - thermalGuardBand;
    if (effective < ambientTemperature) {
        effective = ambientTemperature;
    }
    return effective;
}

// Feed-forward thermal estimate. Because every core runs at a fixed
// frequency, power[i][j] is the steady-state power a thread would draw on
// the target core. The HotSpot RC model has a long time constant relative
// to the 1 ms migration epoch, so the instantaneous reading lags reality
// and a core that looks cool now will overshoot within one epoch. Using
// ambient + Rth * predictedPower lets the policy avoid a hot assignment
// before the core actually heats up. The larger of the two estimates is
// used so neither lag nor an underestimated Rth can hide a hotspot.
double LpThermalRounding::predictedTemperature(
    double instantaneousTemperature,
    double predictedPower) const {

    if (thermalResistance <= 0.0) {
        return instantaneousTemperature;
    }

    const double steadyState =
        ambientTemperature + thermalResistance * predictedPower;
    return max(instantaneousTemperature, steadyState);
}

// Pure steady-state projection used to judge whether a swap *helps*. Unlike
// predictedTemperature(), this deliberately does NOT clamp to the current
// reading: a core sitting at 94 C will still read 94 C instantaneously the
// moment its thread leaves, so taking max(instantaneous, .) would make every
// relief swap look like a zero-improvement and nothing would ever move. The
// thread the core is about to receive determines where it settles, so the
// benefit comparison must be on ambient + Rth * thread_power alone.
double LpThermalRounding::projectedCoreTemperature(
    double instantaneousTemperature,
    double threadPower) const {

    if (thermalResistance <= 0.0) {
        // No feed-forward model available; fall back to the measured
        // temperature so the swap at least reacts to a real hot core.
        return instantaneousTemperature;
    }
    return ambientTemperature + thermalResistance * threadPower;
}

void LpThermalRounding::thermalSwap(
    vector<int> &assignment,
    const vector<vector<double> > &power,
    const vector<double> &temperature) const {

    if (criticalTemperature <= 0.0) {
        return;
    }

    const double effCritical = effectiveCriticalTemperature();
    const unsigned int numThreads = assignment.size();

    bool changed = true;
    while (changed) {
        changed = false;

        // Trigger on the thread whose core has the worst lag-aware predicted
        // temperature above the effective limit. predictedTemperature() keeps
        // the conservative max() here so a core that is physically hot right
        // now (e.g. core 2 at 94 C) is recognised as an emergency even before
        // the steady-state estimate catches up.
        int hotThread = -1;
        double worstPredicted = effCritical;
        for (unsigned int i = 0; i < numThreads; i++) {
            const double pred = predictedTemperature(
                temperature[assignment[i]], power[i][assignment[i]]);
            if (pred > worstPredicted) {
                worstPredicted = pred;
                hotThread = i;
            }
        }
        if (hotThread == -1) {
            break;
        }

        const int hotTarget = assignment[hotThread];

        // Pick the swap partner that most reduces the projected peak of the
        // two affected cores. Benefit is judged on the steady-state
        // projection so relieving the hot core is actually visible.
        int bestThread = -1;
        double bestPeakDrop = minTemperatureDelta;
        for (unsigned int k = 0; k < numThreads; k++) {
            if ((int)k == hotThread || assignment[k] == hotTarget) {
                continue;
            }
            const int coldTarget = assignment[k];

            const double beforePeak = max(
                projectedCoreTemperature(
                    temperature[hotTarget], power[hotThread][hotTarget]),
                projectedCoreTemperature(
                    temperature[coldTarget], power[k][coldTarget]));
            const double afterPeak = max(
                projectedCoreTemperature(
                    temperature[hotTarget], power[k][hotTarget]),
                projectedCoreTemperature(
                    temperature[coldTarget], power[hotThread][coldTarget]));

            const double drop = beforePeak - afterPeak;
            if (drop > bestPeakDrop) {
                bestPeakDrop = drop;
                bestThread = k;
            }
        }

        if (bestThread == -1) {
            // No swap lowers the projected peak: migration alone cannot help
            // this hotspot (the physics limit under fixed frequency).
            cout << "[Scheduler][LpThermalRounding]: thermal swap could not "
                 << "relieve target idx " << hotTarget << " predicted "
                 << fixed << setprecision(1) << worstPredicted << endl;
            break;
        }

        cout << "[Scheduler][LpThermalRounding]: thermal swap target idx "
             << hotTarget << " predicted " << fixed << setprecision(1)
             << worstPredicted << " thread " << hotThread << " <-> thread "
             << bestThread << " (idx " << assignment[bestThread]
             << ") projected_peak_drop=" << bestPeakDrop << endl;
        swap(assignment[hotThread], assignment[bestThread]);
        changed = true;
    }
}
