#include "dynThreadMapping.h"
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <algorithm>

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

float nearestCoreState(const std::vector<float> &core_states, double freq_ghz)
{
    if (core_states.empty()) {
        return 0.0f;
    }

    float best = core_states.front();
    double best_diff = std::fabs(freq_ghz - best);
    for (float state : core_states) {
        double diff = std::fabs(freq_ghz - state);
        if (diff < best_diff) {
            best = state;
            best_diff = diff;
        }
    }
    return best;
}

bool incrementFrequencyIndices(std::vector<size_t> &indices, size_t base)
{
    if (indices.empty() || base == 0) {
        return false;
    }

    for (size_t i = 0; i < indices.size(); ++i) {
        ++indices[i];
        if (indices[i] < base) {
            return true;
        }
        indices[i] = 0;
    }
    return false;
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

}


void DynThreadMapping::logUtilizations(const std::vector<int> &coreIds) {
    std::cout << "[Scheduler][DynThreadMapping][Migration]: core utilizations:" << std::endl;
    for (size_t i = 0; i < coreIds.size(); ++i) {
        int c = coreIds[i];
        float u = performanceCounters->getUtilizationOfCore(c);
        std::cout << "  core " << std::setw(3) << c << ": "
             << std::fixed << std::setprecision(2) << u << std::endl;
    }
}

std::vector<migration> DynThreadMapping::migrate(SubsecondTime time, const std::vector<int> &taskIds, const std::vector<bool> &activeCores){
    std::vector<migration> migrations;

    setCurrentTaskIds(taskIds);
    hasPendingCombinedFrequencies = false;
    pendingCombinedFrequencies.clear();
    clearLastTemperaturePrediction();
    
    // Nothing running
    if (!std::count(activeCores.begin(), activeCores.end(), true)) {
        return migrations;
    }
    std::cout << "[Scheduler][DynThreadMapping][Migration]: Active Cores: ";
    for(bool active:activeCores){
        std::cout<<active<<", ";
    }
    std::cout<<std::endl;

    const int numCores = coreRows * coreColumns;
    const int activeCoreCount = std::count(activeCores.begin(), activeCores.end(), true);

    if (activeCoreCount == 1) {
        migrationOccured = false;

        int sourceCore = -1;
        for (int core = 0; core < numCores && core < static_cast<int>(activeCores.size()); ++core) {
            if (activeCores[core]) {
                sourceCore = core;
                break;
            }
        }
        if (sourceCore < 0) {
            return migrations;
        }

        const unsigned long long nowNs = static_cast<unsigned long long>(time.getNS());
        if (hasLastMasterMigration &&
            nowNs >= lastMasterMigrationNs &&
            nowNs - lastMasterMigrationNs < masterMigrationCooldownNs) {
            std::cout << "[Scheduler][DynThreadMapping][MasterMigration]: time_ns=" << nowNs
                      << " action=skip_cooldown source_core=" << sourceCore
                      << " cooldown_remaining_ns="
                      << (masterMigrationCooldownNs - (nowNs - lastMasterMigrationNs))
                      << std::endl;
            return migrations;
        }

        const double sourceTemp = readCounterValue(
            [&] { return performanceCounters->getTemperatureOfCore(sourceCore); },
            0.0);

        int targetCore = -1;
        double targetTemp = std::numeric_limits<double>::infinity();
        for (int core = 0; core < numCores && core < static_cast<int>(activeCores.size()); ++core) {
            if (core == sourceCore || activeCores[core]) {
                continue;
            }
            if (core >= static_cast<int>(taskIds.size()) || taskIds[core] == -1) {
                continue;
            }

            const double temp = readCounterValue(
                [&] { return performanceCounters->getTemperatureOfCore(core); },
                std::numeric_limits<double>::infinity());
            if (temp < targetTemp) {
                targetTemp = temp;
                targetCore = core;
            }
        }

        const double deltaTemp = sourceTemp - targetTemp;
        if (targetCore < 0 || !std::isfinite(targetTemp)) {
            std::cout << "[Scheduler][DynThreadMapping][MasterMigration]: time_ns=" << nowNs
                      << " action=skip_no_reserved_target source_core=" << sourceCore
                      << " source_task=" << (sourceCore < static_cast<int>(taskIds.size()) ? taskIds[sourceCore] : -1)
                      << " source_temp=" << sourceTemp << "C" << std::endl;
            return migrations;
        }

        if (std::isfinite(sourceTemp) &&
            std::isfinite(targetTemp) &&
            deltaTemp >= masterMigrationTemperatureDeltaThreshold) {
            migrations.emplace_back(migration{
                static_cast<unsigned int>(sourceCore),
                static_cast<unsigned int>(targetCore),
                true});
            hasLastMasterMigration = true;
            lastMasterMigrationNs = nowNs;
            std::cout << "[Scheduler][DynThreadMapping][MasterMigration]: time_ns=" << nowNs
                      << " action=move source_core=" << sourceCore
                      << " target_core=" << targetCore
                      << " source_task=" << (sourceCore < static_cast<int>(taskIds.size()) ? taskIds[sourceCore] : -1)
                      << " target_task=" << (targetCore < static_cast<int>(taskIds.size()) ? taskIds[targetCore] : -1)
                      << " source_temp=" << sourceTemp << "C"
                      << " target_temp=" << targetTemp << "C"
                      << " delta=" << deltaTemp << "C"
                      << " threshold=" << masterMigrationTemperatureDeltaThreshold << "C"
                      << " cooldown_ns=" << masterMigrationCooldownNs << std::endl;
        } else {
            std::cout << "[Scheduler][DynThreadMapping][MasterMigration]: time_ns=" << nowNs
                      << " action=skip_delta source_core=" << sourceCore
                      << " target_core=" << targetCore
                      << " source_task=" << (sourceCore < static_cast<int>(taskIds.size()) ? taskIds[sourceCore] : -1)
                      << " target_task=" << (targetCore < static_cast<int>(taskIds.size()) ? taskIds[targetCore] : -1)
                      << " source_temp=" << sourceTemp << "C"
                      << " target_temp=" << targetTemp << "C"
                      << " delta=" << deltaTemp << "C"
                      << " threshold=" << masterMigrationTemperatureDeltaThreshold << "C" << std::endl;
        }
        return migrations;
    }

    if (activeCoreCount < 2) {
        migrationOccured = false;
        return migrations;
    }

    auto buildMigrationsForPermutation = [&](const std::vector<unsigned int> &perm) {
        std::vector<migration> result;
        std::vector<unsigned int> currentAtCore(numCores);
        std::vector<unsigned int> positionOfOld(numCores);

        std::iota(currentAtCore.begin(), currentAtCore.end(), 0);
        std::iota(positionOfOld.begin(), positionOfOld.end(), 0);

        for (int targetCore = 0; targetCore < numCores; ++targetCore) {
            unsigned int desiredOldCore = perm[targetCore];
            if (currentAtCore[targetCore] == desiredOldCore) {
                continue;
            }

            unsigned int sourceCore = positionOfOld[desiredOldCore];
            result.emplace_back(migration{static_cast<unsigned int>(targetCore), sourceCore, true});

            unsigned int oldAtTarget = currentAtCore[targetCore];
            std::swap(currentAtCore[targetCore], currentAtCore[sourceCore]);
            positionOfOld[desiredOldCore] = targetCore;
            positionOfOld[oldAtTarget] = sourceCore;
        }

        return result;
    };

    // Convert a candidate placement of active workloads onto physical cores into
    // the migration operations understood by SchedulerOpen.
    auto buildMigrationsForWorkloadPlacement = [&](const std::vector<int> &workloadSourceAtCore) {
        std::vector<migration> result;
        std::vector<int> occupantAtCore(numCores, -1);
        std::vector<int> positionOfSource(numCores, -1);
        std::vector<bool> assignedAtCore(numCores, false);

        for (int core = 0; core < numCores; ++core) {
            assignedAtCore[core] =
                core < static_cast<int>(taskIds.size()) && taskIds[core] != -1;
            if (core < static_cast<int>(activeCores.size()) && activeCores[core]) {
                occupantAtCore[core] = core;
                positionOfSource[core] = core;
                assignedAtCore[core] = true;
            }
        }

        auto placementDone = [&]() {
            for (int core = 0; core < numCores; ++core) {
                if (occupantAtCore[core] != workloadSourceAtCore[core]) {
                    return false;
                }
            }
            return true;
        };

        auto appendMove = [&](int fromCore, int toCore) {
            if (fromCore < 0 || toCore < 0 || fromCore == toCore) {
                return;
            }

            const bool swap = occupantAtCore[toCore] != -1 || assignedAtCore[toCore];
            result.emplace_back(migration{
                static_cast<unsigned int>(fromCore),
                static_cast<unsigned int>(toCore),
                swap});

            if (swap) {
                std::swap(occupantAtCore[fromCore], occupantAtCore[toCore]);
                if (occupantAtCore[fromCore] >= 0) {
                    positionOfSource[occupantAtCore[fromCore]] = fromCore;
                }
                if (occupantAtCore[toCore] >= 0) {
                    positionOfSource[occupantAtCore[toCore]] = toCore;
                }
                std::swap(assignedAtCore[fromCore], assignedAtCore[toCore]);
            } else {
                const int movedSource = occupantAtCore[fromCore];
                occupantAtCore[toCore] = movedSource;
                if (movedSource >= 0) {
                    positionOfSource[movedSource] = toCore;
                }
                occupantAtCore[fromCore] = -1;
                assignedAtCore[toCore] = assignedAtCore[fromCore];
                assignedAtCore[fromCore] = false;
            }
        };

        for (int guard = 0; !placementDone() && guard < numCores * numCores + numCores; ++guard) {
            bool progress = false;

            for (int targetCore = 0; targetCore < numCores; ++targetCore) {
                const int sourceCore = workloadSourceAtCore[targetCore];
                if (sourceCore < 0) {
                    continue;
                }

                const int currentCore = positionOfSource[sourceCore];
                if (currentCore != targetCore && occupantAtCore[targetCore] == -1) {
                    appendMove(currentCore, targetCore);
                    progress = true;
                    break;
                }
            }

            if (progress) {
                continue;
            }

            for (int targetCore = 0; targetCore < numCores; ++targetCore) {
                const int sourceCore = workloadSourceAtCore[targetCore];
                if (sourceCore < 0) {
                    continue;
                }

                const int currentCore = positionOfSource[sourceCore];
                if (currentCore != targetCore) {
                    appendMove(currentCore, targetCore);
                    progress = true;
                    break;
                }
            }

            if (!progress) {
                break;
            }
        }

        return result;
    };

    if (sampleExplorationEnabled && numCores > 0 && !core_states.empty()) {
        const double peakTemp = performanceCounters->getPeakTemperature();
        const size_t maxStateIdx = core_states.size() - 1;

        auto firstStateAtLeast = [&](double minFreqGhz) {
            for (size_t idx = 0; idx < core_states.size(); ++idx) {
                if (core_states[idx] >= minFreqGhz) {
                    return idx;
                }
            }
            return maxStateIdx;
        };

        auto lastStateAtMost = [&](double maxFreqGhz) {
            size_t selected = 0;
            for (size_t idx = 0; idx < core_states.size(); ++idx) {
                if (core_states[idx] <= maxFreqGhz) {
                    selected = idx;
                }
            }
            return selected;
        };

        size_t minChoiceIdx = 0;
        size_t maxChoiceIdx = maxStateIdx;

        if (peakTemp >= sampleTargetMaxTemperature) {
            minChoiceIdx = 0;
            maxChoiceIdx = 0;
        } else if (peakTemp >= sampleTargetMaxTemperature - 5.0) {
            minChoiceIdx = 0;
            maxChoiceIdx = lastStateAtMost(2.0);
        } else if (peakTemp < sampleTargetMinTemperature) {
            minChoiceIdx = firstStateAtLeast(2.0);
            maxChoiceIdx = lastStateAtMost(3.0);
        } else {
            minChoiceIdx = 0;
            maxChoiceIdx = lastStateAtMost(2.5);
        }
        if (minChoiceIdx > maxChoiceIdx) {
            minChoiceIdx = maxChoiceIdx;
        }

        std::uniform_int_distribution<size_t> stateDist(minChoiceIdx, maxChoiceIdx);
        pendingCombinedFrequencies.assign(numCores, static_cast<int>(core_states.front() * 1000.0f));
        for (int core = 0; core < numCores; ++core) {
            if (core < static_cast<int>(activeCores.size()) && activeCores[core]) {
                pendingCombinedFrequencies[core] = static_cast<int>(core_states[stateDist(sampleRandomGenerator)] * 1000.0f);
            }
        }

        std::vector<unsigned int> permutation(numCores);
        std::iota(permutation.begin(), permutation.end(), 0);

        std::vector<unsigned int> activeIndexes;
        for (int core = 0; core < numCores; ++core) {
            if (core < static_cast<int>(activeCores.size()) && activeCores[core]) {
                activeIndexes.push_back(static_cast<unsigned int>(core));
            }
        }

        std::uniform_real_distribution<double> probabilityDist(0.0, 1.0);
        if (activeIndexes.size() > 1 && probabilityDist(sampleRandomGenerator) < sampleMigrationProbability) {
            std::vector<unsigned int> shuffled = activeIndexes;
            std::shuffle(shuffled.begin(), shuffled.end(), sampleRandomGenerator);
            for (size_t i = 0; i < activeIndexes.size(); ++i) {
                permutation[activeIndexes[i]] = shuffled[i];
            }
        }

        migrations = buildMigrationsForPermutation(permutation);
        migrationOccured = !migrations.empty();
        hasPendingCombinedFrequencies = true;

        std::cout << "[Scheduler][DynThreadMapping][SampleExplore]: peak_temp=" << peakTemp
                  << "C target=[" << sampleTargetMinTemperature << ", "
                  << sampleTargetMaxTemperature << "] migrations=" << migrations.size()
                  << " choice_freq_range=[" << core_states[minChoiceIdx] << ", "
                  << core_states[maxChoiceIdx] << "]GHz"
                  << " freqs=[";
        for (int freq : pendingCombinedFrequencies) {
            std::cout << freq << " ";
        }
        std::cout << "]" << std::endl;

        return migrations;
    }

    if (hasLoadedThermalModel() && numCores > 0 && numCores <= 6 && !core_states.empty()) {
        const double effectivePredictionBar = effectivePredictionTemperatureBar();
        std::vector<double> currentTemps(numCores, 0.0);
        std::vector<double> currentFreqs(numCores, 0.0);
        std::vector<double> measuredIps(numCores, 0.0);
        std::vector<double> utilizations(numCores, 0.0);
        std::vector<double> cpis(numCores, 0.0);
        std::vector<double> relNucaCpis(numCores, 0.0);
        std::vector<double> powers(numCores, 0.0);
        std::vector<NeighborPrediction::PredictionMap> workloadPredictions(numCores);

        for (int core = 0; core < numCores; ++core) {
            currentTemps[core] = readCounterValue([&] { return performanceCounters->getTemperatureOfCore(core); }, 0.0);
            powers[core] = readCounterValue([&] { return performanceCounters->getPowerOfCore(core); }, 0.0);
            utilizations[core] = readCounterValue([&] { return performanceCounters->getUtilizationOfCore(core); }, 0.0);
            cpis[core] = readCounterValue([&] { return performanceCounters->getCPIOfCore(core); }, 0.0);
            relNucaCpis[core] = readCounterValue([&] { return performanceCounters->getRelNUCACPIOfCore(core); }, 0.0);
            measuredIps[core] = readCounterValue([&] { return performanceCounters->getIPSOfCore(core); }, 0.0);

            double currentFreqMhz = readCounterValue([&] { return performanceCounters->getFreqOfCore(core); }, 0.0);
            currentFreqs[core] = currentFreqMhz > 0.0 ? currentFreqMhz / 1000.0 : core_states.back();

            auto fallbackPrediction = [&]() {
                NeighborPrediction::PredictionMap pm;
                for (float state : core_states) {
                    pm.emplace(state, NeighborPrediction::core_status{
                        "none",
                        state,
                        0.0f,
                        static_cast<float>(cpis[core]),
                        static_cast<float>(currentTemps[core]),
                        static_cast<float>(powers[core])});
                }
                return pm;
            };

            double currentIpsGips = measuredIps[core] / 1e9;
            if (currentIpsGips <= 0.1) {
                workloadPredictions[core] = fallbackPrediction();
            } else {
                try {
                    workloadPredictions[core] = pred.getNearestBenchmark(
                        nearestCoreState(core_states, currentFreqs[core]),
                        static_cast<float>(currentIpsGips));
                } catch (...) {
                    std::cerr << "[Scheduler][DynThreadMapping][Combined]: Warning failed NeighborPrediction for core "
                              << core << "; using zero-IPS fallback" << std::endl;
                    workloadPredictions[core] = fallbackPrediction();
                }
            }
        }

        int measuredBusyCores = 0;
        for (int core = 0; core < numCores; ++core) {
            if (utilizations[core] > 0.30 || measuredIps[core] > 0.1e9) {
                ++measuredBusyCores;
            }
        }

        size_t maxCandidateStateIdx = core_states.size() - 1;
        if (measuredBusyCores < activeCoreCount) {
            for (size_t idx = 0; idx < core_states.size(); ++idx) {
                if (core_states[idx] <= 2.5f) {
                    maxCandidateStateIdx = idx;
                }
            }
            std::cout << "[Scheduler][DynThreadMapping][Combined]: low-util startup guard busy_cores="
                      << measuredBusyCores << "/" << activeCoreCount
                      << " max_candidate_freq=" << core_states[maxCandidateStateIdx]
                      << "GHz" << std::endl;
        }

        struct CombinedCandidate {
            bool initialized;
            bool safe;
            double totalIps;
            double maxPredTemp;
            int migrationCount;
            std::vector<int> workloadSourceAtCore;
            std::vector<int> frequenciesMhz;
            std::vector<double> predictedTemps;

            CombinedCandidate()
                : initialized(false)
                , safe(false)
                , totalIps(0.0)
                , maxPredTemp(std::numeric_limits<double>::infinity())
                , migrationCount(0)
            {
            }
        };

        auto isBetterCandidate = [](bool safe,
                                    double totalIps,
                                    double maxPredTemp,
                                    int migrationCount,
                                    const CombinedCandidate &best) {
            const double eps = 1e-9;
            if (!best.initialized) {
                return true;
            }
            if (safe != best.safe) {
                return safe;
            }
            if (safe) {
                if (totalIps > best.totalIps + eps) {
                    return true;
                }
                if (std::fabs(totalIps - best.totalIps) <= eps) {
                    if (migrationCount < best.migrationCount) {
                        return true;
                    }
                    if (migrationCount == best.migrationCount && maxPredTemp < best.maxPredTemp - eps) {
                        return true;
                    }
                }
                return false;
            }

            if (maxPredTemp < best.maxPredTemp - eps) {
                return true;
            }
            if (std::fabs(maxPredTemp - best.maxPredTemp) <= eps) {
                if (totalIps > best.totalIps + eps) {
                    return true;
                }
                if (std::fabs(totalIps - best.totalIps) <= eps && migrationCount < best.migrationCount) {
                    return true;
                }
            }
            return false;
        };

        CombinedCandidate best;
        size_t mappingsEvaluated = 0;
        size_t candidatesEvaluated = 0;
        size_t safeCandidates = 0;

        std::vector<unsigned int> activeSourceCores;
        for (int core = 0; core < numCores; ++core) {
            if (core < static_cast<int>(activeCores.size()) && activeCores[core]) {
                activeSourceCores.push_back(static_cast<unsigned int>(core));
            }
        }

        std::vector<int> targetForSource(numCores, -1);
        std::vector<bool> targetUsed(numCores, false);

        std::function<void(size_t)> evaluatePlacement = [&](size_t sourceIndex) {
            if (sourceIndex < activeSourceCores.size()) {
                const unsigned int sourceCore = activeSourceCores[sourceIndex];
                for (int targetCore = 0; targetCore < numCores; ++targetCore) {
                    if (targetUsed[targetCore]) {
                        continue;
                    }

                    targetUsed[targetCore] = true;
                    targetForSource[sourceCore] = targetCore;
                    evaluatePlacement(sourceIndex + 1);
                    targetForSource[sourceCore] = -1;
                    targetUsed[targetCore] = false;
                }
                return;
            }

            ++mappingsEvaluated;
            std::vector<int> workloadSourceAtCore(numCores, -1);
            for (unsigned int sourceCore : activeSourceCores) {
                const int targetCore = targetForSource[sourceCore];
                if (targetCore >= 0 && targetCore < numCores) {
                    workloadSourceAtCore[targetCore] = static_cast<int>(sourceCore);
                }
            }

            const int migrationCount =
                static_cast<int>(buildMigrationsForWorkloadPlacement(workloadSourceAtCore).size());
            std::vector<size_t> frequencyIndices(activeSourceCores.size(), 0);

            while (true) {
                bool candidateAllowed = true;
                for (size_t stateIndex : frequencyIndices) {
                    if (stateIndex > maxCandidateStateIdx) {
                        candidateAllowed = false;
                        break;
                    }
                }
                if (!candidateAllowed) {
                    if (!incrementFrequencyIndices(frequencyIndices, core_states.size())) {
                        break;
                    }
                    continue;
                }

                ++candidatesEvaluated;

                std::vector<double> candidateFreqs(
                    numCores, static_cast<double>(core_states.front()));
                std::vector<int> candidateFrequenciesMhz(
                    numCores, static_cast<int>(core_states.front() * 1000.0f));
                std::vector<double> candidateIps(numCores, 0.0);
                std::vector<double> candidateUtilizations(numCores, 0.0);
                std::vector<double> candidateCpis(numCores, 0.0);
                std::vector<double> candidateRelNucaCpis(numCores, 0.0);
                std::vector<double> candidatePowers(numCores, 0.0);
                std::vector<double> thermalExpectedIps(numCores, 0.0);
                std::vector<double> thermalExpectedCpis(numCores, 0.0);
                std::vector<double> thermalExpectedPowers(numCores, 0.0);
                std::vector<double> thermalExpectedUtilizations(numCores, 0.0);
                std::vector<bool> candidateActiveCores(numCores, false);

                for (int core = 0; core < numCores; ++core) {
                    const double idleFreq = candidateFreqs[core];
                    candidateCpis[core] = cpis[core];
                    candidateRelNucaCpis[core] = relNucaCpis[core];
                    candidatePowers[core] = scaledForCandidateFrequency(
                        powers[core],
                        currentFreqs[core],
                        idleFreq);
                    thermalExpectedCpis[core] = cpis[core];
                    thermalExpectedPowers[core] = candidatePowers[core];
                }

                double totalIps = 0.0;
                for (size_t sourceIndex = 0; sourceIndex < activeSourceCores.size(); ++sourceIndex) {
                    const unsigned int oldCore = activeSourceCores[sourceIndex];
                    const int newCore = targetForSource[oldCore];
                    if (newCore < 0 || newCore >= numCores) {
                        continue;
                    }

                    float state = core_states[frequencyIndices[sourceIndex]];
                    candidateFreqs[newCore] = state;
                    candidateFrequenciesMhz[newCore] = static_cast<int>(state * 1000.0f);
                    candidateActiveCores[newCore] = true;

                    const auto status = workloadPredictions[oldCore].find(state);
                    if (status != workloadPredictions[oldCore].end()) {
                        totalIps += status->second.ips;
                        candidateIps[newCore] = status->second.ips * 1e9;
                        candidateCpis[newCore] = status->second.cpi;
                        candidatePowers[newCore] = status->second.power;
                    } else {
                        candidateIps[newCore] = scaledForCandidateFrequency(
                            measuredIps[oldCore],
                            currentFreqs[oldCore],
                            state);
                        totalIps += candidateIps[newCore] / 1e9;
                        candidateCpis[newCore] = cpis[oldCore];
                        candidatePowers[newCore] = scaledForCandidateFrequency(
                            powers[oldCore],
                            currentFreqs[oldCore],
                            state);
                    }

                    candidateUtilizations[newCore] = utilizations[oldCore];
                    candidateRelNucaCpis[newCore] = relNucaCpis[oldCore];
                    thermalExpectedIps[newCore] = scaledForCandidateFrequency(
                        measuredIps[oldCore],
                        currentFreqs[oldCore],
                        state);
                    thermalExpectedCpis[newCore] = cpis[oldCore];
                    thermalExpectedPowers[newCore] = scaledForCandidateFrequency(
                        powers[oldCore],
                        currentFreqs[oldCore],
                        state);
                    thermalExpectedUtilizations[newCore] = utilizations[oldCore];
                }

                bool safe = true;
                double maxPredTemp = -std::numeric_limits<double>::infinity();
                std::vector<double> predictedTemps(numCores, 0.0);

                for (int core = 0; core < numCores; ++core) {
                    int thermalTaskId = core < static_cast<int>(taskIds.size()) ? taskIds[core] : -1;
                    if (core < static_cast<int>(workloadSourceAtCore.size()) &&
                        workloadSourceAtCore[core] >= 0 &&
                        workloadSourceAtCore[core] < static_cast<int>(taskIds.size())) {
                        thermalTaskId = taskIds[workloadSourceAtCore[core]];
                    }
                    const MLTemperaturePredictor &thermalPredictor = getThermalModelForTaskId(thermalTaskId);
                    double predTemp = thermalPredictor.predictNextTemp(
                        core,
                        currentTemps,
                        currentFreqs,
                        candidateFreqs,
                        measuredIps,
                        utilizations,
                        cpis,
                        candidateRelNucaCpis,
                        powers,
                        activeCores,
                        false,
                        &thermalExpectedIps,
                        &thermalExpectedCpis,
                        &thermalExpectedPowers,
                        &thermalExpectedUtilizations,
                        &candidateActiveCores);

                    if (!std::isfinite(predTemp)) {
                        predTemp = std::numeric_limits<double>::infinity();
                    }
                    predictedTemps[core] = predTemp;
                    maxPredTemp = std::max(maxPredTemp, predTemp);
                    if (predTemp >= effectivePredictionBar) {
                        safe = false;
                    }
                }

                if (safe) {
                    ++safeCandidates;
                }

                if (isBetterCandidate(safe, totalIps, maxPredTemp, migrationCount, best)) {
                    best.initialized = true;
                    best.safe = safe;
                    best.totalIps = totalIps;
                    best.maxPredTemp = maxPredTemp;
                    best.migrationCount = migrationCount;
                    best.workloadSourceAtCore = workloadSourceAtCore;
                    best.frequenciesMhz = candidateFrequenciesMhz;
                    best.predictedTemps = predictedTemps;
                }

                if (!incrementFrequencyIndices(frequencyIndices, core_states.size())) {
                    break;
                }
            }
        };

        evaluatePlacement(0);

        if (best.initialized) {
            pendingCombinedFrequencies = best.frequenciesMhz;
            hasPendingCombinedFrequencies = true;
            migrations = buildMigrationsForWorkloadPlacement(best.workloadSourceAtCore);
            migrationOccured = !migrations.empty();
            setLastTemperaturePrediction(best.predictedTemps, best.frequenciesMhz);

            std::cout << "[Scheduler][DynThreadMapping][Combined]: evaluated "
                      << mappingsEvaluated << " mappings and " << candidatesEvaluated
                      << " combined candidates; safe candidates=" << safeCandidates
                      << " active_workloads=" << activeSourceCores.size() << "/" << numCores
                      << " effective_temp_bar=" << effectivePredictionBar << "C" << std::endl;
            std::cout << "[Scheduler][DynThreadMapping][Combined]: selected "
                      << (best.safe ? "safe" : "fallback") << " candidate total_ips="
                      << best.totalIps << " max_pred_temp=" << best.maxPredTemp
                      << "C migrations=" << migrations.size() << " freqs=[";
            for (int freq : pendingCombinedFrequencies) {
                std::cout << freq << " ";
            }
            std::cout << "] pred_temps=[";
            for (double temp : best.predictedTemps) {
                std::cout << temp << " ";
            }
            std::cout << "]" << std::endl;

            return migrations;
        }
    }

    std::vector<double> utilizations;
    utilizations.reserve(numCores);
    
    for (int c = 0; c < numCores; ++c) {
        double u = performanceCounters->getUtilizationOfCore(c);
        utilizations.push_back(u);
    }


    std::vector<unsigned int> high_low(numCores);
    std::vector<unsigned int> low_high(numCores);

    std::iota(high_low.begin(), high_low.end(), 0);
    std::iota(low_high.begin(), low_high.end(), 0);


    std::sort(high_low.begin(), high_low.end(),
              [&](unsigned int a, unsigned int b) {
                  return utilizations[a] > utilizations[b];
              });
    std::sort(low_high.begin(), low_high.end(),
            [&](unsigned int a, unsigned int b) {
                return utilizations[a] < utilizations[b];
            });

    for(int i=0;i<numCores/2;i++){
        unsigned int highCore = high_low[i];
        unsigned int lowCore = low_high[i];
        double utilizationDelta = utilizations[highCore] - utilizations[lowCore];
        bool isHighCoreActive = activeCores[highCore];
        bool isLowCoreActive = activeCores[lowCore];

        if(highCore != lowCore && isHighCoreActive && utilizationDelta >= migrationUtilizationDeltaThreshold){
            bool swap = isLowCoreActive | (taskIds.at(lowCore) != -1);
            if(swap)
                std::cout << "[Scheduler][DynThreadMapping][Migration]: swapping threads on cores: ("<<highCore<<"-"<<lowCore
                          << "), utilization delta=" << utilizationDelta << std::endl;
            else
                std::cout << "[Scheduler][DynThreadMapping][Migration]: moving thread on core "<<highCore<<" to core "<<lowCore
                          << ", utilization delta=" << utilizationDelta << std::endl;

            migrations.emplace_back(migration{highCore,lowCore,swap});
        }
    }
    migrationOccured = !migrations.empty();
    return migrations;
}
