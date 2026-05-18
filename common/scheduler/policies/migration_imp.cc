#include "migration_imp.h"
#include "neighbor_prediction.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <functional>
#include "simulator.h"
#include "magic_server.h"  

using namespace std;

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

}

MigrationImp::MigrationImp(
    const PerformanceCounters *performanceCounters,
    int coreRows,
    int coreColumns,
    float criticalTemperature,
    std::string thermalModelPath,
    bool thermalModelDebug)
    : performanceCounters(performanceCounters),
      coreRows(coreRows),
      coreColumns(coreColumns),
      criticalTemperature(criticalTemperature),
      thermal_model(thermalModelPath, thermalModelDebug),
      thermalModelDebug(thermalModelDebug) {
    cout << "[MigrationImp] Initialized with critical temperature: " << criticalTemperature << "C" << endl;
    if (thermal_model.isLoaded()) {
        cout << "[MigrationImp] ML thermal model loaded from: " << thermalModelPath << endl;
    } else {
        cout << "[MigrationImp] ML thermal model not loaded, using NN predictions only" << endl;
    }
}

std::vector<migration> MigrationImp::migrate(
    SubsecondTime time,
    const std::vector<int> &taskIds,
    const std::vector<bool> &activeCores) {

    std::vector<migration> migrations;
    int numCores = coreRows * coreColumns;

    // Build mapping: core -> thread and collect active cores
    std::unordered_map<int, int> coreToThread;
    std::vector<int> activeCoreList;
    
    for (int c = 0; c < numCores; c++) {
        if (taskIds[c] != -1) {
            coreToThread[c] = c; // In HotSniper, thread typically equals core when assigned
            activeCoreList.push_back(c);
        }
    }

    if (activeCoreList.empty()) {
        cout << "[MigrationImp] No active cores, skipping migration" << endl;
        return migrations;
    }

    // Check current thermal state
    float maxTemp = getMaxTemperature(activeCores);
    cout << "[MigrationImp] Current max temperature: " << maxTemp << "C, threshold: " << criticalTemperature << "C" << endl;

    if (maxTemp <= criticalTemperature) {
        cout << "[MigrationImp] Temperature within limit, no migration needed" << endl;
        return migrations;
    }

    // Check if frequencies are initialized (DVFS must run before migration)
    bool frequenciesValid = true;
    for (int core : activeCoreList) {
        int freqMHz = Sim()->getMagicServer()->getFrequency(core);
        if (freqMHz <= 0) {
            cout << "[MigrationImp] Frequencies not yet initialized (core " << core 
                << " has freq=" << freqMHz << " MHz). Skipping migration." << endl;
            return migrations;
        }
    }

    // Step 3: Gather current performance counter data for all cores (needed for ML predictions)
    const size_t numCoresSize = coreRows * coreColumns;
    std::vector<double> currentTemps(numCoresSize, 0.0);
    std::vector<double> currentFreqs(numCoresSize, 0.0);
    std::vector<double> ips(numCoresSize, 0.0);
    std::vector<double> utilizations(numCoresSize, 0.0);
    std::vector<double> cpis(numCoresSize, 0.0);
    std::vector<double> relNucaCpis(numCoresSize, 0.0);
    std::vector<double> powers(numCoresSize, 0.0);

    for (size_t core = 0; core < numCoresSize; core++) {
        currentTemps[core] = readCounterValue([&] { return performanceCounters->getTemperatureOfCore(core); }, 0.0);
        powers[core] = readCounterValue([&] { return performanceCounters->getPowerOfCore(core); }, 0.0);
        utilizations[core] = readCounterValue([&] { return performanceCounters->getUtilizationOfCore(core); }, 0.0);
        cpis[core] = readCounterValue([&] { return performanceCounters->getCPIOfCore(core); }, 0.0);
        relNucaCpis[core] = readCounterValue([&] { return performanceCounters->getRelNUCACPIOfCore(core); }, 0.0);
        ips[core] = readCounterValue([&] { return performanceCounters->getIPSOfCore(core); }, 0.0);

        double freqMhz = Sim()->getMagicServer()->getFrequency(core);
        currentFreqs[core] = freqMhz > 0.0 ? freqMhz / 1000.0 : 0.0;
    }

    // Initialize neighbor prediction
    NeighborPrediction pred("profile.txt");

    // Phase 1: Get current state and sort threads by IPS (highest first)
    struct ThreadInfo {
        int core;
        int thread;
        float temperature;
        float ips;
    };

    std::vector<ThreadInfo> threads;
    for (int core : activeCoreList) {
        ThreadInfo ti;
        ti.core = core;
        ti.thread = coreToThread[core];
        ti.temperature = performanceCounters->getTemperatureOfCore(core);
        ti.ips = performanceCounters->getIPSOfCore(core);
        threads.push_back(ti);
        
        cout << "[MigrationImp] Core " << core 
             << ": Temp=" << ti.temperature << "C, IPS=" << ti.ips << endl;
    }

    // Sort threads by IPS (descending - highest IPS first)
    std::sort(threads.begin(), threads.end(),
              [](const ThreadInfo &a, const ThreadInfo &b) {
                //   return a.ips > b.ips;
                return a.temperature > b.temperature;
              });

    cout << "[MigrationImp] Processing order (by IPS):" << endl;
    for (size_t i = 0; i < threads.size(); i++) {
        cout << "  " << i << ": Core " << threads[i].core << ", IPS=" << threads[i].ips << endl;
    }

    // Track current assignment state
    std::unordered_map<int, int> currentCoreToThread = coreToThread;
    std::vector<bool> coreAvailable(numCores, true);
    for (int core : activeCoreList) {
        coreAvailable[core] = false; // Currently occupied
    }

    // Phase 2: Greedy assignment - process each thread by IPS order
    for (const auto& threadInfo : threads) {
        int currentCore = threadInfo.core;
        int thread = threadInfo.thread;
        float threadIPS = threadInfo.ips;

        // Step 3: Check if we've met thermal constraint after each assignment
        maxTemp = getMaxTemperature(activeCores);
        if (maxTemp <= criticalTemperature) {
            cout << "[MigrationImp] Thermal constraint satisfied (maxTemp=" << maxTemp 
                 << " <= threshold=" << criticalTemperature << "), stopping" << endl;
            break;
        }

        // Step 1: Find best core for this high-IPS thread based on ΔT/ΔIPS
        struct CoreOption {
            int targetCore;
            bool isSwap;
            float deltaTemp;
            float deltaIPS;
            float ratio;
        };

        std::vector<CoreOption> options;

        // Get current prediction for this thread on its current core
        float currentFreqGHz = Sim()->getMagicServer()->getFrequency(currentCore) / 1000.0f;
        auto currentPred = pred.getNearestBenchmark(currentFreqGHz, threadIPS);
        float currentThreadTemp = currentPred[currentFreqGHz].temp;
        float currentThreadIPS = currentPred[currentFreqGHz].ips;

        // Enhance with ML prediction if available (keeps current state)
        if (thermal_model.isLoaded()) {
            std::vector<double> candidateFreqs = currentFreqs;  // Already at current freq
            std::vector<double> expectedIps = ips;
            std::vector<double> expectedCpis = cpis;
            std::vector<double> expectedPowers = powers;
            std::vector<double> expectedUtilizations = utilizations;

            const double ml_temp = thermal_model.predictNextTemp(
                currentCore,
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

            if (std::isfinite(ml_temp) && ml_temp > 0.0) {
                currentThreadTemp = ml_temp;
                static bool logged_ml_first = false;
                if (!logged_ml_first) {
                    cout << "[MigrationImp][ML] First ML prediction core=" << currentCore 
                         << " freq=" << currentFreqGHz << " temp=" << ml_temp << "C" << endl;
                    logged_ml_first = true;
                }
            }
        }

        // Consider all other cores as potential targets
        for (int targetCore = 0; targetCore < numCores; targetCore++) {
            if (targetCore == currentCore) continue; // Skip current core

            bool isSwap = !coreAvailable[targetCore];
            
            // Predict what happens if this thread moves to targetCore
            float targetFreqGHz = Sim()->getMagicServer()->getFrequency(targetCore) / 1000.0f;
            auto targetPred = pred.getNearestBenchmark(targetFreqGHz, threadIPS);
            float newThreadTemp = targetPred[targetFreqGHz].temp;
            float newThreadIPS = targetPred[targetFreqGHz].ips;

            // Enhance with ML prediction for thread on target core
            if (thermal_model.isLoaded()) {
                std::vector<double> candidateFreqs = currentFreqs;
                candidateFreqs[targetCore] = targetFreqGHz;  // This core would run the thread
                
                std::vector<double> expectedIps = ips;
                std::vector<double> expectedCpis = cpis;
                std::vector<double> expectedPowers = powers;
                std::vector<double> expectedUtilizations = utilizations;
                
                expectedIps[targetCore] = scaledForCandidateFrequency(ips[currentCore], currentFreqGHz, targetFreqGHz);
                expectedPowers[targetCore] = scaledForCandidateFrequency(powers[currentCore], currentFreqGHz, targetFreqGHz);

                const double ml_temp = thermal_model.predictNextTemp(
                    targetCore,
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

                if (std::isfinite(ml_temp) && ml_temp > 0.0) {
                    newThreadTemp = ml_temp;
                }
            }

            float deltaTemp, deltaIPS;

            if (isSwap) {
                // Need to also consider the thread being swapped
                int swapThread = currentCoreToThread[targetCore];
                float swapThreadIPS = performanceCounters->getIPSOfCore(targetCore);
                
                // Predict swap thread on current core
                auto swapPred = pred.getNearestBenchmark(currentFreqGHz, swapThreadIPS);
                float swapThreadNewTemp = swapPred[currentFreqGHz].temp;
                float swapThreadNewIPS = swapPred[currentFreqGHz].ips;
                
                // Enhance swap thread prediction with ML (swap thread moving to currentCore)
                if (thermal_model.isLoaded()) {
                    std::vector<double> candidateFreqs = currentFreqs;
                    candidateFreqs[currentCore] = currentFreqGHz;  // Swap thread would run here
                    
                    std::vector<double> expectedIps = ips;
                    std::vector<double> expectedCpis = cpis;
                    std::vector<double> expectedPowers = powers;
                    std::vector<double> expectedUtilizations = utilizations;
                    
                    expectedIps[currentCore] = scaledForCandidateFrequency(ips[targetCore], targetFreqGHz, currentFreqGHz);
                    expectedPowers[currentCore] = scaledForCandidateFrequency(powers[targetCore], targetFreqGHz, currentFreqGHz);

                    const double ml_temp = thermal_model.predictNextTemp(
                        currentCore,
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

                    if (std::isfinite(ml_temp) && ml_temp > 0.0) {
                        swapThreadNewTemp = ml_temp;
                    }
                }
                
                // Get current state of swap thread
                auto swapCurrentPred = pred.getNearestBenchmark(targetFreqGHz, swapThreadIPS);
                float swapThreadCurrentTemp = swapCurrentPred[targetFreqGHz].temp;
                float swapThreadCurrentIPS = swapCurrentPred[targetFreqGHz].ips;
                
                // Enhance current swap thread temp with ML (already on targetCore)
                if (thermal_model.isLoaded()) {
                    std::vector<double> candidateFreqs = currentFreqs;  // No change, already there
                    std::vector<double> expectedIps = ips;
                    std::vector<double> expectedCpis = cpis;
                    std::vector<double> expectedPowers = powers;
                    std::vector<double> expectedUtilizations = utilizations;

                    const double ml_temp = thermal_model.predictNextTemp(
                        targetCore,
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

                    if (std::isfinite(ml_temp) && ml_temp > 0.0) {
                        swapThreadCurrentTemp = ml_temp;
                    }
                }
                
                // Total delta for both threads
                deltaTemp = (currentThreadTemp + swapThreadCurrentTemp) - (newThreadTemp + swapThreadNewTemp);
                deltaIPS = (currentThreadIPS + swapThreadCurrentIPS) - (newThreadIPS + swapThreadNewIPS);
                
                // Add thermal headroom bonus: prefer cooler cores
                // Bonus for target core being cool, penalty for current core being cool (losing it)
                float targetCoolnessBonus = (newThreadTemp - currentTemps[targetCore]) * 1.0f;
                // float currentCorePenalty = (swapThreadNewTemp - currentTemps[currentCore]) * 0.3f;
                deltaTemp += targetCoolnessBonus ;//- currentCorePenalty;
            } else {
                // Simple move to empty core
                deltaTemp = currentThreadTemp - newThreadTemp;
                deltaIPS = currentThreadIPS - newThreadIPS;
                
                // Add thermal headroom bonus: prefer cores with thermal capacity
                float coolnessBonus = (newThreadTemp - currentTemps[targetCore]) * 0.3f;
                deltaTemp += coolnessBonus;
            }

            CoreOption opt;
            opt.targetCore = targetCore;
            opt.isSwap = isSwap;
            opt.deltaTemp = deltaTemp;
            opt.deltaIPS = deltaIPS;
            
            // Calculate ΔT/ΔIPS ratio
            if (deltaIPS > 0.001f) {
                opt.ratio = deltaTemp / deltaIPS;
            } else if (deltaTemp > 0.0f && deltaIPS <= 0.0f) {
                opt.ratio = 1000.0f; // Ideal: temp reduces, IPS doesn't decrease
            } else {
                opt.ratio = -1000.0f; // Bad move
            }
            
            // Only consider temperature-reducing moves
            if (deltaTemp > 0.0f) {
                options.push_back(opt);
            }
        }

        // Find best option (highest ΔT/ΔIPS ratio)
        CoreOption* bestOption = nullptr;
        float bestRatio = -10000.0f;
        
        for (auto& opt : options) {
            if (opt.ratio > bestRatio) {
                bestRatio = opt.ratio;
                bestOption = &opt;
            }
        }

        // Apply best move if found
        if (bestOption != nullptr && bestOption->ratio > 0.0f) {
            migration m;
            m.fromCore = currentCore;
            m.toCore = bestOption->targetCore;
            m.swap = bestOption->isSwap;
            migrations.push_back(m);

            cout << "[MigrationImp] Thread on core " << currentCore 
                 << " (IPS=" << threadIPS << ") -> core " << bestOption->targetCore
                 << " (swap=" << bestOption->isSwap << ")"
                 << ", ΔT=" << bestOption->deltaTemp
                 << ", ΔIPS=" << bestOption->deltaIPS
                 << ", ΔT/ΔIPS=" << bestOption->ratio << endl;

            // Update tracking
            if (bestOption->isSwap) {
                int swapThread = currentCoreToThread[bestOption->targetCore];
                currentCoreToThread[currentCore] = swapThread;
                currentCoreToThread[bestOption->targetCore] = thread;
            } else {
                currentCoreToThread.erase(currentCore);
                currentCoreToThread[bestOption->targetCore] = thread;
                coreAvailable[currentCore] = true;
                coreAvailable[bestOption->targetCore] = false;
            }

            // Limit migrations to avoid excessive overhead
            if (migrations.size() >= 10) {
                cout << "[MigrationImp] Migration limit reached (10), stopping" << endl;
                break;
            }
        } else {
            cout << "[MigrationImp] No beneficial move for thread on core " << currentCore << endl;
        }
    }

    cout << "[MigrationImp] Selected " << migrations.size() << " migration(s)" << endl;
    
    return migrations;
}

float MigrationImp::getMaxTemperature(const std::vector<bool> &activeCores) const {
    float maxTemp = -1000.0f;
    int numCores = coreRows * coreColumns;

    for (int c = 0; c < numCores; c++) {
        if (activeCores[c]) {
            float temp = performanceCounters->getTemperatureOfCore(c);
            if (temp > maxTemp) {
                maxTemp = temp;
            }
        }
    }

    return maxTemp;
}

float MigrationImp::getAverageIPS(const std::vector<int> &taskIds, const std::vector<bool> &activeCores) const {
    float totalIPS = 0.0f;
    int count = 0;
    int numCores = coreRows * coreColumns;

    for (int c = 0; c < numCores; c++) {
        if (taskIds[c] != -1 && activeCores[c]) {
            totalIPS += performanceCounters->getIPSOfCore(c);
            count++;
        }
    }

    return (count > 0) ? (totalIPS / count) : 0.0f;
}
