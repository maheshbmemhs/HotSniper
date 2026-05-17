#include "migration_imp.h"
#include "neighbor_prediction.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <cmath>
#include "simulator.h"
#include "magic_server.h"  

using namespace std;

MigrationImp::MigrationImp(
    const PerformanceCounters *performanceCounters,
    int coreRows,
    int coreColumns,
    float criticalTemperature)
    : performanceCounters(performanceCounters),
      coreRows(coreRows),
      coreColumns(coreColumns),
      criticalTemperature(criticalTemperature) {
    cout << "[MigrationImp] Initialized with critical temperature: " << criticalTemperature << "C" << endl;
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
        int freqMHz = Sim()->getMagicServer()->getFrequency(core); //performanceCounters->getFreqOfCore(core);
        if (freqMHz <= 0) {
            cout << "[MigrationImp] Frequencies not yet initialized (core " << core 
                << " has freq=" << freqMHz << " MHz). Skipping migration." << endl;
            return migrations;
        }
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
                  return a.ips > b.ips;
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

        // Consider all other cores as potential targets
        for (int targetCore = 0; targetCore < numCores; targetCore++) {
            if (targetCore == currentCore) continue; // Skip current core

            bool isSwap = !coreAvailable[targetCore];
            
            // Predict what happens if this thread moves to targetCore
            float targetFreqGHz = Sim()->getMagicServer()->getFrequency(targetCore) / 1000.0f;
            auto targetPred = pred.getNearestBenchmark(targetFreqGHz, threadIPS);
            float newThreadTemp = targetPred[targetFreqGHz].temp;
            float newThreadIPS = targetPred[targetFreqGHz].ips;

            float deltaTemp, deltaIPS;

            if (isSwap) {
                // Need to also consider the thread being swapped
                int swapThread = currentCoreToThread[targetCore];
                float swapThreadIPS = performanceCounters->getIPSOfCore(targetCore);
                
                // Predict swap thread on current core
                auto swapPred = pred.getNearestBenchmark(currentFreqGHz, swapThreadIPS);
                float swapThreadNewTemp = swapPred[currentFreqGHz].temp;
                float swapThreadNewIPS = swapPred[currentFreqGHz].ips;
                
                // Get current state of swap thread
                auto swapCurrentPred = pred.getNearestBenchmark(targetFreqGHz, swapThreadIPS);
                float swapThreadCurrentTemp = swapCurrentPred[targetFreqGHz].temp;
                float swapThreadCurrentIPS = swapCurrentPred[targetFreqGHz].ips;
                
                // Total delta for both threads
                deltaTemp = (currentThreadTemp + swapThreadCurrentTemp) - (newThreadTemp + swapThreadNewTemp);
                deltaIPS = (currentThreadIPS + swapThreadCurrentIPS) - (newThreadIPS + swapThreadNewIPS);
            } else {
                // Simple move to empty core
                deltaTemp = currentThreadTemp - newThreadTemp;
                deltaIPS = currentThreadIPS - newThreadIPS;
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
