#include "migrationSota.h"
#include <iomanip>
#include <unordered_map>
using namespace std;

migrationSota::migrationSota(
    const PerformanceCounters *performanceCounters,
    int coreRows,
    int coreColumns,
    float criticalTemperature)
    : performanceCounters(performanceCounters),
      coreRows(coreRows),
      coreColumns(coreColumns),
      criticalTemperature(criticalTemperature) {
}

std::vector<int> migrationSota::map( //Initial placement decision
    String taskName,
    int taskCoreRequirement,
    const std::vector<bool> &availableCoresRO,
    const std::vector<bool> &activeCores) {

    std::vector<bool> availableCores(availableCoresRO);
    std::vector<int> cores;

    logTemperatures(availableCores);


    for (; taskCoreRequirement > 0; taskCoreRequirement--) {
        int coldestCore = getColdestCore(availableCores);

        if (coldestCore == -1) {
            // not enough free cores
            std::vector<int> empty;
            return empty;
        } else {
            cores.push_back(coldestCore);
            availableCores.at(coldestCore) = false;
        }
    }

    return cores;
}

std::vector<migration> migrationSota::migrate(
    SubsecondTime time,
    const std::vector<int> &taskIds,
    const std::vector<bool> &activeCores) {

    std::vector<migration> migrations;
    std::vector<bool> availableCores(coreRows * coreColumns);
    int tot_threads = 0;
    std::unordered_map<int, int> threads_core;
    std::unordered_map<int, int> newthreads_core;
    int temp;

    for (int c = 0; c < coreRows * coreColumns; c++) {
        availableCores.at(c) = taskIds.at(c) == -1;
        threads_core[taskIds.at(c)] = c;
        newthreads_core[taskIds.at(c)] = -1;
        if(taskIds.at(c) != -1) {
            tot_threads++;
        }
    }
    // int *threads_core = new int[tot_threads];
    // for (int i = 0; i < tot_threads; i++) {
    //     threads_core[i] = -1;
    // }

    int thread_idx;
    for (int c = 0; c < coreRows * coreColumns; c++) {
        std::unordered_map<int, int> threads_power;
        for (auto& pair : newthreads_core){
            if(newthreads_core[pair.first] == -1) {
                threads_power[pair.first] = (rand() % 100 + 1);//performanceCounters->getPowerOfCore(coreId);
            } else {
                threads_power[pair.first]  = -1;
            }
           
        }

        int maxVal=-1;

        for (auto& pair : threads_power){
            if(threads_power[pair.first] > maxVal) {
                maxVal = threads_power[pair.first];
                thread_idx = pair.first;
            }
        }

        if(maxVal != -1) {
            newthreads_core[thread_idx] = c;
        }
    }

    for (auto& pair : newthreads_core) {
       
        if(threads_core[pair.first] != newthreads_core[pair.first]) {
            migration m;
            m.fromCore = threads_core[pair.first];
            m.toCore = newthreads_core[pair.first];
            if(availableCores.at(newthreads_core[pair.first])) {
                m.swap = false;
                availableCores.at(newthreads_core[pair.first]) = false;
                availableCores.at(threads_core[pair.first]) = true;
                threads_core[pair.first] = newthreads_core[pair.first];

            } else {
                m.swap = true;
                temp = threads_core[pair.first];
                
                for (auto& pair2 : threads_core) {
                    if(pair2.second == newthreads_core[pair.first]) {
                        threads_core[pair2.first] = temp;//new core's thread becomes old core's thread
                        break;
                    }
                }
                threads_core[pair.first] = newthreads_core[pair.first];
            }

            migrations.push_back(m);
        }
        
    }

    return migrations;
}

int migrationSota::getColdestCore(const std::vector<bool> &availableCores) {
    int coldestCore = -1;
    float coldestTemperature = 0;

    // iterate all cores to find coldest
    for (int c = 0; c < coreRows * coreColumns; c++) {
        if (availableCores.at(c)) {
            float temperature = performanceCounters->getTemperatureOfCore(c);

            if ((coldestCore == -1) || (temperature < coldestTemperature)) {
                coldestCore = c;
                coldestTemperature = temperature;
            }
        }
    }

    return coldestCore;
}

void migrationSota::logTemperatures(const std::vector<bool> &availableCores) {
    cout << "[Scheduler][coldestCore-map]: temperatures of available cores:" << endl;

    for (int y = 0; y < coreRows; y++) {
        for (int x = 0; x < coreColumns; x++) {
            if (x > 0) {
                cout << " ";
            }

            int coreId = y * coreColumns + x;

            if (!availableCores.at(coreId)) {
                cout << " - ";
            } else {
                float temperature = performanceCounters->getTemperatureOfCore(coreId);
                cout << fixed << setprecision(1) << temperature;
            }
        }
        cout << endl;
    }
}