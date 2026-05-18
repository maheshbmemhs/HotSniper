#include "greedyMaxIPS.h"
#include <iomanip>
#include <numeric>
#include <algorithm>


void GreedyMaxIPS::logTemps(const std::vector<int> &coreIds) {
    std::cout << "[Scheduler][GreedyMaxIPS][Migration]: core temperature:" << std::endl;
    for (size_t i = 0; i < coreIds.size(); ++i) {
        int c = coreIds[i];
        float u = performanceCounters->getTemperatureOfCore(c);
        std::cout << "  core " << std::setw(3) << c << ": "
             << std::fixed << std::setprecision(2) << u << std::endl;
    }
}

std::vector<migration> GreedyMaxIPS::migrate(SubsecondTime time, const std::vector<int> &taskIds, const std::vector<bool> &activeCores){
    std::vector<migration> migrations;
    
    // Nothing running
    if (!std::count(activeCores.begin(), activeCores.end(), true)) {
        return migrations;
    }
    std::cout << "[Scheduler][GreedyMaxIPS][Migration]: Active Cores: ";
    for(bool active:activeCores){
        std::cout<<active<<", ";
    }
    std::cout<<std::endl;

    const int numCores = coreRows * coreColumns;

    std::vector<double> temps;
    temps.reserve(numCores);
    
    for (int c = 0; c < numCores; ++c) {
        double u = performanceCounters->getTemperatureOfCore(c);
        temps.push_back(u);
    }


    std::vector<unsigned int> high_low(numCores);
    std::vector<unsigned int> low_high(numCores);

    std::iota(high_low.begin(), high_low.end(), 0);
    std::iota(low_high.begin(), low_high.end(), 0);


    std::sort(high_low.begin(), high_low.end(),
              [&](unsigned int a, unsigned int b) {
                  return temps[a] > temps[b];
              });
    std::sort(low_high.begin(), low_high.end(),
            [&](unsigned int a, unsigned int b) {
                return temps[a] < temps[b];
            });

    for(int i=0;i<numCores/2;i++){
        bool isHighCoreActive = activeCores[high_low[i]];
        bool isLowCoreActive = activeCores[low_high[i]];
        if(high_low[i]!=low_high[i] && isHighCoreActive){
            bool swap = isLowCoreActive | (taskIds.at(low_high[i]) != -1);
            if(swap)
                std::cout << "[Scheduler][GreedyMaxIPS][Migration]: swapping threads on cores: ("<<high_low[i]<<"-"<<low_high[i]<<")" << std::endl;
            else
                std::cout << "[Scheduler][GreedyMaxIPS][Migration]: moving thread on core "<<high_low[i]<<" to core "<<low_high[i]<< std::endl;

            migrations.emplace_back(migration{high_low[i],low_high[i],swap});
        }
    }
    migrationOccured = true;
    return migrations;
}