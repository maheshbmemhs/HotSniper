/**
 * This header implements a policy that maps new applications to the coldest core
 * and migrates threads from hot cores to the coldest cores.
 */
#ifndef __MIGRATIONSOTA_H
#define __MIGRATIONSOTA_H

#include <vector>
#include "mappingpolicy.h"
#include "migrationpolicy.h"
#include "performance_counters.h"
#include <unordered_map>

class migrationSota : public MappingPolicy, public MigrationPolicy {
public:
    migrationSota(
        const PerformanceCounters *performanceCounters,
        int coreRows,
        int coreColumns,
    float criticalTemperature);

    virtual std::vector<int> map(
        String taskName,
        int taskCoreRequirement,
        const std::vector<bool> &availableCores,
        const std::vector<bool> &activeCores);

    virtual std::vector<migration> migrate(
        SubsecondTime time,
        const std::vector<int> &taskIds,
        const std::vector<bool> &activeCores);

    std::vector<migration> createMigrations(
        std::unordered_map<int, int>& threads_core,
        std::unordered_map<int, int>& newthreads_core,
        std::vector<bool>& availableCores);

private:
    const PerformanceCounters *performanceCounters;
    unsigned int coreRows;
    unsigned int coreColumns;
    float criticalTemperature;
    int getColdestCore(const std::vector<bool> &availableCores);
    void logTemperatures(const std::vector<bool> &availableCores);
};

#endif
