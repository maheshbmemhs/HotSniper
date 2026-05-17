/**
 * This header implements a thermal-aware migration policy that selects
 * thread migrations based on the best ΔT/ΔIPS ratio.
 * The policy continues selecting migrations until thermal constraints are met.
 */
#ifndef __MIGRATION_IMP_H
#define __MIGRATION_IMP_H

#include <vector>
#include "migrationpolicy.h"
#include "performance_counters.h"

class MigrationImp : public MigrationPolicy {
public:
    MigrationImp(
        const PerformanceCounters *performanceCounters,
        int coreRows,
        int coreColumns,
        float criticalTemperature);

    virtual std::vector<migration> migrate(
        SubsecondTime time,
        const std::vector<int> &taskIds,
        const std::vector<bool> &activeCores);

private:
    const PerformanceCounters *performanceCounters;
    unsigned int coreRows;
    unsigned int coreColumns;
    float criticalTemperature;

    // Helper methods
    float getMaxTemperature(const std::vector<bool> &activeCores) const;
    float getAverageIPS(const std::vector<int> &taskIds, const std::vector<bool> &activeCores) const;
};

#endif
