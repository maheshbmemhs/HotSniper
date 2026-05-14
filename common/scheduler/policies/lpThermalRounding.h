/**
 * Runtime thermal-aware LP-rounding migration policy.
 *
 * This policy mirrors the Python prototype in lp_thermal_rounding.py at the
 * scheduler-policy level: build a fractional assignment, round it with thermal
 * pressure in the score, repair throughput if needed, then emit HotSniper
 * migrations/swaps.
 */
#ifndef __LP_THERMAL_ROUNDING_H
#define __LP_THERMAL_ROUNDING_H

#include <vector>
#include "migrationpolicy.h"
#include "performance_counters.h"

class LpThermalRounding : public MigrationPolicy {
public:
    LpThermalRounding(
        const PerformanceCounters *performanceCounters,
        int coreRows,
        int coreColumns,
        double targetIps,
        double alpha,
        double beta,
        double gamma,
        double lambdaTemp,
        double criticalTemperature,
        double minTemperatureDelta);

    virtual std::vector<migration> migrate(
        SubsecondTime time,
        const std::vector<int> &taskIds,
        const std::vector<bool> &activeCores);

private:
    const PerformanceCounters *performanceCounters;
    unsigned int coreRows;
    unsigned int coreColumns;
    double targetIps;
    double alpha;
    double beta;
    double gamma;
    double lambdaTemp;
    double criticalTemperature;
    double minTemperatureDelta;

    std::vector<std::vector<double> > solveLpRelaxation(
        const std::vector<std::vector<double> > &throughput,
        const std::vector<std::vector<double> > &power,
        double throughputRequirement) const;

    std::vector<int> thermalAwareCapacityRounding(
        const std::vector<std::vector<double> > &aLp,
        const std::vector<std::vector<double> > &power,
        const std::vector<double> &temperature) const;

    void repairAssignment(
        std::vector<int> &assignment,
        const std::vector<std::vector<double> > &throughput,
        const std::vector<std::vector<double> > &power,
        const std::vector<double> &temperature,
        double throughputRequirement) const;

    std::vector<migration> createMigrations(
        const std::vector<int> &sourceCores,
        const std::vector<int> &targetCores,
        const std::vector<int> &assignment) const;

    std::vector<int> greedyAssignment(
        const std::vector<std::vector<double> > &score) const;

    std::vector<std::vector<double> > assignmentToMatrix(
        const std::vector<int> &assignment,
        unsigned int numTargets) const;

    double assignmentThroughput(
        const std::vector<int> &assignment,
        const std::vector<std::vector<double> > &throughput) const;

    double assignmentPower(
        const std::vector<int> &assignment,
        const std::vector<std::vector<double> > &power) const;

    double measuredIpsBillions(unsigned int coreId) const;
    double measuredPower(unsigned int coreId) const;
    double measuredTemperature(unsigned int coreId) const;
    double frequencyScale(unsigned int fromCore, unsigned int toCore) const;
    bool finiteAndPositive(double value) const;
};

#endif
