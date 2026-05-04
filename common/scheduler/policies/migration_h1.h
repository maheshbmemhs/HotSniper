/**
 * This header implements the H1 migration policy.
 */

#ifndef __MIGRATION_H1_H
#define __MIGRATION_H1_H

#include "migrationpolicy.h"
#include "performance_counters.h"

#include <map>
#include <string>
#include <vector>

class MigrationH1 : public MigrationPolicy {
public:
    MigrationH1(const PerformanceCounters *performanceCounters,
                int numberOfCores,
                const std::vector<int> &coreToState,
                const std::vector<double> &enabledStates,
                double targetIPS,
                const std::string &objective,
                double maxTemp,
                double thermalMargin,
                double powerBudget,
                double powerBudgetMargin,
                double perCorePowerGuard,
                const std::string &profileFile,
                bool freezeMaster,
                bool debug = false);

    virtual std::vector<migration> migrate(SubsecondTime time, const std::vector<int> &taskIds, const std::vector<bool> &activeCores);
    virtual std::vector<migration> migrate(SubsecondTime time, const std::vector<int> &taskIds, const std::vector<int> &threadIds, const std::vector<bool> &activeCores);

private:
    struct ProfileEntry {
        double state;
        double ips;
        double cpi;
        double temp;
        double power;
    };

    typedef std::map<int, ProfileEntry> StateProfile;
    typedef std::map<std::string, StateProfile> ProfileMap;

    struct ThermalCandidate {
        std::vector<int> desiredCore;
        double totalIPS;
        double totalPower;
        double maxPredTemp;
        int movedItems;
        bool feasible;
        bool valid;

        ThermalCandidate();
    };

    const PerformanceCounters *performanceCounters;
    int numberOfCores;
    std::vector<int> coreToState;
    std::vector<double> enabledStates;
    double targetIPS;
    std::string objective;
    double maxTemp;
    double thermalMargin;
    double powerBudget;
    double powerBudgetMargin;
    double perCorePowerGuard;
    std::string profileFile;
    bool freezeMaster;
    bool debug;
    bool warnedFallback;
    bool warnedInvalidPowerBudget;

    ProfileMap profile;

    void loadProfile();
    int stateKey(double state) const;
    bool hasAllEnabledStates(const std::string &benchmarkName) const;
    std::string findNearestBenchmark(double currentStateValue, double measuredIPS) const;

    bool isMasterCore(const std::vector<int> &taskIds, const std::vector<int> &threadIds, unsigned int coreId) const;
    std::vector<unsigned int> getActiveCoreIds(const std::vector<int> &taskIds, const std::vector<int> &threadIds, const std::vector<bool> &activeCores) const;
    int getStateForCore(unsigned int coreId) const;
    double getMeasuredIPSBillions(unsigned int coreId, double currentStateValue) const;
    double getMeasuredPower(unsigned int coreId) const;
    double getMeasuredTemperature(unsigned int coreId) const;
    bool isThermalObjective() const;
    bool isPowerBudgetObjective() const;
    bool isTargetIPSMinPowerObjective() const;
    bool isTargetIPSObjective() const;
    double tempLimit() const;
    double effectivePowerBudget() const;

    void buildPredictions(const std::vector<unsigned int> &activeCoreIds,
                          std::vector<std::vector<double> > &predIPS,
                          std::vector<std::vector<double> > &predPower,
                          std::vector<std::vector<double> > &predTemp);

    std::vector<int> runH1(const std::vector<unsigned int> &activeCoreIds,
                           const std::vector<int> &taskIds,
                           const std::vector<int> &threadIds,
                           const std::vector<std::vector<double> > &predIPS,
                           const std::vector<std::vector<double> > &predPower);

    std::vector<int> runThermalMaxIPS(const std::vector<unsigned int> &activeCoreIds,
                                      const std::vector<int> &taskIds,
                                      const std::vector<int> &threadIds,
                                      const std::vector<std::vector<double> > &predIPS,
                                      const std::vector<std::vector<double> > &predPower,
                                      const std::vector<std::vector<double> > &predTemp,
                                      ThermalCandidate &selected) const;
    std::vector<int> runPowerBudgetMaxIPS(const std::vector<unsigned int> &activeCoreIds,
                                          const std::vector<int> &taskIds,
                                          const std::vector<int> &threadIds,
                                          const std::vector<std::vector<double> > &predIPS,
                                          const std::vector<std::vector<double> > &predPower,
                                          const std::vector<std::vector<double> > &predTemp,
                                          ThermalCandidate &selected) const;
    std::vector<int> runTargetIPSMinPower(const std::vector<unsigned int> &activeCoreIds,
                                          const std::vector<int> &taskIds,
                                          const std::vector<int> &threadIds,
                                          const std::vector<std::vector<double> > &predIPS,
                                          const std::vector<std::vector<double> > &predPower,
                                          const std::vector<std::vector<double> > &predTemp,
                                          ThermalCandidate &selected) const;
    std::vector<int> getAvailableFixedCores(const std::vector<unsigned int> &activeCoreIds,
                                            const std::vector<int> &taskIds,
                                            const std::vector<int> &threadIds) const;
    ThermalCandidate evaluateThermalCandidate(const std::vector<unsigned int> &activeCoreIds,
                                              const std::vector<int> &desiredCore,
                                              const std::vector<std::vector<double> > &predIPS,
                                              const std::vector<std::vector<double> > &predPower,
                                              const std::vector<std::vector<double> > &predTemp) const;
    ThermalCandidate evaluatePowerBudgetCandidate(const std::vector<unsigned int> &activeCoreIds,
                                                  const std::vector<int> &desiredCore,
                                                  const std::vector<std::vector<double> > &predIPS,
                                                  const std::vector<std::vector<double> > &predPower,
                                                  const std::vector<std::vector<double> > &predTemp) const;
    ThermalCandidate evaluateTargetIPSCandidate(const std::vector<unsigned int> &activeCoreIds,
                                                const std::vector<int> &desiredCore,
                                                const std::vector<std::vector<double> > &predIPS,
                                                const std::vector<std::vector<double> > &predPower,
                                                const std::vector<std::vector<double> > &predTemp) const;
    bool isBetterThermalCandidate(const ThermalCandidate &candidate,
                                  const ThermalCandidate &best,
                                  bool requireFeasible) const;
    bool isBetterPowerBudgetCandidate(const ThermalCandidate &candidate,
                                      const ThermalCandidate &best,
                                      bool requireFeasible) const;
    bool isBetterTargetIPSCandidate(const ThermalCandidate &candidate,
                                    const ThermalCandidate &best,
                                    bool requireFeasible) const;

    std::vector<migration> convertDesiredStatesToMigrations(const std::vector<unsigned int> &activeCoreIds,
                                                            const std::vector<int> &desiredState,
                                                            const std::vector<int> &taskIds,
                                                            const std::vector<int> &threadIds) const;
    std::vector<migration> convertDesiredCoresToMigrations(const std::vector<unsigned int> &activeCoreIds,
                                                           const std::vector<int> &desiredCore,
                                                           const std::vector<int> &taskIds,
                                                           const std::vector<int> &threadIds) const;

    double predictedTotalIPS(const std::vector<std::vector<double> > &predIPS, const std::vector<int> &states) const;
    double predictedCurrentIPS(const std::vector<unsigned int> &activeCoreIds, const std::vector<std::vector<double> > &predIPS) const;
    void logPrediction(unsigned int coreId,
                       double measuredIPS,
                       double currentStateValue,
                       const std::string &benchmarkName,
                       const std::vector<double> &ips,
                       const std::vector<double> &power,
                       const std::vector<double> &temp) const;
};

#endif
