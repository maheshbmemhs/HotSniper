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
                const std::string &profileFile,
                bool debug = false);

    virtual std::vector<migration> migrate(SubsecondTime time, const std::vector<int> &taskIds, const std::vector<bool> &activeCores);

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

    const PerformanceCounters *performanceCounters;
    int numberOfCores;
    std::vector<int> coreToState;
    std::vector<double> enabledStates;
    double targetIPS;
    std::string profileFile;
    bool debug;
    bool warnedFallback;

    ProfileMap profile;

    void loadProfile();
    int stateKey(double state) const;
    bool hasAllEnabledStates(const std::string &benchmarkName) const;
    std::string findNearestBenchmark(double currentStateValue, double measuredIPS) const;

    std::vector<unsigned int> getActiveCoreIds(const std::vector<int> &taskIds, const std::vector<bool> &activeCores) const;
    int getStateForCore(unsigned int coreId) const;
    double getMeasuredIPSBillions(unsigned int coreId, double currentStateValue) const;
    double getMeasuredPower(unsigned int coreId) const;

    void buildPredictions(const std::vector<unsigned int> &activeCoreIds,
                          std::vector<std::vector<double> > &predIPS,
                          std::vector<std::vector<double> > &predPower);

    std::vector<int> runH1(const std::vector<unsigned int> &activeCoreIds,
                           const std::vector<std::vector<double> > &predIPS,
                           const std::vector<std::vector<double> > &predPower);

    std::vector<migration> convertDesiredStatesToMigrations(const std::vector<unsigned int> &activeCoreIds,
                                                            const std::vector<int> &desiredState,
                                                            const std::vector<bool> &activeCores) const;

    double predictedTotalIPS(const std::vector<std::vector<double> > &predIPS, const std::vector<int> &states) const;
    double predictedCurrentIPS(const std::vector<unsigned int> &activeCoreIds, const std::vector<std::vector<double> > &predIPS) const;
    void logPrediction(unsigned int coreId,
                       double measuredIPS,
                       double currentStateValue,
                       const std::string &benchmarkName,
                       const std::vector<double> &ips,
                       const std::vector<double> &power) const;
};

#endif
