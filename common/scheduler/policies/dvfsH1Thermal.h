/**
 * H1-style thermal DVFS policy.
 */

#ifndef __DVFS_H1_THERMAL_H
#define __DVFS_H1_THERMAL_H

#include "dvfspolicy.h"
#include "performance_counters.h"

#include <map>
#include <string>
#include <vector>

class DVFSH1Thermal : public DVFSPolicy {
public:
    DVFSH1Thermal(const PerformanceCounters *performanceCounters,
                  int numberOfCores,
                  const std::vector<double> &enabledStates,
                  const std::vector<int> &frequencies,
	                  const std::string &objective,
	                  double maxTemp,
	                  double thermalMargin,
	                  double powerBudget,
	                  double powerBudgetMargin,
	                  double perCorePowerGuard,
	                  const std::string &profileFile,
	                  bool debug = false);

    virtual std::vector<int> getFrequencies(const std::vector<int> &oldFrequencies, const std::vector<bool> &activeCores);

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

	    struct Candidate {
	        std::vector<int> states;
	        double totalIPS;
	        double totalPower;
	        double maxPredTemp;
	        int changedCores;
	        bool feasible;
	        bool valid;

	        Candidate();
	    };

    const PerformanceCounters *performanceCounters;
    int numberOfCores;
    std::vector<double> enabledStates;
    std::vector<int> frequencies;
	    std::string objective;
	    double maxTemp;
	    double thermalMargin;
	    double powerBudget;
	    double powerBudgetMargin;
	    double perCorePowerGuard;
	    std::string profileFile;
	    bool debug;
	    bool warnedFallback;
	    bool warnedInvalidPowerBudget;

    ProfileMap profile;

    void loadProfile();
    int stateKey(double state) const;
    bool hasAllEnabledStates(const std::string &benchmarkName) const;
    std::string findNearestBenchmark(double currentStateValue, double measuredIPS) const;
    int getStateForFrequency(int frequency) const;
    double getMeasuredIPSBillions(unsigned int coreId, double currentStateValue) const;
	    double getMeasuredPower(unsigned int coreId) const;
	    double getMeasuredTemperature(unsigned int coreId) const;
	    bool isPowerBudgetObjective() const;
	    double tempLimit() const;
	    double effectivePowerBudget() const;
	    void buildPrediction(unsigned int coreId,
                         int currentStateIndex,
                         double measuredIPS,
                         std::string &nearestBenchmark,
                         std::vector<double> &predIPS,
                         std::vector<double> &predPower,
                         std::vector<double> &predTemp);
	    int selectState(const std::vector<double> &predIPS,
	                    const std::vector<double> &predPower,
	                    const std::vector<double> &predTemp) const;
	    Candidate evaluatePowerBudgetCandidate(const std::vector<int> &chosenStates,
	                                           const std::vector<int> &currentStates,
	                                           const std::vector<std::vector<double> > &predIPS,
	                                           const std::vector<std::vector<double> > &predPower,
	                                           const std::vector<std::vector<double> > &predTemp) const;
	    bool isBetterPowerBudgetCandidate(const Candidate &candidate,
	                                      const Candidate &best,
	                                      bool requireFeasible) const;
	    std::vector<int> selectPowerBudgetStates(const std::vector<int> &currentStates,
	                                             const std::vector<std::vector<double> > &predIPS,
	                                             const std::vector<std::vector<double> > &predPower,
	                                             const std::vector<std::vector<double> > &predTemp,
	                                             Candidate &selected) const;
	    void logPrediction(unsigned int coreId,
	                       double measuredIPS,
	                       double currentStateValue,
	                       const std::string &benchmarkName,
	                       const std::vector<double> &ips,
	                       const std::vector<double> &power,
	                       const std::vector<double> &temp) const;
	};

#endif
