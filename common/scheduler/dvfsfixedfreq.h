/**
* This header implements the fixed frequency DVFS policy.
*/
#ifndef __DVFS_FIXEDFREQ_H
#define __DVFS_FIXEDFREQ_H
#include <vector>
#include "dvfspolicy.h"
#include "performance_counters.h"
class DVFSFixedFreq : public DVFSPolicy {
public:
    DVFSFixedFreq(
    const PerformanceCounters *performanceCounters,
        int coreRows,
        int coreColumns,
        int minFrequency,
        int maxFrequency
    );

    virtual std::vector<int> getFrequencies(
        const std::vector<int> &oldFrequencies,
        const std::vector<bool> &activeCores);
        
private:
    const PerformanceCounters *performanceCounters;
    unsigned int coreRows;
    unsigned int coreColumns;
    int minFrequency;
    int maxFrequency;
    };
#endif