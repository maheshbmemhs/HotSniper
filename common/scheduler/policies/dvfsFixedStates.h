/**
 * This header implements a fixed per-core DVFS policy.
 */

#ifndef __DVFS_FIXED_STATES_H
#define __DVFS_FIXED_STATES_H

#include "dvfspolicy.h"

#include <vector>

class DVFSFixedStates : public DVFSPolicy {
public:
    DVFSFixedStates(int numberOfCores, const std::vector<int> &frequencies);
    virtual std::vector<int> getFrequencies(const std::vector<int> &oldFrequencies, const std::vector<bool> &activeCores);

private:
    int numberOfCores;
    std::vector<int> frequencies;
    bool logged;
};

#endif
