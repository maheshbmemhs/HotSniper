/**
 * This header implements a migration policy that migrates tasks
 * after they have completed 50% of their expected execution time.
 */

#ifndef __MIGRATION_AT_50_PERCENT_H
#define __MIGRATION_AT_50_PERCENT_H

#include "migrationpolicy.h"
#include <map>
#include <utility>

class MigrationAt50Percent : public MigrationPolicy {
private:
    // Track when each (taskId, coreId) pair started - key is (taskId, coreId)
    std::map<std::pair<int, unsigned int>, SubsecondTime> coreTaskStartTimes;
    // Track if task on specific core has already migrated - key is (taskId, coreId)
    std::map<std::pair<int, unsigned int>, bool> hasMigrated;
    SubsecondTime expectedDuration;               // Expected task duration

public:
    MigrationAt50Percent(SubsecondTime expected);
    virtual ~MigrationAt50Percent() {}
    
    virtual std::vector<migration> migrate(
        SubsecondTime time,
        const std::vector<int> &taskIds,
        const std::vector<bool> &activeCores
    ) override;
};

#endif // __MIGRATION_AT_50_PERCENT_H
