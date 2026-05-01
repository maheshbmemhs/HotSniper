/**
 * Implementation of migration policy that migrates tasks
 * after 50% of their expected execution time.
 */

#include "migrationAt50Percent.h"
#include <iostream>

MigrationAt50Percent::MigrationAt50Percent(SubsecondTime expected)
    : expectedDuration(expected)
{
    std::cout << "[MigrationPolicy] Initialized MigrationAt50Percent with expected duration: " 
              << expected.getNS() << " ns" << std::endl;
}

std::vector<migration> MigrationAt50Percent::migrate(
    SubsecondTime time,
    const std::vector<int> &taskIds,
    const std::vector<bool> &activeCores
) {
    std::vector<migration> migrations;
    
    // Track which cores are already targeted for migration in this call
    std::vector<bool> targetedCores = activeCores;  // Start with current active cores
    
    // Iterate through all cores to check task progress
    for (unsigned int coreId = 0; coreId < taskIds.size(); coreId++) {
        int taskId = taskIds[coreId];
        
        // Skip if no task on this core or core is not active
        if (taskId == -1 || !activeCores[coreId]) {
            continue;
        }
        
        // Create a key for this (task, core) pair
        std::pair<int, unsigned int> key = std::make_pair(taskId, coreId);
        
        // Record start time if this is the first time seeing this task on this core
        if (coreTaskStartTimes.find(key) == coreTaskStartTimes.end()) {
            coreTaskStartTimes[key] = time;
            hasMigrated[key] = false;
            std::cout << "[MigrationPolicy] Task " << taskId 
                      << " started on core " << coreId 
                      << " at time " << time.getNS() << " ns" << std::endl;
        }
        
        // Calculate elapsed time for this task on this specific core
        SubsecondTime elapsed = time - coreTaskStartTimes[key];
        SubsecondTime halfDuration = expectedDuration / 2;
        
        // Check if 50% completed and not already migrated
        if (!hasMigrated[key] && elapsed >= halfDuration) {
            std::cout << "[MigrationPolicy] Task " << taskId 
                      << " on core " << coreId
                      << " reached 50% completion (elapsed: " << elapsed.getNS() 
                      << " ns, threshold: " << halfDuration.getNS() << " ns)" << std::endl;
            
            // Find a free core to migrate to (not active and not already targeted)
            for (unsigned int targetCore = 0; targetCore < activeCores.size(); targetCore++) {
                if (!targetedCores[targetCore] && targetCore != coreId) {
                    migration m;
                    m.fromCore = coreId;
                    m.toCore = targetCore;
                    m.swap = false;
                    migrations.push_back(m);
                    hasMigrated[key] = true;
                    
                    // Mark this core as targeted so it won't be selected again in this call
                    targetedCores[targetCore] = true;
                    
                    std::cout << "[MigrationPolicy] Scheduling migration of task " << taskId 
                              << " from core " << coreId << " to core " << targetCore << std::endl;
                    break;  // Only migrate once per task per core
                }
            }
            
            // If no free core found, log it
            if (!hasMigrated[key]) {
                std::cout << "[MigrationPolicy] Task " << taskId 
                          << " on core " << coreId
                          << " at 50% but no free core available for migration" << std::endl;
            }
        }
    }
    
    return migrations;
}
