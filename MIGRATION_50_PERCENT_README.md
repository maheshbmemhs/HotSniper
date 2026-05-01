# Example Configuration for Migration at 50% Policy

To use the migration at 50% completion policy:

1. Edit your configuration file (e.g., config/base.cfg or your custom .cfg):

[scheduler/open/migration]
logic = at50percent              # Enable migration at 50% completion
epoch = 100000                   # Check every 100 microseconds (100000 ns)
expected_duration = 50000000     # Expected task duration: 50 milliseconds (50000000 ns)

2. Key Parameters:
   - logic: Set to "at50percent" to enable this policy
   - epoch: How frequently the migration policy is checked (in nanoseconds)
   - expected_duration: The expected execution time of tasks (in nanoseconds)
     * Tasks will be migrated when they reach 50% of this duration
     * Adjust based on your workload characteristics
     * Example values:
       - 10000000 ns = 10 ms
       - 50000000 ns = 50 ms
       - 100000000 ns = 100 ms

3. How It Works:
   - When a task starts, the policy records its start time
   - Every 'epoch' nanoseconds, the policy checks if any task has executed for >= 50% of expected_duration
   - If a task reaches 50% and there is a free core available, it migrates the task to that core
   - Each task is only migrated once

4. Example Output:
   [MigrationPolicy] Initialized MigrationAt50Percent with expected duration: 50000000 ns
   [MigrationPolicy] Task 0 started on core 0 at time 1000000 ns
   [MigrationPolicy] Task 0 reached 50% completion (elapsed: 25000000 ns, threshold: 25000000 ns)
   [MigrationPolicy] Scheduling migration of task 0 from core 0 to core 2
   [Scheduler] moving thread 123 from core 0 to core 2

5. To disable migration:
   Set logic = off in the [scheduler/open/migration] section
