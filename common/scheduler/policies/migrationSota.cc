#include "migrationSota.h"
#include <iomanip>
#include <unordered_map>
#include "neighbor_prediction.h"
#include <algorithm> 
#include <set>
#include <climits> 
using namespace std;

migrationSota::migrationSota(
    const PerformanceCounters *performanceCounters,
    int coreRows,
    int coreColumns,
    float criticalTemperature)
    : performanceCounters(performanceCounters),
      coreRows(coreRows),
      coreColumns(coreColumns),
      criticalTemperature(criticalTemperature) {
}

std::vector<int> migrationSota::map( //Initial placement decision
    String taskName,
    int taskCoreRequirement,
    const std::vector<bool> &availableCoresRO,
    const std::vector<bool> &activeCores) {

    std::vector<bool> availableCores(availableCoresRO);
    std::vector<int> cores;

    logTemperatures(availableCores);


    // for (; taskCoreRequirement > 0; taskCoreRequirement--) {
    //     int coldestCore = getColdestCore(availableCores);

    //     if (coldestCore == -1) {
    //         // not enough free cores
    //         std::vector<int> empty;
    //         return empty;
    //     } else {
    //         cores.push_back(coldestCore);
    //         availableCores.at(coldestCore) = false;
    //     }
    // }
    // cores.push_back(0);
    // cores.push_back(1);
    // cores.push_back(3);
    
    // return cores;
}

std::vector<migration> migrationSota::migrate(
    SubsecondTime time,
    const std::vector<int> &taskIds,
    const std::vector<bool> &activeCores) {

    std::vector<migration> migrations;
    std::vector<bool> availableCores(coreRows * coreColumns);
    int tot_threads = 0;
    std::unordered_map<int, int> threads_core;
    std::unordered_map<int, int> newthreads_core;
    std::unordered_map<int, int> newcore_ips;
    

    int temp;
    float T=4.3;
    NeighborPrediction  pred("profile.txt");

    for (int c = 0; c < coreRows * coreColumns; c++) {
        availableCores.at(c) = taskIds.at(c) == -1;
        // newthreads_core[c] = -1;
        if(taskIds.at(c) != -1) {
            threads_core[c] = c; //taskIds.at(c)
            newthreads_core[c] = -1;
            tot_threads++;
        }
    }

    std::unordered_map<int, int> original_threads_core = threads_core;
    // int *threads_core = new int[tot_threads];
    // for (int i = 0; i < tot_threads; i++) {
    //     threads_core[i] = -1;
    // }

    cout << "[DEBUG] taskIds array: [";
    for (int c = 0; c < coreRows * coreColumns; c++) {
        cout << taskIds.at(c);
        if (c < coreRows * coreColumns - 1) cout << ", ";
    }
    cout << "]" << endl;


    cout << "[DEBUG] Total threads found: " << tot_threads << endl;
    cout << "[DEBUG] Threads in system:" << endl;
    for (auto& pair : threads_core) {
        cout << "  Thread " << pair.first << " currently on core " << pair.second << endl;
    }

    int thread_idx;
    std::unordered_map<int, int> threads_power;
    for (int c = coreRows * coreColumns - 1; c >= 0; c--) {
        cout << "[DEBUG] Processing core " << c << ", threads available for assignment:" << endl;
        for (auto& pair : threads_power) {
            cout << "  Thread " << pair.first << " power=" << pair.second << endl;
        }

        std::unordered_map<int, int> threads_power;
        for (auto& pair : newthreads_core){
            if(newthreads_core[pair.first] == -1) {
                auto IPS = performanceCounters->getIPSOfCore(threads_core[pair.first]);
                auto near = pred.getNearestBenchmark(c+1,IPS);
                threads_power[pair.first] =near[c+1].power ;//(rand() % 100 + 1);//performanceCounters->getPowerOfCore(coreId);
            } else {
                threads_power[pair.first]  = INT_MAX;//-1;
            }
           
        }

        int maxVal=-1;
        int minVal=INT_MAX;
        // for (auto& pair : threads_power){
        //     if(threads_power[pair.first] > maxVal) {
        //         maxVal = threads_power[pair.first];
        //         thread_idx = pair.first;
        //     }
        // }
        for (auto& pair : threads_power){
            if(threads_power[pair.first] < minVal) {
                minVal = threads_power[pair.first];
                thread_idx = pair.first;
            }
        }

        if(minVal != INT_MAX) {
            newthreads_core[thread_idx] = c;
            cout << "[DEBUG] Core " << c << " assigned thread " << thread_idx << " with power " << minVal << endl;
            auto IPS = performanceCounters->getIPSOfCore(threads_core[thread_idx]);
            auto near = pred.getNearestBenchmark(c+1,IPS);
            newcore_ips[c] = near[c+1].ips;
            // cout << "[DEBUG] Core " << c << " predicted IPS after migration: " << newcore_ips[c] << endl;
        }
    }

    double ips_sum = 0;
    for(auto& pair:newcore_ips){
        ips_sum+=pair.second;
    }
    auto avg_ips = ips_sum / tot_threads;
    cout << "[DEBUG] Average predicted IPS after initial assignment: " << avg_ips << endl;
    if(avg_ips >= T){

        migrations = createMigrations(threads_core, newthreads_core, availableCores);
        return migrations;
    }
    
    // Apply initial assignment
    threads_core = newthreads_core;

    // ===== MOVE GENERATION PHASE =====
    struct Exchange {
        int thread1;  // Thread on core c
        int thread2;  // Thread on core c+1
        int core1;    // Core c
        int core2;    // Core c+1
        double priority;  // Delta_throughput / Delta_power
    };

    std::vector<Exchange> all_exchanges;

    for (int c = 0; c < coreRows * coreColumns - 1; c++) {
        // Find threads on core c and c+1
        int t1 = -1, t2 = -1;
        
        for (auto& pair : threads_core) {
            if (pair.second == c) t1 = pair.first;
            if (pair.second == c + 1) t2 = pair.first;
        }

        if (t1 == -1 || t2 == -1) continue;  // No threads to swap

        // Calculate current throughput and power
        double current_throughput = 0, current_power = 0;
        for (auto& pair : threads_core) {
            int thread = pair.first;
            int core = pair.second;
            auto IPS = performanceCounters->getIPSOfCore(core);
            auto near = pred.getNearestBenchmark(core + 1, IPS);
            current_throughput += near[core+1].ips;
            current_power += near[core+1].power;
        }

        // Simulate swap: t1 goes to c+1, t2 goes to c
        double new_throughput = 0, new_power = 0;
        for (auto& pair : threads_core) {
            int thread = pair.first;
            int core = pair.second;
            
            // Determine which core this thread will be on after swap
            int new_core = core;
            if (thread == t1) new_core = c + 1;
            else if (thread == t2) new_core = c;

            auto IPS = performanceCounters->getIPSOfCore(threads_core[thread]);
            auto near = pred.getNearestBenchmark(new_core + 1, IPS);
            new_throughput += near[new_core+1].ips;
            new_power += near[new_core+1].power;
        }

        double delta_throughput = new_throughput - current_throughput;
        double delta_power = new_power - current_power;
        double priority = (delta_power != 0) ? delta_throughput / delta_power : 0;

        if (priority > 0) {
            Exchange ex;
            ex.thread1 = t1;
            ex.thread2 = t2;
            ex.core1 = c;
            ex.core2 = c + 1;
            ex.priority = priority;
            all_exchanges.push_back(ex);
            
            cout << "[DEBUG] Potential exchange: Thread " << t1 << " (core " << c 
                 << ") <-> Thread " << t2 << " (core " << c+1 
                 << "), priority=" << priority << endl;
        }
    }

    // Sort exchanges by priority (descending)
    std::sort(all_exchanges.begin(), all_exchanges.end(), 
              [](const Exchange& a, const Exchange& b) { return a.priority > b.priority; });

    // ===== EXCHANGE PHASE =====
    double current_avg_throughput = 0;
    for (auto& pair : newcore_ips) {
        current_avg_throughput += pair.second;
    }
    current_avg_throughput /= tot_threads;

    size_t exchange_idx = 0;
    while (current_avg_throughput < T && exchange_idx < all_exchanges.size()) {
        Exchange& best_swap = all_exchanges[exchange_idx];
        
        // Perform swap
        int temp_core = threads_core[best_swap.thread1];
        threads_core[best_swap.thread1] = threads_core[best_swap.thread2];
        threads_core[best_swap.thread2] = temp_core;

        cout << "[DEBUG] Performing exchange: Thread " << best_swap.thread1 
             << " <-> Thread " << best_swap.thread2 << endl;

        // Recalculate throughput
        current_avg_throughput = 0;
        for (auto& pair : threads_core) {
            int thread = pair.first;
            int core = pair.second;
            auto IPS = performanceCounters->getIPSOfCore(threads_core[thread]);
            auto near = pred.getNearestBenchmark(core + 1, IPS);
            current_avg_throughput += near[core+1].ips;
        }
        current_avg_throughput /= tot_threads;

        exchange_idx++;
    }

    cout << "[DEBUG] Final average throughput: " << current_avg_throughput << endl;

    // Create migrations based on final thread assignment
    migrations = createMigrations(original_threads_core, threads_core, availableCores);
    
    return migrations;
}

// std::vector<migration> migrationSota::createMigrations(
//     std::unordered_map<int, int>& original_positions,
//     std::unordered_map<int, int>& final_positions,
//     std::vector<bool>& availableCores) {
    
//     std::vector<migration> migrations;
//     std::unordered_map<int, int> current_positions = original_positions;
//     std::set<int> processed_threads;
    
//     for (auto& pair : final_positions) {
//         int thread = pair.first;
//         int final_core = pair.second;
        
//         if (processed_threads.count(thread) > 0) {
//             continue;  // Already processed
//         }
        
//         int original_core = current_positions[thread];
        
//         if (original_core == final_core) {
//             continue;  // No movement
//         }
        
//         // Check if there's a thread at the destination in current state
//         int thread_at_dest = -1;
//         for (auto& pair2 : current_positions) {
//             if (pair2.second == final_core && pair2.first != thread) {
//                 thread_at_dest = pair2.first;
//                 break;
//             }
//         }
        
//         migration m;
//         m.fromCore = original_core;
//         m.toCore = final_core;
        
//         if (thread_at_dest == -1) {
//             // Destination is empty - simple move
//             m.swap = false;
//             current_positions[thread] = final_core;
            
//             cout << "[DEBUG] Move: Thread " << thread << " from core " 
//                  << original_core << " -> core " << final_core << endl;
//         } else {
//             // Check if it's a mutual swap
//             if (final_positions[thread_at_dest] == original_core) {
//                 m.swap = true;
//                 current_positions[thread] = final_core;
//                 current_positions[thread_at_dest] = original_core;
//                 processed_threads.insert(thread_at_dest);
                
//                 cout << "[DEBUG] Swap: Thread " << thread << " (core " << original_core 
//                      << ") <-> Thread " << thread_at_dest << " (core " << final_core << ")" << endl;
//             } else {
//                 // Destination occupied but not a swap - error case
//                 cout << "[ERROR] Thread " << thread << " wants core " << final_core 
//                      << " but it's occupied by thread " << thread_at_dest 
//                      << " which is moving to core " << final_positions[thread_at_dest] << endl;
//                 continue;  // Skip this migration
//             }
//         }
        
//         migrations.push_back(m);
//         processed_threads.insert(thread);
//     }
    
//     cout << "[DEBUG] Total migrations: " << migrations.size() << endl;
//     return migrations;
// }



std::vector<migration> migrationSota::createMigrations(
    std::unordered_map<int, int>& threads_core,
    std::unordered_map<int, int>& newthreads_core,
    std::vector<bool>& availableCores) {
    
    std::vector<migration> migrations;
    int temp;
    
    for (auto& pair : newthreads_core) {
        if(threads_core[pair.first] != newthreads_core[pair.first]) {
            migration m;
            m.fromCore = threads_core[pair.first];
            m.toCore = newthreads_core[pair.first];
            
            if(availableCores.at(newthreads_core[pair.first])) {
                m.swap = false;
                availableCores.at(newthreads_core[pair.first]) = false;
                availableCores.at(threads_core[pair.first]) = true;
                threads_core[pair.first] = newthreads_core[pair.first];
            } else {
                m.swap = true;
                temp = threads_core[pair.first];
                
                for (auto& pair2 : threads_core) {
                    if(pair2.second == newthreads_core[pair.first]) {
                        threads_core[pair2.first] = temp;
                        break;
                    }
                }
                threads_core[pair.first] = newthreads_core[pair.first];
            }
            
            migrations.push_back(m);
        }
    }
    
    cout << "[DEBUG] Total migrations: " << migrations.size() << endl;
    for (size_t i = 0; i < migrations.size(); i++) {
        cout << "[DEBUG] Migration " << i << ": Thread from core " 
             << migrations[i].fromCore << " -> core " << migrations[i].toCore 
             << " (swap=" << (migrations[i].swap ? "true" : "false") << ")" << endl;
    }
    
    return migrations;
}






int migrationSota::getColdestCore(const std::vector<bool> &availableCores) {
    int coldestCore = -1;
    float coldestTemperature = 0;

    // iterate all cores to find coldest
    for (int c = 0; c < coreRows * coreColumns; c++) {
        if (availableCores.at(c)) {
            float temperature = performanceCounters->getTemperatureOfCore(c);

            if ((coldestCore == -1) || (temperature < coldestTemperature)) {
                coldestCore = c;
                coldestTemperature = temperature;
            }
        }
    }

    return coldestCore;
}

void migrationSota::logTemperatures(const std::vector<bool> &availableCores) {
    cout << "[Scheduler][coldestCore-map]: temperatures of available cores:" << endl;

    for (int y = 0; y < coreRows; y++) {
        for (int x = 0; x < coreColumns; x++) {
            if (x > 0) {
                cout << " ";
            }

            int coreId = y * coreColumns + x;

            if (!availableCores.at(coreId)) {
                cout << " - ";
            } else {
                float temperature = performanceCounters->getTemperatureOfCore(coreId);
                cout << fixed << setprecision(1) << temperature;
            }
        }
        cout << endl;
    }
}