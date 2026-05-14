/**
 * scheduler_open
 * This header implements open scheduler functionality of SniperPlus. The class is extended from default "Pinned" scheduler.
 */


#ifndef __SCHEDULER_OPEN_H
#define __SCHEDULER_OPEN_H

#include "scheduler_pinned_base.h"
#include "thermalComponentModel.h"
#include "thermalModel.h"
#include "performance_counters.h"
#include "policies/dvfspolicy.h"
#include "policies/mappingpolicy.h"
#include "policies/migrationpolicy.h"
#include "policies/dynThreadMapping.h"

#include <fstream>

class SchedulerOpen : public SchedulerPinnedBase {

	public:
		SchedulerOpen (ThreadManager *thread_manager); //This function is the constructor for Open System Scheduler.
		virtual void periodic(SubsecondTime time);
		virtual void threadSetInitialAffinity(thread_id_t thread_id);
		virtual bool threadSetAffinity(thread_id_t calling_thread_id, thread_id_t thread_id, size_t cpusetsize, const cpu_set_t *mask);
		virtual core_id_t threadCreate(thread_id_t thread_id);
		virtual void threadExit(thread_id_t thread_id, SubsecondTime time);

		

	private:
		int coreRows;
		int coreColumns;
		int nodesPerCore;

		PerformanceCounters *performanceCounters;
		MappingPolicy *mappingPolicy = NULL;
		long mappingEpoch;
		void initMappingPolicy(String policyName);
		bool executeMappingPolicy(int taskID, SubsecondTime time);
		int getCoreNb(int y, int x);
		bool isAssignedToTask(int coreId);
		bool isAssignedToThread(int coreId);

		DynThreadMapping* dynThdMap;

		DVFSPolicy *dvfsPolicy = NULL;
		long dvfsEpoch;
		void initDVFSPolicy(String policyName);
		void executeDVFSPolicy();
		const int maxDVFSPatience = 0;
		std::vector<int> downscalingPatience; // can be used by the DVFS control loop to delay DVFS downscaling for very little violations
		std::vector<int> upscalingPatience; // can be used by the DVFS control loop to delay DVFS upscaling for very little violations
		bool delayDVFSTransition(int coreCounter, int oldFrequency, int newFrequency);
		void DVFSTransitionDelayed(int coreCounter, int oldFrequency, int newFrequency);
		void DVFSTransitionNotDelayed(int coreCounter);
		void setFrequency(int coreCounter, int frequency);
		ThermalComponentModel *thermalComponentModel;
		ThermalModel *thermalModel;
		int minFrequency;
		int maxFrequency;
		int frequencyStepSize;

		void initPerforationPolicy(String policyName, int taskCount);
		void executePerforationPolicy();

		MigrationPolicy *migrationPolicy = NULL;
		long migrationEpoch;
		void initMigrationPolicy(String policyName);
		void executeMigrationPolicy(SubsecondTime time);
		void migrateThread(thread_id_t thread_id, core_id_t core_id);

		std::string formatTime(SubsecondTime time);

		struct ThermalSampleSnapshot {
			UInt64 timeNs;
			std::vector<int> frequenciesMhz;
			std::vector<int> activeCores;
			std::vector<int> threadActive;
			std::vector<int> taskIds;
			std::vector<int> threadIds;
			std::vector<double> tempsC;
			std::vector<double> powersW;
			std::vector<double> utilizations;
			std::vector<double> cpis;
			std::vector<double> relNucaCpis;
			std::vector<double> ips;
		};
			bool thermalSampleEnabled = false;
			bool thermalSampleDebug = false;
			bool hasThermalSampleStart = false;
		UInt64 thermalSampleCycle = 0;
		long thermalSampleEpoch = 1000000;
		std::string thermalSamplePath;
		std::ofstream thermalSampleFile;
			ThermalSampleSnapshot thermalSampleStart;
			std::vector<int> thermalSampleNewFrequenciesMhz;
			std::vector<double> thermalSamplePredictedTempsC;
			void initThermalSampler();
			ThermalSampleSnapshot captureThermalSampleSnapshot(SubsecondTime time);
			void writeThermalSampleRow(const ThermalSampleSnapshot &start, const ThermalSampleSnapshot &end, const std::vector<int> &newFrequenciesMhz, const std::vector<double> &predictedTempsC);
		void updateThermalSamplerBeforePolicies(SubsecondTime time);
		void updateThermalSamplerAfterPolicies();

		core_id_t getNextCore(core_id_t core_first);
		core_id_t getFreeCore(core_id_t core_first);

		const int m_interleaving;
		std::vector<bool> m_core_mask;
		core_id_t m_next_core;

		int setAffinity (thread_id_t thread_id);
		bool schedule (int taskID, bool isInitialCall, SubsecondTime time);
};

#endif // __SCHEDULER_OPEN_H
