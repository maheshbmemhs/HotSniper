#include "dvfsfixedfreq.h"
#include <iomanip>
#include <iostream>

using namespace std;

DVFSFixedFreq::DVFSFixedFreq(
    const PerformanceCounters *performanceCounters,
    int coreRows,
    int coreColumns,
    int minFrequency,
    int maxFrequency)
    : performanceCounters(performanceCounters),
      coreRows(coreRows),
      coreColumns(coreColumns),
      minFrequency(minFrequency),
      maxFrequency(maxFrequency) {
}

std::vector<int> DVFSFixedFreq::getFrequencies(const std::vector<int> &oldFrequencies,const std::vector<bool> &activeCores) {
    std::vector<int> frequencies(coreRows * coreColumns, maxFrequency);
    frequencies.at(0)=1000;
    frequencies.at(1)=2000;
    frequencies.at(2)=3000;
    frequencies.at(3)=4000;

    return frequencies;
}