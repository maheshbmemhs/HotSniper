#include "dvfsFixedStates.h"

#include <iostream>

using namespace std;

DVFSFixedStates::DVFSFixedStates(int numberOfCores, const vector<int> &frequencies)
    : numberOfCores(numberOfCores)
    , frequencies(frequencies)
    , logged(false)
{
    if ((int)this->frequencies.size() < numberOfCores) {
        cout << "[Scheduler][DVFSFixedStates][Warning]: fewer fixed frequencies than cores; missing cores keep their old frequency." << endl;
    }
}

vector<int> DVFSFixedStates::getFrequencies(const vector<int> &oldFrequencies, const vector<bool> &activeCores)
{
    (void)activeCores;

    vector<int> result(numberOfCores);
    for (int core = 0; core < numberOfCores; core++) {
        if (core < (int)frequencies.size() && frequencies.at(core) > 0) {
            result.at(core) = frequencies.at(core);
        } else if (core < (int)oldFrequencies.size()) {
            result.at(core) = oldFrequencies.at(core);
        } else {
            result.at(core) = 0;
        }
    }

    if (!logged) {
        cout << "[Scheduler][DVFSFixedStates]: fixed frequencies MHz=[";
        for (int core = 0; core < numberOfCores; core++) {
            if (core > 0) {
                cout << ",";
            }
            cout << result.at(core);
        }
        cout << "]" << endl;
        logged = true;
    }

    return result;
}
