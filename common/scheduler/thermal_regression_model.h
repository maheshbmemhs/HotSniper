#ifndef __THERMAL_REGRESSION_MODEL_H
#define __THERMAL_REGRESSION_MODEL_H

#include <string>
#include <vector>

class ThermalRegressionModel {
public:
    struct Inputs {
        std::vector<double> oldFrequenciesMhz;
        std::vector<double> newFrequenciesMhz;
        std::vector<double> activeCores;
        std::vector<double> startTempsC;
        std::vector<double> startPowersW;
        std::vector<double> startUtilizations;
        std::vector<double> startCpis;
        std::vector<double> startRelNucaCpis;
        std::vector<double> startIps;
    };

    explicit ThermalRegressionModel(const std::string& modelPath);

    std::vector<double> predictEndTemperatures(const Inputs& inputs) const;

    unsigned int getNumCores() const { return numCores; }

private:
    unsigned int numCores{0};
    unsigned int numFeatures{0};
    unsigned int numTargets{0};

    std::vector<std::string> featureNames;
    std::vector<std::string> targetNames;
    std::vector<double> xMean;
    std::vector<double> xScale;
    std::vector<double> yMean;
    std::vector<double> yScale;
    std::vector<std::vector<double> > coefficients;
    std::vector<double> intercept;

    std::vector<double> buildFeatures(const Inputs& inputs) const;
    void validateInputs(const Inputs& inputs) const;
};

#endif
