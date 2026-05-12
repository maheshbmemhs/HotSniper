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
        double startPeakTempC{-1.0};
    };

    explicit ThermalRegressionModel(const std::string& modelPath);

    std::vector<double> predictEndTemperatures(const Inputs& inputs);

    unsigned int getNumCores() const { return numCores; }
    bool isEnabled() const { return enabled; }
    void setDebug(bool debugEnabled) { debug = debugEnabled; }

private:
    enum ModelType {
        MODEL_NONE,
        MODEL_LINEAR,
        MODEL_LGBM
    };

    struct LgbmNode {
        bool isLeaf{false};
        bool isCategorical{false};
        unsigned int splitFeature{0};
        double threshold{0.0};
        int left{-1};
        int right{-1};
        double leafValue{0.0};
        std::vector<int> catValues;
    };

    typedef std::vector<LgbmNode> LgbmTree;

    bool enabled{false};
    bool phaseAware{false};
    bool usesPhaseInteractions{false};
    bool debug{false};
    ModelType modelType{MODEL_NONE};
    unsigned int numCores{0};
    unsigned int numFeatures{0};
    unsigned int numTargets{0};

    std::vector<std::string> phaseNames;
    std::vector<std::string> featureNames;
    std::vector<std::string> targetNames;
    std::vector<double> xMean;
    std::vector<double> xScale;
    std::vector<double> yMean;
    std::vector<double> yScale;
    std::vector<std::vector<double> > coefficients;
    std::vector<double> intercept;
    std::vector<unsigned int> categoricalFeatureIndices;
    std::vector<std::vector<LgbmTree> > lgbmTrees;

    void loadV1(std::ifstream& file);
    void loadV2PhaseAware(std::ifstream& file);
    void loadLgbmV1(std::ifstream& file);
    void validateModel() const;
    void disable(const std::string& reason);

    std::vector<double> buildFeatures(const Inputs& inputs, std::string* classifiedPhase = nullptr) const;
    std::vector<double> buildLgbmFeatures(const Inputs& inputs, std::string* classifiedPhase = nullptr) const;
    double predictLgbmDelta(unsigned int target, const std::vector<double>& features) const;
    void validateInputs(const Inputs& inputs) const;
};

#endif
