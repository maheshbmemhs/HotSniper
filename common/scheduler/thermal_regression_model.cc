#include "thermal_regression_model.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace {

std::string readToken(std::ifstream& file, const std::string& expected)
{
    std::string token;
    file >> token;
    if (token != expected) {
        throw std::runtime_error("Thermal regression model parse error: expected " + expected + ", got " + token);
    }
    return token;
}

unsigned int readLabeledUInt(std::ifstream& file, const std::string& label)
{
    readToken(file, label);
    unsigned int value = 0;
    file >> value;
    if (!file) {
        throw std::runtime_error("Thermal regression model parse error while reading " + label);
    }
    return value;
}

std::vector<std::string> readNames(std::ifstream& file, const std::string& label, unsigned int count)
{
    readToken(file, label);
    std::vector<std::string> names;
    names.reserve(count);
    for (unsigned int i = 0; i < count; i++) {
        std::string name;
        file >> name;
        if (!file) {
            throw std::runtime_error("Thermal regression model parse error while reading " + label);
        }
        names.push_back(name);
    }
    return names;
}

std::vector<double> readVector(std::ifstream& file, const std::string& label, unsigned int count)
{
    readToken(file, label);
    std::vector<double> values(count);
    for (unsigned int i = 0; i < count; i++) {
        file >> values[i];
        if (!file) {
            throw std::runtime_error("Thermal regression model parse error while reading " + label);
        }
    }
    return values;
}

double mean(const std::vector<double>& values)
{
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double populationStd(const std::vector<double>& values)
{
    if (values.empty()) {
        return 0.0;
    }

    const double avg = mean(values);
    double sumSquareDiff = 0.0;
    for (double value : values) {
        const double diff = value - avg;
        sumSquareDiff += diff * diff;
    }
    return std::sqrt(sumSquareDiff / values.size());
}

std::vector<double> neighborMean(const std::vector<double>& values)
{
    std::vector<double> result(values.size(), 0.0);
    if (values.size() <= 1) {
        return result;
    }

    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    for (unsigned int i = 0; i < values.size(); i++) {
        result[i] = (sum - values[i]) / (values.size() - 1);
    }
    return result;
}

std::vector<double> cpiToIpc(const std::vector<double>& cpis)
{
    std::vector<double> ipc(cpis.size(), 0.0);
    for (unsigned int i = 0; i < cpis.size(); i++) {
        if (cpis[i] < 100.0) {
            ipc[i] = 1.0 / std::max(cpis[i], 1e-6);
        }
    }
    return ipc;
}

std::vector<double> safeLogIps(const std::vector<double>& ips)
{
    std::vector<double> result(ips.size(), 0.0);
    for (unsigned int i = 0; i < ips.size(); i++) {
        result[i] = std::log10(std::max(ips[i], 0.0) + 1.0);
    }
    return result;
}

void addVectorFeatures(std::vector<double>& features, const std::vector<double>& values)
{
    features.insert(features.end(), values.begin(), values.end());
}

void requireSize(const std::vector<double>& values, unsigned int expected, const std::string& name)
{
    if (values.size() != expected) {
        throw std::runtime_error("Thermal regression input " + name + " has wrong size");
    }
}

}

ThermalRegressionModel::ThermalRegressionModel(const std::string& modelPath)
{
    std::ifstream file(modelPath.c_str());
    if (!file) {
        throw std::runtime_error("Could not open thermal regression model: " + modelPath);
    }

    readToken(file, "THERMAL_REGRESSION_MODEL_V1");
    numCores = readLabeledUInt(file, "num_cores");
    numFeatures = readLabeledUInt(file, "num_features");
    numTargets = readLabeledUInt(file, "num_targets");

    featureNames = readNames(file, "feature_names", numFeatures);
    targetNames = readNames(file, "target_names", numTargets);
    xMean = readVector(file, "x_mean", numFeatures);
    xScale = readVector(file, "x_scale", numFeatures);
    yMean = readVector(file, "y_mean", numTargets);
    yScale = readVector(file, "y_scale", numTargets);

    readToken(file, "coef");
    coefficients.assign(numTargets, std::vector<double>(numFeatures, 0.0));
    for (unsigned int target = 0; target < numTargets; target++) {
        for (unsigned int feature = 0; feature < numFeatures; feature++) {
            file >> coefficients[target][feature];
            if (!file) {
                throw std::runtime_error("Thermal regression model parse error while reading coefficients");
            }
        }
    }

    intercept = readVector(file, "intercept", numTargets);

    if (numTargets != numCores) {
        throw std::runtime_error("Thermal regression model must have one temperature target per core");
    }
}

std::vector<double> ThermalRegressionModel::predictEndTemperatures(const Inputs& inputs) const
{
    validateInputs(inputs);

    const std::vector<double> features = buildFeatures(inputs);
    if (features.size() != numFeatures) {
        throw std::runtime_error("Thermal regression feature count mismatch");
    }

    std::vector<double> scaledFeatures(numFeatures, 0.0);
    for (unsigned int i = 0; i < numFeatures; i++) {
        const double scale = (xScale[i] == 0.0) ? 1.0 : xScale[i];
        scaledFeatures[i] = (features[i] - xMean[i]) / scale;
    }

    std::vector<double> outputs(numTargets, 0.0);
    for (unsigned int target = 0; target < numTargets; target++) {
        double scaledOutput = intercept[target];
        for (unsigned int feature = 0; feature < numFeatures; feature++) {
            scaledOutput += coefficients[target][feature] * scaledFeatures[feature];
        }
        outputs[target] = scaledOutput * yScale[target] + yMean[target];
    }

    std::vector<double> endTemperatures(numCores, 0.0);
    for (unsigned int core = 0; core < numCores; core++) {
        endTemperatures[core] = inputs.startTempsC[core] + outputs[core];
    }
    return endTemperatures;
}

void ThermalRegressionModel::validateInputs(const Inputs& inputs) const
{
    requireSize(inputs.oldFrequenciesMhz, numCores, "oldFrequenciesMhz");
    requireSize(inputs.newFrequenciesMhz, numCores, "newFrequenciesMhz");
    requireSize(inputs.activeCores, numCores, "activeCores");
    requireSize(inputs.startTempsC, numCores, "startTempsC");
    requireSize(inputs.startPowersW, numCores, "startPowersW");
    requireSize(inputs.startUtilizations, numCores, "startUtilizations");
    requireSize(inputs.startCpis, numCores, "startCpis");
    requireSize(inputs.startRelNucaCpis, numCores, "startRelNucaCpis");
    requireSize(inputs.startIps, numCores, "startIps");
}

std::vector<double> ThermalRegressionModel::buildFeatures(const Inputs& inputs) const
{
    const unsigned int n = numCores;

    std::vector<double> oldF(n);
    std::vector<double> newF(n);
    std::vector<double> deltaF(n);
    for (unsigned int i = 0; i < n; i++) {
        oldF[i] = inputs.oldFrequenciesMhz[i] / 1000.0;
        newF[i] = inputs.newFrequenciesMhz[i] / 1000.0;
        deltaF[i] = newF[i] - oldF[i];
    }

    const std::vector<double>& active = inputs.activeCores;
    const std::vector<double>& startTemp = inputs.startTempsC;
    const std::vector<double>& startPower = inputs.startPowersW;
    const std::vector<double>& startUtil = inputs.startUtilizations;

    const std::vector<double> startIpc = cpiToIpc(inputs.startCpis);
    const std::vector<double> logStartIps = safeLogIps(inputs.startIps);

    const double startPeak = *std::max_element(startTemp.begin(), startTemp.end());

    std::vector<double> tempCentered(n);
    std::vector<double> tempToPeak(n);
    for (unsigned int i = 0; i < n; i++) {
        tempCentered[i] = startTemp[i] - 55.0;
        tempToPeak[i] = startPeak - startTemp[i];
    }

    const std::vector<double> tempNeighbor = neighborMean(startTemp);
    const std::vector<double> powerNeighbor = neighborMean(startPower);

    std::vector<double> newF2(n);
    std::vector<double> newF3(n);
    std::vector<double> oldF2(n);
    std::vector<double> oldF3(n);
    std::vector<double> activeNewF(n);
    std::vector<double> activeNewF3(n);
    std::vector<double> utilNewF(n);
    std::vector<double> utilNewF3(n);
    std::vector<double> workloadIndex(n);
    std::vector<double> tempPowerInteraction(n);
    std::vector<double> tempFreqInteraction(n);
    std::vector<double> tempWorkInteraction(n);

    for (unsigned int i = 0; i < n; i++) {
        newF2[i] = newF[i] * newF[i];
        newF3[i] = newF2[i] * newF[i];
        oldF2[i] = oldF[i] * oldF[i];
        oldF3[i] = oldF2[i] * oldF[i];
        activeNewF[i] = active[i] * newF[i];
        activeNewF3[i] = active[i] * newF3[i];
        utilNewF[i] = startUtil[i] * newF[i];
        utilNewF3[i] = startUtil[i] * newF3[i];
        workloadIndex[i] = active[i] * startUtil[i] * newF3[i] * startIpc[i];
        tempPowerInteraction[i] = tempCentered[i] * startPower[i];
        tempFreqInteraction[i] = tempCentered[i] * newF[i];
        tempWorkInteraction[i] = tempCentered[i] * workloadIndex[i];
    }

    std::vector<double> features;
    features.reserve(numFeatures);

    features.push_back(startPeak);
    features.push_back(mean(startTemp));
    features.push_back(*std::max_element(startTemp.begin(), startTemp.end()));
    features.push_back(*std::min_element(startTemp.begin(), startTemp.end()));
    features.push_back(populationStd(startTemp));

    features.push_back(mean(startPower));
    features.push_back(std::accumulate(startPower.begin(), startPower.end(), 0.0));
    features.push_back(*std::max_element(startPower.begin(), startPower.end()));

    features.push_back(mean(startUtil));
    features.push_back(std::accumulate(startUtil.begin(), startUtil.end(), 0.0));

    features.push_back(std::accumulate(active.begin(), active.end(), 0.0));

    features.push_back(mean(oldF));
    features.push_back(mean(newF));
    features.push_back(*std::max_element(newF.begin(), newF.end()));
    features.push_back(*std::min_element(newF.begin(), newF.end()));
    features.push_back(std::accumulate(newF.begin(), newF.end(), 0.0));

    features.push_back(mean(deltaF));
    features.push_back(*std::max_element(deltaF.begin(), deltaF.end()));
    features.push_back(*std::min_element(deltaF.begin(), deltaF.end()));
    features.push_back(std::accumulate(deltaF.begin(), deltaF.end(), 0.0));

    addVectorFeatures(features, oldF);
    addVectorFeatures(features, newF);
    addVectorFeatures(features, deltaF);

    addVectorFeatures(features, newF2);
    addVectorFeatures(features, newF3);
    addVectorFeatures(features, oldF2);
    addVectorFeatures(features, oldF3);

    addVectorFeatures(features, active);
    addVectorFeatures(features, activeNewF);
    addVectorFeatures(features, activeNewF3);

    addVectorFeatures(features, startTemp);
    addVectorFeatures(features, tempCentered);
    addVectorFeatures(features, tempToPeak);
    addVectorFeatures(features, tempNeighbor);

    addVectorFeatures(features, startPower);
    addVectorFeatures(features, powerNeighbor);

    addVectorFeatures(features, startUtil);
    addVectorFeatures(features, utilNewF);
    addVectorFeatures(features, utilNewF3);

    addVectorFeatures(features, startIpc);
    addVectorFeatures(features, inputs.startRelNucaCpis);

    addVectorFeatures(features, logStartIps);

    addVectorFeatures(features, workloadIndex);
    addVectorFeatures(features, tempPowerInteraction);
    addVectorFeatures(features, tempFreqInteraction);
    addVectorFeatures(features, tempWorkInteraction);

    return features;
}
