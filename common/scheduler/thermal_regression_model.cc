#include "thermal_regression_model.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <sstream>

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

std::vector<std::string> readNamesUntilLabel(
    std::ifstream& file,
    const std::string& label,
    const std::string& endLabel)
{
    readToken(file, label);
    std::vector<std::string> names;
    while (true) {
        std::string name;
        file >> name;
        if (!file) {
            throw std::runtime_error("Thermal regression model parse error while reading " + label);
        }
        if (name == endLabel) {
            break;
        }
        names.push_back(name);
    }
    return names;
}

std::vector<std::string> readNamesAfterLabel(std::ifstream& file, const std::string& label, unsigned int count)
{
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

std::string classifyPhase(const std::vector<double>& startUtil)
{
    bool hasTransition = false;
    for (double util : startUtil) {
        if (util >= 0.01 && util <= 0.30) {
            hasTransition = true;
            break;
        }
    }
    if (hasTransition) {
        return "transition";
    }

    std::vector<bool> active(startUtil.size(), false);
    unsigned int totalActive = 0;
    for (unsigned int i = 0; i < startUtil.size(); i++) {
        active[i] = startUtil[i] > 0.30;
        if (active[i]) {
            totalActive++;
        }
    }

    const bool masterActive = !active.empty() && active[0];
    unsigned int workerCount = 0;
    for (unsigned int i = 1; i < active.size(); i++) {
        if (active[i]) {
            workerCount++;
        }
    }

    if (totalActive == 0) {
        return "idle";
    }
    if (masterActive && workerCount == 0) {
        return "master_only";
    }
    if (!masterActive && workerCount > 0) {
        return "workers_only";
    }
    if (masterActive && workerCount > 0) {
        return "full_parallel";
    }
    return "mixed";
}

bool hasPhase(const std::vector<std::string>& phaseNames, const std::string& phase)
{
    return std::find(phaseNames.begin(), phaseNames.end(), phase) != phaseNames.end();
}

std::string classifyPhaseForModel(
    const std::vector<double>& startUtil,
    const std::vector<std::string>& phaseNames)
{
    if (phaseNames.size() == 2
        && phaseNames[0] == "master_only"
        && phaseNames[1] == "workers_only") {
        unsigned int workerCount = 0;
        for (unsigned int i = 1; i < startUtil.size(); i++) {
            if (startUtil[i] > 0.30) {
                workerCount++;
            }
        }
        return workerCount > 0 ? "workers_only" : "master_only";
    }

    const std::string phase = classifyPhase(startUtil);
    if (hasPhase(phaseNames, phase)) {
        return phase;
    }

    if (hasPhase(phaseNames, "workers_only")) {
        for (unsigned int i = 1; i < startUtil.size(); i++) {
            if (startUtil[i] > 0.30) {
                return "workers_only";
            }
        }
    }

    if (hasPhase(phaseNames, "master_only")) {
        return "master_only";
    }

    return phaseNames.empty() ? phase : phaseNames[0];
}

std::string joinDoublesForLog(const std::vector<double>& values)
{
    std::ostringstream out;
    for (unsigned int i = 0; i < values.size(); i++) {
        if (i != 0) {
            out << ";";
        }
        out << values[i];
    }
    return out.str();
}

}

ThermalRegressionModel::ThermalRegressionModel(const std::string& modelPath)
{
    try {
        std::ifstream file(modelPath.c_str());
        if (!file) {
            throw std::runtime_error("Could not open thermal regression model: " + modelPath);
        }

        std::string header;
        file >> header;
        if (!file) {
            throw std::runtime_error("Thermal regression model parse error: empty model file");
        }

        if (header == "THERMAL_REGRESSION_MODEL_V1") {
            loadV1(file);
        } else if (header == "THERMAL_REGRESSION_MODEL_V2_PHASE_AWARE") {
            loadV2PhaseAware(file);
        } else {
            throw std::runtime_error("Unsupported thermal regression model header: " + header);
        }

        validateModel();
        enabled = true;
    } catch (const std::exception& e) {
        disable(e.what());
    }
}

void ThermalRegressionModel::loadV1(std::ifstream& file)
{
    phaseAware = false;
    usesPhaseInteractions = false;
    phaseNames.clear();

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
        throw std::runtime_error("Thermal regression V1 model must have one temperature target per core");
    }
}

void ThermalRegressionModel::loadV2PhaseAware(std::ifstream& file)
{
    phaseAware = true;
    numCores = readLabeledUInt(file, "num_cores");
    numFeatures = readLabeledUInt(file, "num_features");
    numTargets = readLabeledUInt(file, "num_targets");
    usesPhaseInteractions = readLabeledUInt(file, "uses_phase_interactions") != 0;

    phaseNames = readNamesUntilLabel(file, "phase_names", "feature_names");
    featureNames = readNamesAfterLabel(file, "feature_names", numFeatures);
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

    if (phaseNames.empty()) {
        throw std::runtime_error("Thermal regression V2 model must contain at least one phase name");
    }
}

void ThermalRegressionModel::validateModel() const
{
    if (numCores != 4) {
        throw std::runtime_error("Thermal regression model num_cores must be 4");
    }
    if (numTargets != 4) {
        throw std::runtime_error("Thermal regression model num_targets must be 4");
    }
    if (featureNames.size() != numFeatures) {
        throw std::runtime_error("Thermal regression model feature_names length does not match num_features");
    }
    if (targetNames.size() != numTargets) {
        throw std::runtime_error("Thermal regression model target_names length does not match num_targets");
    }
    if (xMean.size() != numFeatures) {
        throw std::runtime_error("Thermal regression model x_mean length does not match num_features");
    }
    if (xScale.size() != numFeatures) {
        throw std::runtime_error("Thermal regression model x_scale length does not match num_features");
    }
    if (yMean.size() != numTargets) {
        throw std::runtime_error("Thermal regression model y_mean length does not match num_targets");
    }
    if (yScale.size() != numTargets) {
        throw std::runtime_error("Thermal regression model y_scale length does not match num_targets");
    }
    if (coefficients.size() != numTargets) {
        throw std::runtime_error("Thermal regression model coefficient row count does not match num_targets");
    }
    for (unsigned int target = 0; target < coefficients.size(); target++) {
        if (coefficients[target].size() != numFeatures) {
            throw std::runtime_error("Thermal regression model coefficient column count does not match num_features");
        }
    }
    if (intercept.size() != numTargets) {
        throw std::runtime_error("Thermal regression model intercept length does not match num_targets");
    }
    if (phaseAware && phaseNames.empty()) {
        throw std::runtime_error("Thermal regression phase-aware model has no phase_names");
    }
}

void ThermalRegressionModel::disable(const std::string& reason)
{
    enabled = false;
    phaseAware = false;
    usesPhaseInteractions = false;
    numCores = 0;
    numFeatures = 0;
    numTargets = 0;
    phaseNames.clear();
    featureNames.clear();
    targetNames.clear();
    xMean.clear();
    xScale.clear();
    yMean.clear();
    yScale.clear();
    coefficients.clear();
    intercept.clear();

    std::cerr << "[ThermalRegressionModel]: warning: disabling ML thermal predictor: "
              << reason << std::endl;
}

std::vector<double> ThermalRegressionModel::predictEndTemperatures(const Inputs& inputs)
{
    if (!enabled) {
        return inputs.startTempsC;
    }

    std::string classifiedPhase;
    std::vector<double> features;
    try {
        validateInputs(inputs);
        features = buildFeatures(inputs, &classifiedPhase);
        if (features.size() != numFeatures) {
            std::ostringstream out;
            out << "Thermal regression feature count mismatch: constructed "
                << features.size() << ", expected " << numFeatures;
            throw std::runtime_error(out.str());
        }
    } catch (const std::exception& e) {
        disable(e.what());
        return inputs.startTempsC;
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

    if (debug) {
        std::cout << "[ThermalRegressionModel]: phase="
                  << (classifiedPhase.empty() ? "none" : classifiedPhase)
                  << " feature_count=" << features.size()
                  << " predicted_delta_temps_c=" << joinDoublesForLog(outputs)
                  << " predicted_end_temps_c=" << joinDoublesForLog(endTemperatures)
                  << std::endl;
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

std::vector<double> ThermalRegressionModel::buildFeatures(const Inputs& inputs, std::string* classifiedPhase) const
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

    const double startPeak = (inputs.startPeakTempC >= 0.0)
        ? inputs.startPeakTempC
        : *std::max_element(startTemp.begin(), startTemp.end());

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

    if (phaseAware) {
        const std::vector<double> baseFeatures(features);
        const std::string phase = classifyPhaseForModel(startUtil, phaseNames);
        if (classifiedPhase != nullptr) {
            *classifiedPhase = phase;
        }

        for (const std::string& phaseName : phaseNames) {
            features.push_back(phase == phaseName ? 1.0 : 0.0);
        }

        if (usesPhaseInteractions) {
            for (const std::string& phaseName : phaseNames) {
                const double flag = (phase == phaseName) ? 1.0 : 0.0;
                for (double value : baseFeatures) {
                    features.push_back(flag * value);
                }
            }
        }
    } else if (classifiedPhase != nullptr) {
        *classifiedPhase = "";
    }

    return features;
}
