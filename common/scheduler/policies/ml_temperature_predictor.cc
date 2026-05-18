#include "ml_temperature_predictor.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace {

const char *kExpectedHeader = "THERMAL_LGBM_SINGLE_CORE_MODEL_V1";

double safeValue(const std::vector<double> &values, size_t idx, double fallback = 0.0)
{
    if (idx >= values.size()) {
        return fallback;
    }
    double value = values[idx];
    return std::isfinite(value) ? value : fallback;
}

double safeMean(const std::vector<double> &values)
{
    if (values.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (double value : values) {
        sum += std::isfinite(value) ? value : 0.0;
    }
    return sum / static_cast<double>(values.size());
}

double safeSum(const std::vector<double> &values)
{
    double sum = 0.0;
    for (double value : values) {
        sum += std::isfinite(value) ? value : 0.0;
    }
    return sum;
}

double safeMax(const std::vector<double> &values)
{
    if (values.empty()) {
        return 0.0;
    }
    double result = -std::numeric_limits<double>::infinity();
    for (double value : values) {
        if (std::isfinite(value)) {
            result = std::max(result, value);
        }
    }
    return std::isfinite(result) ? result : 0.0;
}

double safeMin(const std::vector<double> &values)
{
    if (values.empty()) {
        return 0.0;
    }
    double result = std::numeric_limits<double>::infinity();
    for (double value : values) {
        if (std::isfinite(value)) {
            result = std::min(result, value);
        }
    }
    return std::isfinite(result) ? result : 0.0;
}

std::vector<double> withoutIndex(const std::vector<double> &values, size_t idx)
{
    std::vector<double> result;
    result.reserve(values.size() > 0 ? values.size() - 1 : 0);
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != idx) {
            result.push_back(values[i]);
        }
    }
    return result;
}

double cpiToIpc(double cpi)
{
    if (std::isfinite(cpi) && cpi > 0.0 && cpi < 100.0) {
        return 1.0 / std::max(cpi, 1e-6);
    }
    return 0.0;
}

double logIps(double ips)
{
    if (!std::isfinite(ips) || ips < 0.0) {
        ips = 0.0;
    }
    return std::log10(ips + 1.0);
}

void expectToken(std::istream &in, const std::string &expected)
{
    std::string token;
    if (!(in >> token) || token != expected) {
        throw std::runtime_error("expected token '" + expected + "'");
    }
}

} // namespace

MLTemperaturePredictor::Node::Node()
    : type(Leaf)
    , split_feature(-1)
    , threshold(0.0)
    , left(-1)
    , right(-1)
    , leaf_value(0.0)
{
}

MLTemperaturePredictor::MLTemperaturePredictor()
    : loaded(false)
    , debug_enabled(false)
{
}

MLTemperaturePredictor::MLTemperaturePredictor(const std::string &model_path, bool debug_enabled_)
    : loaded(false)
    , debug_enabled(debug_enabled_)
{
    if (model_path.empty()) {
        std::cerr << "[MLTemperaturePredictor] Warning: no model path configured; using fallback temperatures" << std::endl;
        return;
    }

    loaded = load(model_path);
    if (loaded) {
        std::cout << "[MLTemperaturePredictor] loaded model from " << model_path
                  << " (" << trees.size() << " trees, "
                  << feature_names.size() << " features)" << std::endl;
    } else {
        std::cerr << "[MLTemperaturePredictor] Warning: failed to load model from "
                  << model_path << "; using fallback temperatures" << std::endl;
    }
}

bool MLTemperaturePredictor::isLoaded() const
{
    return loaded;
}

bool MLTemperaturePredictor::load(const std::string &model_path)
{
    try {
        std::ifstream file(model_path.c_str());
        if (!file) {
            return false;
        }

        std::string header;
        std::getline(file, header);
        if (header != kExpectedHeader) {
            std::cerr << "[MLTemperaturePredictor] Warning: unsupported model header: "
                      << header << std::endl;
            return false;
        }

        std::string token;
        int num_features = 0;
        expectToken(file, "num_features");
        file >> num_features;

        expectToken(file, "target_name");
        file >> token;
        if (token != "delta_temp") {
            std::cerr << "[MLTemperaturePredictor] Warning: unsupported target_name: "
                      << token << std::endl;
            return false;
        }

        expectToken(file, "prediction_kind");
        file >> token;
        if (token != "end_temp_equals_start_temp_plus_predicted_delta") {
            std::cerr << "[MLTemperaturePredictor] Warning: unsupported prediction_kind: "
                      << token << std::endl;
            return false;
        }

        expectToken(file, "phase_names");
        phase_names.clear();
        while (file >> token) {
            if (token == "feature_names") {
                break;
            }
            phase_names.push_back(token);
        }
        if (token != "feature_names") {
            return false;
        }

        feature_names.clear();
        for (int i = 0; i < num_features; ++i) {
            if (!(file >> token)) {
                return false;
            }
            feature_names.push_back(token);
        }

        expectToken(file, "categorical_feature_indices");
        std::string indices_line;
        std::getline(file, indices_line);
        if (indices_line.empty()) {
            std::getline(file, indices_line);
        }
        std::istringstream indices(indices_line);
        int cat_idx = 0;
        while (indices >> cat_idx) {
            categorical_feature_indices.push_back(cat_idx);
        }

        int num_trees = 0;
        expectToken(file, "num_trees");
        file >> num_trees;
        if (num_trees <= 0) {
            return false;
        }

        trees.clear();
        trees.reserve(static_cast<size_t>(num_trees));
        for (int tree_id = 0; tree_id < num_trees; ++tree_id) {
            std::string tree_token;
            int parsed_tree_id = -1;
            std::string nodes_token;
            int num_nodes = 0;
            file >> tree_token >> parsed_tree_id >> nodes_token >> num_nodes;
            if (tree_token != "tree" || nodes_token != "num_nodes" || num_nodes <= 0) {
                return false;
            }

            Tree tree;
            tree.reserve(static_cast<size_t>(num_nodes));
            for (int node_id = 0; node_id < num_nodes; ++node_id) {
                std::string kind;
                file >> kind;

                Node node;
                if (kind == "L") {
                    node.type = Node::Leaf;
                    file >> node.leaf_value;
                } else if (kind == "N") {
                    node.type = Node::NumericLessEqual;
                    file >> node.split_feature >> node.threshold >> node.left >> node.right;
                } else if (kind == "C") {
                    node.type = Node::CategoricalEqual;
                    std::string cat_values;
                    file >> node.split_feature >> cat_values >> node.left >> node.right;
                    std::istringstream cat_stream(cat_values);
                    std::string cat_token;
                    while (std::getline(cat_stream, cat_token, ',')) {
                        if (!cat_token.empty()) {
                            node.cat_values.push_back(std::stoi(cat_token));
                        }
                    }
                } else {
                    return false;
                }

                tree.push_back(node);
            }
            trees.push_back(tree);
        }
    } catch (const std::exception &e) {
        std::cerr << "[MLTemperaturePredictor] Warning: parse error: "
                  << e.what() << std::endl;
        phase_names.clear();
        feature_names.clear();
        categorical_feature_indices.clear();
        trees.clear();
        return false;
    }

    return !feature_names.empty() && !trees.empty();
}

double MLTemperaturePredictor::predictNextTemp(
    unsigned int core_id,
    const std::vector<double> &current_temps_c,
    const std::vector<double> &current_freqs_ghz,
    const std::vector<double> &candidate_freqs_ghz,
    const std::vector<double> &ips,
    const std::vector<double> &utilization,
    const std::vector<double> &cpi,
    const std::vector<double> &rel_nuca_cpi,
    const std::vector<double> &power_w,
    const std::vector<bool> &active_cores,
    bool emit_debug,
    const std::vector<double> *expected_ips,
    const std::vector<double> *expected_cpi,
    const std::vector<double> *expected_power_w,
    const std::vector<double> *expected_utilization,
    const std::vector<bool> *expected_active_cores
) const {
    if (!loaded) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    std::vector<double> features;
    if (!buildFeatures(
            core_id,
            current_temps_c,
            current_freqs_ghz,
            candidate_freqs_ghz,
            ips,
            utilization,
            cpi,
            rel_nuca_cpi,
            power_w,
            active_cores,
            expected_ips,
            expected_cpi,
            expected_power_w,
            expected_utilization,
            expected_active_cores,
            features)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double current_temp = safeValue(current_temps_c, core_id, 0.0);
    const double predicted_delta = predictDelta(features);
    const double predicted_end_temp = current_temp + predicted_delta;

    if (debug_enabled && emit_debug) {
        std::cout << "[MLTemperaturePredictor] core=" << core_id
                  << " cur_temp=" << current_temp
                  << " cur_freq=" << safeValue(current_freqs_ghz, core_id, 0.0)
                  << " cand_freq=" << safeValue(candidate_freqs_ghz, core_id, 0.0)
                  << " pred_temp=" << predicted_end_temp << std::endl;
    }

    return predicted_end_temp;
}

bool MLTemperaturePredictor::buildFeatures(
    unsigned int core_id,
    const std::vector<double> &current_temps_c,
    const std::vector<double> &current_freqs_ghz,
    const std::vector<double> &candidate_freqs_ghz,
    const std::vector<double> &ips,
    const std::vector<double> &utilization,
    const std::vector<double> &cpi,
    const std::vector<double> &rel_nuca_cpi,
    const std::vector<double> &power_w,
    const std::vector<bool> &active_cores,
    const std::vector<double> *expected_ips,
    const std::vector<double> *expected_cpi,
    const std::vector<double> *expected_power_w,
    const std::vector<double> *expected_utilization,
    const std::vector<bool> *expected_active_cores,
    std::vector<double> &features
) const {
    if (core_id >= current_temps_c.size()) {
        return false;
    }

    const size_t num_cores = current_temps_c.size();
    std::vector<double> old_f(num_cores, 0.0);
    std::vector<double> new_f(num_cores, 0.0);
    std::vector<double> delta_f(num_cores, 0.0);
    std::vector<double> active(num_cores, 0.0);
    std::vector<double> expected_active(num_cores, 0.0);
    std::vector<double> start_ipc(num_cores, 0.0);
    std::vector<double> expected_ipc_values(num_cores, 0.0);
    std::vector<double> log_start_ips(num_cores, 0.0);
    std::vector<double> log_expected_ips(num_cores, 0.0);
    std::vector<double> busy(num_cores, 0.0);
    std::vector<double> expected_busy(num_cores, 0.0);
    const std::vector<double> &expected_ips_values = expected_ips ? *expected_ips : ips;
    const std::vector<double> &expected_cpi_values = expected_cpi ? *expected_cpi : cpi;
    const std::vector<double> &expected_power_values = expected_power_w ? *expected_power_w : power_w;
    const std::vector<double> &expected_utilization_values = expected_utilization ? *expected_utilization : utilization;

    for (size_t i = 0; i < num_cores; ++i) {
        old_f[i] = safeValue(current_freqs_ghz, i, 0.0);
        new_f[i] = safeValue(candidate_freqs_ghz, i, old_f[i]);
        delta_f[i] = new_f[i] - old_f[i];
        active[i] = (i < active_cores.size() && active_cores[i]) ? 1.0 : 0.0;
        expected_active[i] =
            (expected_active_cores && i < expected_active_cores->size() && (*expected_active_cores)[i])
                ? 1.0
                : (expected_active_cores ? 0.0 : active[i]);
        start_ipc[i] = cpiToIpc(safeValue(cpi, i, 0.0));
        expected_ipc_values[i] = cpiToIpc(safeValue(expected_cpi_values, i, safeValue(cpi, i, 0.0)));
        log_start_ips[i] = logIps(safeValue(ips, i, 0.0));
        log_expected_ips[i] = logIps(safeValue(expected_ips_values, i, safeValue(ips, i, 0.0)));
        busy[i] = safeValue(utilization, i, 0.0) > 0.30 ? 1.0 : 0.0;
        expected_busy[i] =
            safeValue(expected_utilization_values, i, safeValue(utilization, i, 0.0)) > 0.30 ? 1.0 : 0.0;
    }

    const size_t i = core_id;
    const std::vector<double> neighbor_temps = withoutIndex(current_temps_c, i);
    const std::vector<double> neighbor_power = withoutIndex(power_w, i);
    const std::vector<double> neighbor_expected_power = withoutIndex(expected_power_values, i);
    const std::vector<double> neighbor_util = withoutIndex(utilization, i);
    const std::vector<double> neighbor_busy = withoutIndex(busy, i);
    const std::vector<double> neighbor_expected_util = withoutIndex(expected_utilization_values, i);
    const std::vector<double> neighbor_expected_busy = withoutIndex(expected_busy, i);

    const double start_peak = safeMax(current_temps_c);
    const double mean_start_temp = safeMean(current_temps_c);
    const double workload_index =
        active[i] * safeValue(utilization, i, 0.0) * std::pow(new_f[i], 3.0) * start_ipc[i];
    const double expected_workload_index =
        active[i] * safeValue(utilization, i, 0.0) * std::pow(new_f[i], 3.0) * expected_ipc_values[i];
    const double expected_runtime_index =
        expected_active[i] *
        safeValue(expected_utilization_values, i, safeValue(utilization, i, 0.0)) *
        std::pow(new_f[i], 3.0) *
        expected_ipc_values[i];

    std::unordered_map<std::string, double> values;
    values["start_peak_temp"] = start_peak;
    values["mean_start_temp"] = mean_start_temp;
    values["max_start_temp"] = safeMax(current_temps_c);
    values["min_start_temp"] = safeMin(current_temps_c);
    values["temp_spread"] = safeMax(current_temps_c) - safeMin(current_temps_c);

    values["sum_start_power"] = safeSum(power_w);
    values["max_start_power"] = safeMax(power_w);
    values["mean_start_power"] = safeMean(power_w);
    values["sum_expected_power"] = safeSum(expected_power_values);
    values["max_expected_power"] = safeMax(expected_power_values);
    values["mean_expected_power"] = safeMean(expected_power_values);

    values["num_active_flags"] = safeSum(active);
    values["num_busy_cores"] = safeSum(busy);
    values["num_expected_active_flags"] = safeSum(expected_active);
    values["num_expected_busy_cores"] = safeSum(expected_busy);

    values["max_new_freq"] = safeMax(new_f);
    values["mean_new_freq"] = safeMean(new_f);
    values["sum_new_freq"] = safeSum(new_f);

    values["max_delta_freq"] = safeMax(delta_f);
    values["min_delta_freq"] = safeMin(delta_f);
    values["sum_delta_freq"] = safeSum(delta_f);

    values["core_id"] = static_cast<double>(core_id);
    values["old_freq"] = old_f[i];
    values["new_freq"] = new_f[i];
    values["delta_freq"] = delta_f[i];

    values["active"] = active[i];
    values["is_busy"] = busy[i];
    values["expected_active"] = expected_active[i];
    values["expected_active_delta"] = expected_active[i] - active[i];
    values["expected_is_busy"] = expected_busy[i];

    values["start_temp"] = safeValue(current_temps_c, i, 0.0);
    values["temp_vs_peak"] = safeValue(current_temps_c, i, 0.0) - start_peak;
    values["temp_vs_mean"] = safeValue(current_temps_c, i, 0.0) - mean_start_temp;

    values["start_power"] = safeValue(power_w, i, 0.0);
    values["expected_power"] = safeValue(expected_power_values, i, safeValue(power_w, i, 0.0));
    values["expected_power_delta"] = values["expected_power"] - values["start_power"];
    values["start_util"] = safeValue(utilization, i, 0.0);
    values["expected_util"] = safeValue(expected_utilization_values, i, safeValue(utilization, i, 0.0));
    values["expected_util_delta"] = values["expected_util"] - values["start_util"];
    values["start_ipc"] = start_ipc[i];
    values["expected_ipc"] = expected_ipc_values[i];
    values["expected_ipc_delta"] = expected_ipc_values[i] - start_ipc[i];
    values["expected_cpi"] = safeValue(expected_cpi_values, i, safeValue(cpi, i, 0.0));
    values["start_rel_nuca_cpi"] = safeValue(rel_nuca_cpi, i, 0.0);
    values["log_start_ips"] = log_start_ips[i];
    values["log_expected_ips"] = log_expected_ips[i];

    values["workload_index"] = workload_index;
    values["expected_workload_index"] = expected_workload_index;
    values["expected_runtime_index"] = expected_runtime_index;

    values["neighbor_temp_mean"] = safeMean(neighbor_temps);
    values["neighbor_temp_max"] = safeMax(neighbor_temps);
    values["neighbor_temp_min"] = safeMin(neighbor_temps);
    values["neighbor_power_mean"] = safeMean(neighbor_power);
    values["neighbor_power_max"] = safeMax(neighbor_power);
    values["neighbor_expected_power_mean"] = safeMean(neighbor_expected_power);
    values["neighbor_expected_power_max"] = safeMax(neighbor_expected_power);
    values["neighbor_util_mean"] = safeMean(neighbor_util);
    values["neighbor_busy_count"] = safeSum(neighbor_busy);
    values["neighbor_expected_util_mean"] = safeMean(neighbor_expected_util);
    values["neighbor_expected_busy_count"] = safeSum(neighbor_expected_busy);

    values["phase_id"] = static_cast<double>(classifyPhase(utilization));

    features.assign(feature_names.size(), 0.0);
    for (size_t feature_idx = 0; feature_idx < feature_names.size(); ++feature_idx) {
        const auto found = values.find(feature_names[feature_idx]);
        if (found != values.end() && std::isfinite(found->second)) {
            features[feature_idx] = found->second;
        }
    }

    return true;
}

double MLTemperaturePredictor::predictDelta(const std::vector<double> &features) const
{
    double prediction = 0.0;
    for (const Tree &tree : trees) {
        prediction += predictTree(tree, features);
    }
    return prediction;
}

double MLTemperaturePredictor::predictTree(const Tree &tree, const std::vector<double> &features) const
{
    if (tree.empty()) {
        return 0.0;
    }

    int node_idx = 0;
    for (size_t depth_guard = 0; depth_guard <= tree.size(); ++depth_guard) {
        if (node_idx < 0 || node_idx >= static_cast<int>(tree.size())) {
            return 0.0;
        }

        const Node &node = tree[static_cast<size_t>(node_idx)];
        if (node.type == Node::Leaf) {
            return node.leaf_value;
        }

        double value = 0.0;
        if (node.split_feature >= 0 && node.split_feature < static_cast<int>(features.size())) {
            value = features[static_cast<size_t>(node.split_feature)];
        }

        if (node.type == Node::CategoricalEqual) {
            const int category = static_cast<int>(std::floor(value + 0.5));
            const bool match = std::find(
                node.cat_values.begin(),
                node.cat_values.end(),
                category) != node.cat_values.end();
            node_idx = match ? node.left : node.right;
        } else {
            node_idx = value <= node.threshold ? node.left : node.right;
        }
    }

    return 0.0;
}

int MLTemperaturePredictor::classifyPhase(const std::vector<double> &utilization) const
{
    for (size_t i = 1; i < utilization.size(); ++i) {
        if (safeValue(utilization, i, 0.0) > 0.30) {
            return 1;
        }
    }
    return 0;
}
