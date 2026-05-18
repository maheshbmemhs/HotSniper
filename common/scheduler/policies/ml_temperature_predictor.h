#ifndef __ML_TEMPERATURE_PREDICTOR_H
#define __ML_TEMPERATURE_PREDICTOR_H

#include <string>
#include <vector>

class MLTemperaturePredictor {
public:
    MLTemperaturePredictor();
    explicit MLTemperaturePredictor(const std::string &model_path, bool debug_enabled = false);

    bool isLoaded() const;

    double predictNextTemp(
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
        bool emit_debug = true,
        const std::vector<double> *expected_ips = nullptr,
        const std::vector<double> *expected_cpi = nullptr,
        const std::vector<double> *expected_power_w = nullptr,
        const std::vector<double> *expected_utilization = nullptr,
        const std::vector<bool> *expected_active_cores = nullptr
    ) const;

private:
    struct Node {
        enum Type {
            Leaf,
            NumericLessEqual,
            CategoricalEqual
        };

        Type type;
        int split_feature;
        double threshold;
        int left;
        int right;
        double leaf_value;
        std::vector<int> cat_values;

        Node();
    };

    typedef std::vector<Node> Tree;

    bool load(const std::string &model_path);
    bool buildFeatures(
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
    ) const;
    double predictDelta(const std::vector<double> &features) const;
    double predictTree(const Tree &tree, const std::vector<double> &features) const;
    int classifyPhase(const std::vector<double> &utilization) const;

    bool loaded;
    bool debug_enabled;
    std::vector<std::string> phase_names;
    std::vector<std::string> feature_names;
    std::vector<int> categorical_feature_indices;
    std::vector<Tree> trees;
};

#endif
