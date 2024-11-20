#ifndef NNET_GARNET_H_
#define NNET_GARNET_H_

#include "hls_math.h"
#include "hls_stream.h"
#include "nnet_common.h"

namespace nnet {
namespace garnet_utils {

template <class CONFIG_T>
inline typename std::enable_if<std::is_class<typename CONFIG_T::distance_t>::value>::type
initialize_edge_weights_table(typename CONFIG_T::edge_weight_t edge_weights_table[]) {
    typedef ap_uint<CONFIG_T::distance_width> index_t;

    unsigned const table_size = (1 << CONFIG_T::distance_width);

    index_t index;
    typename CONFIG_T::distance_t distance;

    // edge_weight_t is ap_ufixed with 0 iwidth -> let index 0 be a saturated version of 1
    edge_weights_table[0] = ap_ufixed<CONFIG_T::edge_weight_t::width, 0, AP_TRN, AP_SAT>(1.);

    for (unsigned iw = 1; iw < table_size; ++iw) {
        index = iw;
        distance.range(CONFIG_T::distance_width - 1, 0) = index.range(CONFIG_T::distance_width - 1, 0);
        edge_weights_table[iw] = hls::exp(-distance * distance);
    }
}

template <class CONFIG_T>
inline typename std::enable_if<not std::is_class<typename CONFIG_T::distance_t>::value>::type
initialize_edge_weights_table(typename CONFIG_T::edge_weight_t edge_weights_table[]) {
    unsigned const table_size = (1 << CONFIG_T::distance_width);
    double const step = 64. / table_size;

    typename CONFIG_T::distance_t v = -32.;
    for (unsigned iw = 0; iw < table_size; ++iw) {
        edge_weights_table[iw] = std::exp(-v * v);
        v += step;
    }
}

template <class CONFIG_T>
inline typename std::enable_if<std::is_class<typename CONFIG_T::distance_t>::value, typename CONFIG_T::edge_weight_t>::type
get_edge_weight(typename CONFIG_T::distance_t distance, typename CONFIG_T::edge_weight_t edge_weights_table[]) {
    typedef ap_uint<CONFIG_T::distance_width> index_t;

    index_t index(distance.range(CONFIG_T::distance_width - 1, 0));

    return edge_weights_table[index];
}

template <class CONFIG_T>
inline
    typename std::enable_if<not std::is_class<typename CONFIG_T::distance_t>::value, typename CONFIG_T::edge_weight_t>::type
    get_edge_weight(typename CONFIG_T::distance_t distance, typename CONFIG_T::edge_weight_t edge_weights_table[]) {
    unsigned const table_size = (1 << CONFIG_T::distance_width);
    double const step = 64. / table_size;

    int index = (distance + 32.) / step;
    if (index < 0)
        index = 0;
    else if (index >= table_size)
        index = table_size - 1;

    return edge_weights_table[index];
}

template <class CONFIG_T> typename CONFIG_T::edge_weight_t compute_edge_weight(typename CONFIG_T::distance_t distance) {
    if (CONFIG_T::is_stack) {
        #pragma HLS INLINE OFF
    }
#ifdef __SYNTHESIS__
    typename CONFIG_T::edge_weight_t edge_weights_table[1 << CONFIG_T::distance_width];
    // unsigned const reshape_factor = CONFIG_T::n_aggregators * CONFIG_T::n_in_features * (CONFIG_T::n_vertices /
    // CONFIG_T::reuse_factor);
    // #pragma HLS ARRAY_RESHAPE variable=edge_weights_table cyclic factor=reshape_factor dim=1
    bool initialized = false;
#else
    static typename CONFIG_T::edge_weight_t edge_weights_table[1 << CONFIG_T::distance_width];
    static bool initialized = false;
#endif
    if (not initialized) {
        initialize_edge_weights_table<CONFIG_T>(edge_weights_table);
        initialized = true;
    }

    return get_edge_weight<CONFIG_T>(distance, edge_weights_table);
}

template <class dividend_T, class exponent_T>
inline typename std::enable_if<std::is_class<dividend_T>::value, dividend_T>::type
normalize_log2(dividend_T dividend, exponent_T exponent) {
    #pragma HLS INLINE
    return dividend >> exponent;
}

template <class dividend_T, class exponent_T>
inline typename std::enable_if<not std::is_class<dividend_T>::value, dividend_T>::type
normalize_log2(dividend_T dividend, exponent_T exponent) {
    #pragma HLS INLINE
    return dividend / std::pow(2., exponent);
}

template <class CONFIG_T, class E = typename CONFIG_T::edge_weight_t> struct Means {
    typedef E edge_weight_t;

    edge_weight_t edge_weight_mean[CONFIG_T::n_aggregators];
    typename CONFIG_T::aggr_t weighted_feature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_features];

    Means() {
        #pragma HLS INLINE
        #pragma HLS ARRAY_PARTITION variable=edge_weight_mean complete
        #pragma HLS ARRAY_PARTITION variable=weighted_feature_mean complete
        #pragma HLS UNROLL region

    Aggregators:
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
            edge_weight_mean[ia] = 0.;

        InFeatures:
            for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
                unsigned const iax = ia * CONFIG_T::n_in_features + ix;
                weighted_feature_mean[iax] = 0.;
            }
        }
    }

    void set_weight(unsigned, edge_weight_t const &) {
        #pragma HLS INLINE
    }

    void add_means_normalized(Means<CONFIG_T, edge_weight_t> const &local) {
        #pragma HLS INLINE
        // Always called within a pipelined region - no UNROLL needed

        unsigned const log2_unroll_factor = CONFIG_T::n_vertices_width - CONFIG_T::log2_reuse_factor;

    Aggregators:
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
            edge_weight_mean[ia] += normalize_log2(local.edge_weight_mean[ia], log2_unroll_factor);

        InFeatures:
            for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
                unsigned const iax = ia * CONFIG_T::n_in_features + ix;
                weighted_feature_mean[iax] += normalize_log2(local.weighted_feature_mean[iax], log2_unroll_factor);
            }
        }
    }

    template <class nvtx_T, class arrays_T, class T = CONFIG_T>
    typename std::enable_if<T::mean_by_nvert>::type set_means_normalized(nvtx_T const nvtx, arrays_T const &accum) {
        #pragma HLS INLINE
        #pragma HLS UNROLL region

        // accum comes divided by unroll factor
        typename T::norm_t nvtx_norm = (T::n_vertices / T::reuse_factor) / nvtx;

    Aggregators:
        for (unsigned ia = 0; ia < T::n_aggregators; ++ia) {
            edge_weight_mean[ia] = accum.edge_weight_mean[ia] * nvtx_norm;

        InFeatures:
            for (unsigned ix = 0; ix < T::n_in_features; ++ix) {
                unsigned const iax = ia * T::n_in_features + ix;

                weighted_feature_mean[iax] = accum.weighted_feature_mean[iax] * nvtx_norm;
            }
        }
    }

    template <class nvtx_T, class arrays_T, class T = CONFIG_T>
    typename std::enable_if<not T::mean_by_nvert>::type set_means_normalized(nvtx_T const nvtx, arrays_T const &accum) {
        #pragma HLS INLINE
        #pragma HLS UNROLL region

    Aggregators:
        for (unsigned ia = 0; ia < T::n_aggregators; ++ia) {

            edge_weight_mean[ia] = normalize_log2(accum.edge_weight_mean[ia], T::log2_reuse_factor);

        InFeatures:
            for (unsigned ix = 0; ix < T::n_in_features; ++ix) {
                unsigned const iax = ia * T::n_in_features + ix;

                weighted_feature_mean[iax] = normalize_log2(accum.weighted_feature_mean[iax], T::log2_reuse_factor);
            }
        }
    }
};

template <class CONFIG_T, class E = typename CONFIG_T::edge_weight_t> struct WeightsAndMeans : public Means<CONFIG_T, E> {
    typedef E edge_weight_t;

    edge_weight_t edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators];

    WeightsAndMeans() : Means<CONFIG_T, E>() {
        #pragma HLS INLINE
        unsigned const reshape_factor = CONFIG_T::n_aggregators * (CONFIG_T::n_vertices / CONFIG_T::reuse_factor);
        #pragma HLS ARRAY_PARTITION variable=edge_weights cyclic factor=reshape_factor
    }

    void set_weight(unsigned iva, edge_weight_t const &weight) {
        #pragma HLS INLINE
        edge_weights[iva] = weight;
    }
};

template <class CONFIG_T, class nvtx_T, class Enable = void> struct OutputBiasNormalizer;

template <class CONFIG_T, class nvtx_T>
struct OutputBiasNormalizer<CONFIG_T, nvtx_T, typename std::enable_if<CONFIG_T::mean_by_nvert>::type> {
    typedef typename CONFIG_T::output_transform_biases_t biases_t;

    biases_t const (&output_biases)[CONFIG_T::n_out_features];

    OutputBiasNormalizer(nvtx_T const) : output_biases{CONFIG_T::output_transform_biases} {
        #pragma HLS INLINE
    }
};

template <class CONFIG_T, class nvtx_T>
struct OutputBiasNormalizer<CONFIG_T, nvtx_T, typename std::enable_if<not CONFIG_T::mean_by_nvert>::type> {
    typedef typename CONFIG_T::output_transform_biases_t biases_t;

    biases_t output_biases[CONFIG_T::n_out_features];

    OutputBiasNormalizer(nvtx_T const nvtx) {
        #pragma HLS ARRAY_PARTITION variable=output_biases complete
        #pragma HLS UNROLL region

        // Cannot add a loop label here due to a Vivado HLS bug, apparently
        for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
            typename CONFIG_T::aggr_t bias = CONFIG_T::output_transform_biases[io];
            bias *= nvtx;
            output_biases[io] = normalize_log2(bias, CONFIG_T::n_vertices_width);
        }
    }
};

template <class CONFIG_T, class data_T> struct InputDataGetter {
    typedef data_T data_t;

    data_T const *dataref;

    InputDataGetter(data_T const *d) : dataref{d} {
        #pragma HLS INLINE
    }
    data_T const &get(unsigned iv, unsigned ix) const {
        #pragma HLS INLINE
        unsigned const ivx = iv * CONFIG_T::n_in_features + ix;
        return dataref[ivx];
    }
};

template <class CONFIG_T, class data_T> struct SingleVertexDataGetter {
    typedef data_T data_t;

    data_T const (&dataref)[CONFIG_T::n_in_features];

    SingleVertexDataGetter(data_T const (&d)[CONFIG_T::n_in_features]) : dataref{d} {
        #pragma HLS INLINE
    }
    data_T const &get(unsigned, unsigned ix) const {
        #pragma HLS INLINE
        return dataref[ix];
    }
};

template <class CONFIG_T, class res_T> struct OutputResSetter {
    typedef res_T res_t;

    res_T *resref;

    OutputResSetter(res_T *r) : resref{r} {
        #pragma HLS INLINE
    }
    void set(unsigned iv, unsigned io, res_T const &acc) {
        #pragma HLS INLINE
        unsigned const ivo = iv * CONFIG_T::n_out_features + io;
        resref[ivo] = acc;
    }
};

template <class CONFIG_T, class res_T> struct SingleVertexResSetter {
    typedef res_T res_t;

    res_T (&resref)[CONFIG_T::n_out_features];

    SingleVertexResSetter(res_T (&r)[CONFIG_T::n_out_features]) : resref{r} {
        #pragma HLS INLINE
    }
    void set(unsigned, unsigned io, res_T const &acc) {
        #pragma HLS INLINE
        resref[io] = acc;
    }
};

template <class CONFIG_T, class data_getter_T, class arrays_local_T, class arrays_T>
inline void
compute_weights_aggregates(data_getter_T const &data_getter, unsigned iv, arrays_local_T &arrays_local, arrays_T &arrays) {
    #pragma HLS INLINE

Aggregators:
    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
        typename CONFIG_T::distance_t distance = CONFIG_T::aggregator_distance_biases[ia];

    InFeatures1:
        for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
            unsigned const iax = ia * CONFIG_T::n_in_features + ix;

            typename CONFIG_T::distance_t incr = data_getter.get(iv, ix) * CONFIG_T::aggregator_distance_weights[iax];

            distance += incr;
        }

        typename CONFIG_T::edge_weight_t edge_weight =
            garnet_utils::compute_edge_weight<typename CONFIG_T::base_t>(distance);

        arrays_local.edge_weight_mean[ia] += edge_weight;

    InFeatures2:
        for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
            unsigned const iax = ia * CONFIG_T::n_in_features + ix;

            typename data_getter_T::data_t incr = data_getter.get(iv, ix) * edge_weight;

            arrays_local.weighted_feature_mean[iax] += incr;
        }

        unsigned const iva = iv * CONFIG_T::n_aggregators + ia;
        arrays.set_weight(iva, edge_weight);
    }
}

template <class CONFIG_T, class arrays_T>
inline typename CONFIG_T::aggr_t compute_output_base_core(arrays_T const &arrays, unsigned io, unsigned ia) {
    #pragma HLS INLINE
    #pragma HLS UNROLL region

    unsigned const ioa = io * CONFIG_T::n_aggregators + ia;
    typename CONFIG_T::aggr_t aggr = arrays.edge_weight_mean[ia] * CONFIG_T::input_transform_biases[ioa];

InFeatures:
    for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
        unsigned const ioax = ioa * CONFIG_T::n_in_features + ix;
        unsigned const iax = ia * CONFIG_T::n_in_features + ix;

        aggr += arrays.weighted_feature_mean[iax] * CONFIG_T::input_transform_weights[ioax];
    }

    return aggr;
}

template <class CONFIG_T, class arrays_T>
inline void compute_output_base(
    arrays_T const &arrays, typename CONFIG_T::aggr_t output_base[CONFIG_T::n_out_features * CONFIG_T::n_aggregators]
) {
    #pragma HLS INLINE
    #pragma HLS UNROLL region

OutFeatures:
    for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
    Aggregators:
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
            unsigned const ioa = io * CONFIG_T::n_aggregators + ia;

            output_base[ioa] = compute_output_base_core<CONFIG_T>(arrays, io, ia);
        }
    }
}

template <class CONFIG_T, class arrays_T, class res_setter_T>
inline void compute_vertex_output(
    arrays_T const &arrays,
    unsigned iv,
    typename CONFIG_T::aggr_t const output_base[CONFIG_T::n_out_features * CONFIG_T::n_aggregators],
    res_setter_T &res_setter
) {
    #pragma HLS INLINE

    typename arrays_T::edge_weight_t edge_weights[CONFIG_T::n_aggregators];
    #pragma HLS ARRAY_PARTITION variable=edge_weights complete

Aggregators1:
    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
        unsigned const iva = iv * CONFIG_T::n_aggregators + ia;

        edge_weights[ia] = arrays.edge_weights[iva];
    }

OutFeatures:
    for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
        typename res_setter_T::res_t acc = CONFIG_T::output_transform_biases[io];

    Aggregators2:
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
            unsigned const ioa = io * CONFIG_T::n_aggregators + ia;

            typename res_setter_T::res_t incr = edge_weights[ia] * output_base[ioa];
            acc += incr;
        }

        res_setter.set(iv, io, acc);
    }
}

template <class CONFIG_T, class data_T, class nvtx_T, class arrays_T>
void aggregate(data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features], nvtx_T const nvtx, arrays_T &arrays) {
    InputDataGetter<CONFIG_T, data_T> data_getter(data);

    unsigned const unroll_factor = CONFIG_T::n_vertices >> CONFIG_T::log2_reuse_factor;

    Means<CONFIG_T, typename CONFIG_T::edge_weight_aggr_t> means_accum;

VerticesOuter:
    for (unsigned ivv = 0; ivv < CONFIG_T::reuse_factor; ++ivv) {
        #pragma HLS PIPELINE

        if (ivv * unroll_factor >= nvtx)
            break;

        Means<CONFIG_T, typename CONFIG_T::edge_weight_aggr_t> means_local;

    VerticesInner:
        for (unsigned ir = 0; ir < unroll_factor; ++ir) {
            unsigned iv = ivv * unroll_factor + ir;

            if (iv == nvtx)
                break;

            compute_weights_aggregates<CONFIG_T>(data_getter, iv, means_local, arrays);
        }

        means_accum.add_means_normalized(means_local);
    }

    arrays.set_means_normalized(nvtx, means_accum);
}

template <class CONFIG_T, class nvtx_T, class arrays_T, class res_T>
void distribute(nvtx_T const nvtx, arrays_T const &arrays, res_T res[CONFIG_T::n_vertices * CONFIG_T::n_out_features]) {
    OutputResSetter<CONFIG_T, res_T> res_setter(res);

    typename CONFIG_T::aggr_t output_base[CONFIG_T::n_out_features * CONFIG_T::n_aggregators];
    #pragma HLS ARRAY_PARTITION variable=output_base complete

    compute_output_base<CONFIG_T>(arrays, output_base);

    unsigned const unroll_factor = CONFIG_T::n_vertices >> CONFIG_T::log2_reuse_factor;

VerticesOuter:
    for (unsigned ivv = 0; ivv < CONFIG_T::reuse_factor; ++ivv) {
        #pragma HLS PIPELINE

        if (ivv * unroll_factor >= nvtx)
            break;

    VerticesInner:
        for (unsigned ir = 0; ir < unroll_factor; ++ir) {
            unsigned iv = ivv * unroll_factor + ir;

            if (iv == nvtx)
                break;

            compute_vertex_output<CONFIG_T>(arrays, iv, output_base, res_setter);
        }
    }
}

template <class CONFIG_T, class output_biases_T, class arrays_T, class res_T>
void set_output(
    output_biases_T const &output_transform_biases, arrays_T const &arrays, res_T res[CONFIG_T::n_out_features]
) {
    #pragma HLS PIPELINE

OutFeatures:
    for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
        res_T acc = output_transform_biases.output_biases[io];

    Aggregators:
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
            typename CONFIG_T::aggr_t aggr = compute_output_base_core<CONFIG_T>(arrays, io, ia);

            acc += arrays.edge_weight_mean[ia] * aggr;
        }

        res[io] = acc;
    }
}

template <class prev_layer_t, class current_layer_t, class nvtx_T, class prev_arrays_T, class current_arrays_T>
void distribute_aggregate(nvtx_T const nvtx, prev_arrays_T const &prev_arrays, current_arrays_T &current_arrays) {
    typedef typename prev_layer_t::output_t data_T;

    typename prev_layer_t::aggr_t prev_output_base[prev_layer_t::n_out_features * prev_layer_t::n_aggregators];
    #pragma HLS ARRAY_PARTITION variable=prev_output_base complete

    compute_output_base<prev_layer_t>(prev_arrays, prev_output_base);

    unsigned const unroll_factor = current_layer_t::n_vertices >> current_layer_t::log2_reuse_factor;

    Means<current_layer_t, typename current_layer_t::edge_weight_aggr_t> means_accum;

VerticesOuter:
    for (unsigned ivv = 0; ivv < current_layer_t::reuse_factor; ++ivv) {
        #pragma HLS PIPELINE

        if (ivv * unroll_factor >= nvtx)
            break;

        Means<current_layer_t, typename current_layer_t::edge_weight_aggr_t> means_local;

    VerticesInner:
        for (unsigned ir = 0; ir < unroll_factor; ++ir) {
            unsigned iv = ivv * unroll_factor + ir;

            if (iv == nvtx)
                break;

            data_T data[prev_layer_t::n_out_features];
            #pragma HLS ARRAY_PARTITION variable=data complete

            SingleVertexResSetter<prev_layer_t, data_T> res_setter(data);

            compute_vertex_output<prev_layer_t>(prev_arrays, iv, prev_output_base, res_setter);

            SingleVertexDataGetter<current_layer_t, data_T> data_getter(data);

            compute_weights_aggregates<current_layer_t>(data_getter, iv, means_local, current_arrays);
        }

        means_accum.add_means_normalized(means_local);
    }

    current_arrays.set_means_normalized(nvtx, means_accum);
}

template <
    class prev_layer_t,
    class current_layer_t,
    class last_layer_t,
    class nvtx_T,
    class prev_arrays_T,
    class last_arrays_T>
inline typename std::enable_if<std::is_same<current_layer_t, last_layer_t>::value>::type
sublayer(nvtx_T const nvtx, prev_arrays_T const &prev_arrays, last_arrays_T &last_arrays) {
    #pragma HLS INLINE

    distribute_aggregate<prev_layer_t, current_layer_t>(nvtx, prev_arrays, last_arrays);
}

template <
    class prev_layer_t,
    class current_layer_t,
    class last_layer_t,
    class nvtx_T,
    class prev_arrays_T,
    class last_arrays_T>
inline typename std::enable_if<not std::is_same<current_layer_t, last_layer_t>::value>::type
sublayer(nvtx_T const nvtx, prev_arrays_T const &prev_arrays, last_arrays_T &last_arrays) {
    #pragma HLS INLINE

    WeightsAndMeans<current_layer_t> current_arrays;

    distribute_aggregate<prev_layer_t, current_layer_t>(nvtx, prev_arrays, current_arrays);

    sublayer<current_layer_t, typename current_layer_t::next_layer_t, last_layer_t>(nvtx, current_arrays, last_arrays);
}
} // namespace garnet_utils

struct garnet_config {
    // Layer specs
    static const unsigned n_vertices_width = 8;
    static const unsigned n_vertices = (1 << n_vertices_width);
    static const unsigned n_in_features = 4;
    static const unsigned n_propagate = 4;
    static const unsigned n_aggregators = 4;
    static const unsigned n_out_features = 4;
    static const unsigned distance_width = 12;

    // Internal data type definitions
    typedef float input_transform_weights_t;
    typedef float input_transform_biases_t;
    typedef float output_transform_weights_t;
    typedef float output_transform_biases_t;
    typedef float aggregator_distance_weights_t;
    typedef float aggregator_distance_biases_t;

    typedef float norm_t;
    typedef float distance_t;
    typedef float edge_weight_t;
    typedef float edge_weight_aggr_t;
    typedef float aggr_t;
    typedef float output_t;

    /* static const input_transform_weights_t (&input_transform_weights)[n_out_features * n_aggregators * n_in_features]; */
    /* static const input_transform_biases_t (&input_transform_biases)[n_out_features * n_aggregators]; */
    /* static const aggregator_distance_weights_t (&aggregator_distance_weights)[n_aggregators * n_in_features]; */
    /* static const aggregator_distance_biases_t (&aggregator_distance_biases)[n_aggregators]; */
    /* static const output_transform_biases_t (&output_transform_biases)[n_out_features]; */

    enum OutputCollapse { no_collapse, collapse_mean, collapse_max };

    static const unsigned output_collapse = no_collapse;

    static const bool mean_by_nvert = false;
    static const bool is_stack = false;

    // Optimization specs
    static const unsigned reuse_factor = 64;
    static const unsigned log2_reuse_factor = 6;
};

// vertices -> vertices
template <class data_T, class nvtx_T, class res_T, typename CONFIG_T>
typename std::enable_if<CONFIG_T::output_collapse == CONFIG_T::no_collapse>::type garnet(
    data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    nvtx_T const nvtx[1],
    res_T res[CONFIG_T::n_vertices * CONFIG_T::n_out_features]
) {
    #pragma HLS DATAFLOW

    garnet_utils::WeightsAndMeans<CONFIG_T> arrays;

    garnet_utils::aggregate<CONFIG_T>(data, nvtx[0], arrays);

    garnet_utils::distribute<CONFIG_T>(nvtx[0], arrays, res);
}

// vertices -> out features
template <class data_T, class nvtx_T, class res_T, class CONFIG_T>
typename std::enable_if<CONFIG_T::output_collapse == CONFIG_T::collapse_mean>::type garnet(
    data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    nvtx_T const nvtx[1],
    res_T res[CONFIG_T::n_out_features]
) {
    #pragma HLS DATAFLOW

    garnet_utils::Means<CONFIG_T> arrays;

    garnet_utils::aggregate<CONFIG_T>(data, nvtx[0], arrays);

    garnet_utils::OutputBiasNormalizer<CONFIG_T, nvtx_T> normalize_bias(nvtx[0]);

    garnet_utils::set_output<CONFIG_T>(normalize_bias, arrays, res);
}

// vertices -> vertices
template <class data_T, class nvtx_T, class res_T, class CONFIG_T>
typename std::enable_if<CONFIG_T::output_collapse == CONFIG_T::no_collapse>::type garnet_stack(
    data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    nvtx_T const nvtx[1],
    res_T res[CONFIG_T::n_vertices * CONFIG_T::n_out_features]
) {
    #pragma HLS DATAFLOW

    typedef typename CONFIG_T::template sublayer_t<0> first_layer_t;
    unsigned const ilast = CONFIG_T::n_sublayers - 1;
    typedef typename CONFIG_T::template sublayer_t<ilast> last_layer_t;

    garnet_utils::WeightsAndMeans<first_layer_t> arrays_first;
    garnet_utils::Means<last_layer_t> arrays_last;

    garnet_utils::aggregate<first_layer_t>(data, nvtx[0], arrays_first);

    garnet_utils::sublayer<first_layer_t, typename first_layer_t::next_layer_t, last_layer_t>(
        nvtx[0], arrays_first, arrays_last
    );

    garnet_utils::distribute<last_layer_t>(nvtx[0], arrays_last, res);
}

// vertices -> out features
template <class data_T, class nvtx_T, class res_T, class CONFIG_T>
typename std::enable_if<CONFIG_T::output_collapse == CONFIG_T::collapse_mean>::type garnet_stack(
    data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    nvtx_T const nvtx[1],
    res_T res[CONFIG_T::n_out_features]
) {
    #pragma HLS DATAFLOW

    typedef typename CONFIG_T::template sublayer_t<0> first_layer_t;
    unsigned const ilast = CONFIG_T::n_sublayers - 1;
    typedef typename CONFIG_T::template sublayer_t<ilast> last_layer_t;

    garnet_utils::WeightsAndMeans<first_layer_t> arrays_first;
    garnet_utils::Means<last_layer_t> arrays_last;

    garnet_utils::aggregate<first_layer_t>(data, nvtx[0], arrays_first);

    garnet_utils::sublayer<first_layer_t, typename first_layer_t::next_layer_t, last_layer_t>(
        nvtx[0], arrays_first, arrays_last
    );

    garnet_utils::OutputBiasNormalizer<last_layer_t, nvtx_T> normalize_bias(nvtx[0]);

    garnet_utils::set_output<last_layer_t>(normalize_bias, arrays_last, res);
}

/* Reference (dumb) implementation returning (Vertices, Features) */
template <class data_T, class nvtx_T, class res_T, typename CONFIG_T>
typename std::enable_if<CONFIG_T::output_collapse == CONFIG_T::no_collapse>::type garnet_ref(
    data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    nvtx_T const nvtx[1],
    res_T res[CONFIG_T::n_vertices * CONFIG_T::n_out_features]
) {
    typename CONFIG_T::edge_weight_t edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators];
    typename CONFIG_T::aggr_t propagated_features[CONFIG_T::n_vertices * CONFIG_T::n_propagate];

    for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
        if (iv == nvtx[0])
            break;

        for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
            unsigned const ivp = iv * CONFIG_T::n_propagate + ip;

            propagated_features[ivp] = CONFIG_T::input_transform_biases[ip];

            for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
                unsigned const ivx = iv * CONFIG_T::n_in_features + ix;
                unsigned const ipx = ip * CONFIG_T::n_in_features + ix;

                propagated_features[ivp] += data[ivx] * CONFIG_T::input_transform_weights[ipx];
            }
        }

        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
            unsigned const iva = iv * CONFIG_T::n_aggregators + ia;

            typename CONFIG_T::aggr_t distance = CONFIG_T::aggregator_distance_biases[ia];

            for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
                unsigned const ivx = iv * CONFIG_T::n_in_features + ix;
                unsigned const iax = ia * CONFIG_T::n_in_features + ix;

                distance += data[ivx] * CONFIG_T::aggregator_distance_weights[iax];
            }

            edge_weights[iva] = garnet_utils::compute_edge_weight<CONFIG_T>(distance);
        }
    }

    typename CONFIG_T::aggr_t aggregated_features[CONFIG_T::n_aggregators * CONFIG_T::n_propagate];

    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
        for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
            unsigned const iap = ia * CONFIG_T::n_propagate + ip;

            aggregated_features[iap] = 0.;

            for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
                if (iv == nvtx[0])
                    break;

                unsigned const iva = iv * CONFIG_T::n_aggregators + ia;
                unsigned const ivp = iv * CONFIG_T::n_propagate + ip;

                aggregated_features[iap] += edge_weights[iva] * propagated_features[ivp];
            }
        }
    }

    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
        for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
            unsigned const iap = ia * CONFIG_T::n_propagate + ip;

            if (CONFIG_T::mean_by_nvert)
                aggregated_features[iap] /= nvtx[0];
            else {
                // Not using right shift in case aggr_t is float or double
                aggregated_features[iap] /= CONFIG_T::n_vertices;
            }
        }
    }

    for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
        if (iv == nvtx[0])
            break;

        for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
            unsigned const ivo = iv * CONFIG_T::n_out_features + io;

            typename CONFIG_T::aggr_t acc = CONFIG_T::output_transform_biases[io];

            for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
                unsigned const iva = iv * CONFIG_T::n_aggregators + ia;
                unsigned const ioa = io * CONFIG_T::n_aggregators + ia;

                typename CONFIG_T::aggr_t aggr = 0.;

                for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
                    unsigned const iap = ia * CONFIG_T::n_propagate + ip;
                    unsigned const ioap = ioa * CONFIG_T::n_propagate + ip;

                    aggr += CONFIG_T::output_transform_weights[ioap] * aggregated_features[iap];
                }

                acc += edge_weights[iva] * aggr;
            }

            res[ivo] = acc;
        }
    }
}

/* Reference (dumb) implementation returning (Features) - output averaged over vertices already */
template <class data_T, class nvtx_T, class res_T, typename CONFIG_T>
typename std::enable_if<CONFIG_T::output_collapse == CONFIG_T::collapse_mean>::type garnet_ref(
    data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    nvtx_T const nvtx[1],
    res_T res[CONFIG_T::n_out_features]
) {
    typename CONFIG_T::aggr_t vertex_res[CONFIG_T::n_vertices * CONFIG_T::n_out_features];

    garnet_ref<CONFIG_T>(data, nvtx, vertex_res);

    for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
        typename CONFIG_T::aggr_t acc = 0.;

        for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
            if (iv == nvtx[0])
                break;

            unsigned const ivo = iv * CONFIG_T::n_out_features + io;

            acc += vertex_res[ivo];
        }

        if (CONFIG_T::mean_by_nvert)
            acc /= nvtx[0];
        else {
            // Not using right shift in case aggr_t is float or double
            acc /= CONFIG_T::n_vertices;
        }

        res[io] = acc;
    }
}

} // namespace nnet

#endif
