#ifndef NNET_EMBED_STREAM_H_
#define NNET_EMBED_STREAM_H_

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void embedding(
    stream<data_T> &data,
    stream<res_T> &res,
    const typename CONFIG_T::embeddings_t embeddings[CONFIG_T::vocab_size * CONFIG_T::n_out]
) {
    data_T in_data = data.read();

InputSequence:
    #pragma ii CONFIG_T::reuse_factor
    for (int j = 0; j < data_T::size; j++) {

        res_T res_pack;

    DenseEmbedding:
        #pragma unroll
        for (int i = 0; i < CONFIG_T::n_out; i++) {
            res_pack[i] = embeddings[in_data[j] * CONFIG_T::n_out + i];
        }

        res.write(res_pack);
    }
}

} // namespace nnet

#endif
