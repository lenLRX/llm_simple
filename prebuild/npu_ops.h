#pragma once
#include "acl/acl.h"

#include "defs.hpp"

void npu_embedding_layer(void* output_dev,
                         void* weight_dev,
                         void* index_dev,
                         int seqlen,
                         int hidden_dim,
                         DataType dt,
                         aclrtStream& stream);

void npu_rmsnorm_layer(void* output_dev,
                         void* weight_dev,
                         void* input_dev,
                         int cur_size,
                         int hidden_dim,
                         float eps,
                         DataType dt,
                         aclrtStream& stream);

void npu_softmax_layer(void* output_dev,
                         void* input_dev,
                         int n_head_cur_size,
                         int cur_size,
                         DataType dt,
                         aclrtStream& stream);

void npu_rope_layer(void* output_q_dev,
                    void* output_k_dev,
                         void* freqs_cis_dev,
                         void* input_q_dev,
                         void* input_k_dev,
                         int start_pos,
                         int cur_size,
                         int n_heads,
                         int hidden_dim,
                         DataType dt,
                         aclrtStream& stream);

void npu_batch_matmul_layer(void* output_dev,
                         void* lhs_dev,
                         void* rhs_dev,
                         int batch,
                         int m,
                         int n,
                         int k,
                         float scale,
                         DataType dt,
                         aclrtStream& stream);

void npu_batch_matmul_trans_v_layer(void* output_dev,
                         void* lhs_dev,
                         void* rhs_dev,
                         int batch,
                         int m,
                         int n,
                         int k,
                         float scale,
                         DataType dt,
                         aclrtStream& stream);


void npu_batch_matmul_causual_layer(void* output_dev,
                         void* lhs_dev,
                         void* rhs_dev,
                         int batch,
                         int m,
                         int n,
                         int k,
                         int causual_offset,
                         float scale,
                         DataType dt,
                         aclrtStream& stream);

void npu_batch_matmul_qk_trans_causual_layer(void* output_dev,
                         void* lhs_dev,
                         void* rhs_dev,
                         int batch,
                         int m,
                         int n,
                         int k,
                         int causual_offset,
                         float scale,
                         DataType dt,
                         aclrtStream& stream);

void npu_silu_mul_layer(void* output_dev,
                         void* w1_dev,
                         void* w3_dev,
                         int total_size,
                         DataType dt,
                         aclrtStream& stream);

void npu_add_layer(void* output_dev,
                         void* lhs,
                         void* rhs,
                         int total_size,
                         DataType dt,
                         aclrtStream& stream);

void npu_matmul_layer(void* output_dev,
                         void* lhs_dev,
                         void* rhs_dev,
                         int m,
                         int n,
                         int k,
                         DataType dt,
                         aclrtStream& stream);

void npu_matmul_nz_layer(void* output_dev,
                         void* lhs_dev,
                         void* rhs_dev,
                         int m,
                         int n,
                         int k,
                         DataType dt,
                         aclrtStream& stream);

void npu_mamtul_weight_transpose_layer(void* output_dev,
                         void* input,
                         int n,
                         int k,
                         DataType dt,
                         aclrtStream& stream);