/**
 * @file   move_boundary_cuda.cpp
 * @author Rachel Selina (DREAMPlaceFPGA), Yibo Lin (DREAMPlace)
 * @date   Aug 2023
 * @brief  Move out-of-bound cells back to inside placement region 
 */
#include "utility/src/torch.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int computeMoveBoundaryMapCudaLauncher(
        T* x_tensor, T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* regionBox2xl, const T* regionBox2yl,
        const T* regionBox2xh, const T* regionBox2yh,
        const int* node2regionBox_map,
        const T xl, const T yl, const T xh, const T yh, 
        const int num_nodes, 
        const int num_movable_nodes, 
        const int num_filler_nodes
        );

#define CHECK_FLAT(x) AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on GPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

at::Tensor move_boundary_forward(
        at::Tensor pos,
        at::Tensor node_size_x,
        at::Tensor node_size_y,
        at::Tensor regionBox2xl,
        at::Tensor regionBox2yl,
        at::Tensor regionBox2xh,
        at::Tensor regionBox2yh,
        at::Tensor node2regionBox_map,
        double xl, 
        double yl, 
        double xh, 
        double yh, 
        int num_movable_nodes, 
        int num_filler_nodes
        )
{
    CHECK_FLAT(pos); 
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);

    // Call the cuda kernel launcher
    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "computeMoveBoundaryMapCudaLauncher", [&] {
            computeMoveBoundaryMapCudaLauncher<scalar_t>(
                    DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t)+pos.numel()/2, 
                    DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t), 
                    DREAMPLACE_TENSOR_DATA_PTR(regionBox2xl, scalar_t), 
                    DREAMPLACE_TENSOR_DATA_PTR(regionBox2yl, scalar_t), 
                    DREAMPLACE_TENSOR_DATA_PTR(regionBox2xh, scalar_t), 
                    DREAMPLACE_TENSOR_DATA_PTR(regionBox2yh, scalar_t), 
                    DREAMPLACE_TENSOR_DATA_PTR(node2regionBox_map, int), 
                    xl, yl, xh, yh, 
                    pos.numel()/2, 
                    num_movable_nodes, 
                    num_filler_nodes
                    );
            });

    return pos; 
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::move_boundary_forward, "MoveBoundary forward (CUDA)");
}
