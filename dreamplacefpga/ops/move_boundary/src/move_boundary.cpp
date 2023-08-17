/**
 * @file   move_boundary.cpp
 * @author Rachel Selina (DREAMPlaceFPGA), Yibo Lin (DREAMPlace)
 * @date   Aug 2023
 * @brief  Move out-of-bound cells back to inside placement region 
 */
#include "utility/src/torch.h"
#include "utility/src/Msg.h"

static const int INVALID = -1;

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int computeMoveBoundaryMapLauncher(
        T* x_tensor, T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* regionBox2xl, const T* regionBox2yl,
        const T* regionBox2xh, const T* regionBox2yh,
        const int* node2regionBox_map,
        const T xl, const T yl, const T xh, const T yh, 
        const int num_nodes, const int num_movable_nodes, 
        const int num_filler_nodes, const int num_threads);

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
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
        int num_filler_nodes, 
        int num_threads
        ) 
{
    CHECK_FLAT(pos); 
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);

    // Call the cuda kernel launcher
    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "computeMoveBoundaryMapLauncher", [&] {
            computeMoveBoundaryMapLauncher<scalar_t>(
                    DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t)+pos.numel()/2, 
                    DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(regionBox2xl, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(regionBox2yl, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(regionBox2xh, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(regionBox2yh, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(node2regionBox_map, int), 
                    xl, yl, xh, yh, pos.numel()/2,
                    num_movable_nodes, num_filler_nodes, 
                    num_threads);
            });

    return pos; 
}

template <typename T>
int computeMoveBoundaryMapLauncher(
        T* x_tensor, T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* regionBox2xl, const T* regionBox2yl,
        const T* regionBox2xh, const T* regionBox2yh,
        const int* node2regionBox_map,
        const T xl, const T yl, const T xh, const T yh, 
        const int num_nodes, const int num_movable_nodes, 
        const int num_filler_nodes, const int num_threads)
{
#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_nodes; ++i)
    {
        if (i < num_movable_nodes || i >= num_nodes-num_filler_nodes)
        {
            if (node2regionBox_map[i] == INVALID)
            {
                x_tensor[i] = std::max(x_tensor[i], xl); 
                x_tensor[i] = std::min(x_tensor[i], xh-node_size_x_tensor[i]); 

                y_tensor[i] = std::max(y_tensor[i], yl); 
                y_tensor[i] = std::min(y_tensor[i], yh-node_size_y_tensor[i]); 
            } else if(i < num_movable_nodes)
            {
                int regionId = node2regionBox_map[i];
                T region_xl = regionBox2xl[regionId];
                T region_yl = regionBox2yl[regionId];
                T region_xh = regionBox2xh[regionId];
                T region_yh = regionBox2yh[regionId];

                x_tensor[i] = std::max(x_tensor[i], region_xl); 
                x_tensor[i] = std::min(x_tensor[i], region_xh-node_size_x_tensor[i]); 

                y_tensor[i] = std::max(y_tensor[i], region_yl); 
                y_tensor[i] = std::min(y_tensor[i], region_yh-node_size_y_tensor[i]); 
            }
        }
    }

    return 0; 
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::move_boundary_forward, "MoveBoundary forward");
}
