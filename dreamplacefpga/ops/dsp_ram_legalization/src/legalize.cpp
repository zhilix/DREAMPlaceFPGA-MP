/**
 * @file   dsp_ram_legalization.cpp
 * @author Rachel Selina Rajarathnam (DREAMPlaceFPGA)
 * @date   Oct 2020
 * @brief  Legalize DSP/RAM instances at the end of Global Placement.
 */
#include <omp.h>
#include <chrono>
#include <limits>
#include <vector>
#include <sstream>
#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "utility/src/utils.h"
#include "utility/src/torch.h"
// Lemon for min cost flow
#include "lemon/list_graph.h"
#include "lemon/network_simplex.h"

DREAMPLACE_BEGIN_NAMESPACE

static const int INVALID = -1;

// legalize Cascade shape
template <typename T>
int legalizeCascadeInstsLauncher(
        const T* pos_x,
        const T* pos_y,
        const T* site_xy,
        const T* regionBox2xl,
        const T* regionBox2yl,
        const T* regionBox2xh,
        const T* regionBox2yh,
        const T* resource_size_x,
        const T* resource_size_y,
        const int* cascade_sites,
        const int* spiral_accessor,
        const int* site_types,
        const int* node2regionBox_map,
        const int* cascade_insts_index,
        const int* cascade_sites_cumsum,
        const T xl,
        const T yl,
        const T xh,
        const T yh,
        const int site_type_id,
        const int spiralBegin,
        const int spiralEnd,
        const int num_cascade_insts,
        const int num_sites_x,
        const int num_sites_y,
        T* movVal,
        T* outLocX,
        T* outLocY)
{
    int maxRad = DREAMPLACE_STD_NAMESPACE::max(num_sites_x, num_sites_y);
    T stepSize = resource_size_y[site_type_id];

    int scaled_y_grids = DREAMPLACE_STD_NAMESPACE::ceil(num_sites_y/stepSize);
    //Keep track of sites available
    std::vector<unsigned> occupied_sites(num_sites_x*scaled_y_grids, 0);
    //Keep track of sites considered with a 1-unit distance-based spiral accessor
    std::vector<unsigned> considered_sites(num_sites_x*scaled_y_grids, 0);

    std::vector<int> sorted_instIds(num_cascade_insts);
    std::iota(sorted_instIds.begin(),sorted_instIds.end(),0); //Initializing

    //Sort insts based on Slices occupied
    std::sort(sorted_instIds.begin(),sorted_instIds.end(), [&](int i,int j){return cascade_sites[i]>cascade_sites[j];} );

    for (int i = 0; i < num_cascade_insts; ++i)
    {
        const int idx = sorted_instIds[i];
        const int instId = cascade_insts_index[idx];
        int siteSpread = cascade_sites[idx];
        dreamplaceAssertMsg(siteSpread, "Cascade shape of size 0 encountered - CHECK");

        T initX = pos_x[instId];
        T initY = pos_y[instId];

        ////DBG
        //std::cout << "Consider inst: " << instId << " that spans " << siteSpread << " sites at (" << initX << ", " << initY << ")" << std::endl;
        ////DBG

        int beg(spiralBegin), end(spiralEnd);
        T bestX = INVALID;
        T bestY = INVALID;
        T bestScore = 10000000;

        T instLimit_xl(xl), instLimit_yl(yl), instLimit_xh(xh), instLimit_yh(yh);

        //Update limits of instance if region constraint exists
        if (node2regionBox_map[instId] != INVALID)
        {
            int regionId = node2regionBox_map[instId];
            instLimit_xl = regionBox2xl[regionId];
            instLimit_yl = regionBox2yl[regionId];
            instLimit_xh = regionBox2xh[regionId];
            instLimit_yh = regionBox2yh[regionId];
            ////DBG
            //std::cout << "Updated limits of inst: " << instId << " in region " << regionId << " to (" 
            //          << instLimit_xl << ", " << instLimit_yl << "," << instLimit_xh 
            //          << ", " << instLimit_yh << ")" << std::endl;
            ////DBG
        }

        for (int sId = beg; sId < end; ++sId)
        {
            int xVal = initX + spiral_accessor[2*sId]; 
            T yVal = DREAMPLACE_STD_NAMESPACE::round(initY + spiral_accessor[2*sId+1]);
            yVal = DREAMPLACE_STD_NAMESPACE::round(yVal/stepSize)*stepSize;
            int siteId = xVal * num_sites_y + yVal;
            //Keep track of considered sites
            int cSiteId = xVal*scaled_y_grids + DREAMPLACE_STD_NAMESPACE::round(yVal/stepSize);

            //Check within bounds
            if (xVal < instLimit_xl || xVal + resource_size_x[site_type_id] > instLimit_xh ||
                    yVal < instLimit_yl || yVal + siteSpread*stepSize > instLimit_yh ||
                    site_types[siteId] != site_type_id || considered_sites[cSiteId] == 1)
            {
                continue;
            }
            considered_sites[cSiteId] = 1;

            ////DBG
            //std::cout << "For inst: " << instId << " consider site: (" << xVal << ", " << yVal << ") of type: " << site_types[siteId] << " in region: " << site_type_id  << " for spiral accessor (" << spiral_accessor[2*sId] << ", " << spiral_accessor[2*sId+1] << std::endl;
            ////DBG


            T startY = DREAMPLACE_STD_NAMESPACE::round((yVal + (siteSpread-1)*stepSize)/stepSize)*stepSize;
            ////DBG
            //std::cout << "For inst: " << instId << " consider y location from " << startY << " to  " << yVal << " that comprises of " << siteSpread << " sites" << std::endl;
            ////DBG

            char space_available = 1;
            for (T yId = yVal; yId <= startY; yId += stepSize)
            {
                int siteMap = xVal*scaled_y_grids + DREAMPLACE_STD_NAMESPACE::round(yId/stepSize);
                if (occupied_sites[siteMap] != 0)
                {
                    space_available = 0;
                }
            }

            if (space_available == 1)
            {
                if (bestScore == 10000000)
                {
                    int r = DREAMPLACE_STD_NAMESPACE::abs(spiral_accessor[2*sId]) +
                        DREAMPLACE_STD_NAMESPACE::abs(spiral_accessor[2*sId+1]); 
                    r += 2;
                    int nwR = DREAMPLACE_STD_NAMESPACE::min(maxRad, r);
                    end = nwR ? 2 * (nwR + 1) * nwR + 1 : 1;
                }

                T dist_score = DREAMPLACE_STD_NAMESPACE::abs(pos_x[instId] - site_xy[siteId*2]) + 
                    DREAMPLACE_STD_NAMESPACE::abs(pos_y[instId] - site_xy[siteId*2+1]);
                if (dist_score < bestScore)
                {
                    bestX = xVal;
                    bestY = yVal;
                    bestScore = dist_score;
                    ////DBG
                    //std::cout << "Inst " << instId << " has best site as (" << bestX
                    //    << ", " << bestY << ")" << std::endl;
                    ////DBG

                }
            }
        }

        if (bestX != INVALID && bestY != INVALID)
        {
            movVal[0] =  DREAMPLACE_STD_NAMESPACE::max(bestScore, movVal[0]);
            movVal[1] += bestScore;

            int start_shape = cascade_sites_cumsum[idx] - siteSpread;
            T shape_maxY = DREAMPLACE_STD_NAMESPACE::round((bestY + siteSpread*stepSize)/stepSize)*stepSize;
            T shape_minY = bestY;

            ////DBG
            //std::cout << "Inst " << instId << " that spans " << cascade_sites[idx] << " sites is legalized at (" << bestX
            //    << ", " << bestY << ")" << std::endl;
            ////DBG

            int ij = start_shape;

            for (T yId = shape_minY; yId < shape_maxY; yId += stepSize)
            {
                int bSiteId = bestX * scaled_y_grids + DREAMPLACE_STD_NAMESPACE::round(yId/stepSize);
                outLocX[ij] = bestX;
                outLocY[ij] = yId;
                occupied_sites[bSiteId] = 1;
                ++ij;
            }
        }
        ////DBG
        //std::cout << "For inst: " << instId << " the best location is (" << bestX << ", " << bestY << std::endl;
        ////DBG
    }

    return 0;
}

//Min-cost flow
void legalize(
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> const& locX,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> const& locY,
    int const num_nodes, int const num_sites,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> const& sites,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> const& precond,
    double const &lg_max_dist_init, double const &lg_max_dist_incr,
    double const &lg_flow_cost_scale, pybind11::list &movVal, pybind11::list &out)
{
    typedef lemon::ListDigraph graphType;
    graphType graph; 
    graphType::ArcMap<double> capLo(graph);
    graphType::ArcMap<double> capHi(graph);
    graphType::ArcMap<double> cost(graph);
    std::vector<graphType::Node> lNodes, rNodes;
    std::vector<graphType::Arc> lArcs, rArcs, mArcs;
    std::vector<std::pair<int, int>> mArcPairs;

    //Source and target Nodes
    graphType::Node s = graph.addNode(), t = graph.addNode();

    //Add left nodes (blocks) and arcs between source node and left nodes
    for (int i = 0; i < num_nodes; ++i)
    {
        lNodes.emplace_back(graph.addNode());
        lArcs.emplace_back(graph.addArc(s, lNodes.back()));
        cost[lArcs.back()] = 0.0;
        capLo[lArcs.back()] = 0.0;
        capHi[lArcs.back()] = 1.0;
    }

    //Add right nodes (sites) and arc between right nodes and target node
    for (int j=0; j < num_sites; ++j)
    {
        rNodes.emplace_back(graph.addNode());
        rArcs.emplace_back(graph.addArc(rNodes.back(), t));
        cost[rArcs.back()] = 0.0;
        capLo[rArcs.back()] = 0.0;
        capHi[rArcs.back()] = 1.0;
    }

    //To improve efficiency, we do not run matching for complete bipartite graph but incrementally add arcs when needed
    double distMin = 0.0;
    double distMax = lg_max_dist_init;

    while (true)
    {
        //Generate arcs between left (blocks) and right (sites) nodes, pruning based on distance
        for (int blk = 0; blk < num_nodes; ++blk)
        {
            for (int st = 0; st < num_sites; ++st)
            {
                double dist = std::abs(locX.at(blk) - sites.at(st*2)) + std::abs(locY.at(blk) - sites.at(st*2+1));
                if (dist >= distMin && dist < distMax)
                {
                    mArcs.emplace_back(graph.addArc(lNodes[blk], rNodes[st]));
                    mArcPairs.emplace_back(blk, st);
                    double mArcCost = dist * precond.at(blk) * lg_flow_cost_scale;
                    cost[mArcs.back()] = mArcCost;
                    capLo[mArcs.back()] = 0.0;
                    capHi[mArcs.back()] = 1.0;
                }
            }
        }

        //Run min-cost flow
        lemon::NetworkSimplex<graphType, double> mcf(graph);
        mcf.stSupply(s, t, num_nodes);
        mcf.lowerMap(capLo).upperMap(capHi).costMap(cost);
        mcf.run();

        //A feasible solution must have flow size equal to the no of blocks
        //If not, we need to increase the max distance constraint
        double flowSize = 0.0;
        for (const auto &arc : rArcs)
        {
            flowSize += mcf.flow(arc);
        }
        if (flowSize != num_nodes)
        {
            //Increase searching range
            distMin = distMax;
            distMax += lg_max_dist_incr;
            continue;
        }

        double maxMov = 0;
        double avgMov = 0;
        //If the execution hits here, we found a feasible solution
        for (int i = 0; i < mArcs.size(); ++i)
        {
            if (mcf.flow(mArcs[i]))
            {
                const auto &p = mArcPairs[i];
                double mov = std::abs(locX.at(p.first) - sites.at(p.second*2)) + std::abs(locY.at(p.first) - sites.at(p.second*2+1));
                avgMov += mov;
                maxMov = std::max(maxMov, mov);
                out[p.first] = sites.at(p.second*2);
                out[num_nodes+p.first] = sites.at(p.second*2+1);
            }
        }
        if (num_nodes)
        {
            avgMov /= num_nodes;
        }
        movVal[0] = maxMov;
        movVal[1] = avgMov;
        return;
    }
}

//Legalize Cascade shapes
void legalizeCascadeInsts(
        at::Tensor pos,
        at::Tensor site_xy,
        at::Tensor regionBox2xl,
        at::Tensor regionBox2yl,
        at::Tensor regionBox2xh,
        at::Tensor regionBox2yh,
        at::Tensor resource_size_x,
        at::Tensor resource_size_y,
        at::Tensor cascade_sites,
        at::Tensor spiral_accessor,
        at::Tensor site_types,
        at::Tensor node2regionBox_map,
        at::Tensor cascade_insts_index,
        at::Tensor cascade_sites_cumsum,
        double xl,
        double yl,
        double xh,
        double yh,
        int site_type_id,
        int spiralBegin,
        int spiralEnd,
        int num_cascade_insts,
        int num_sites_x,
        int num_sites_y,
        at::Tensor movVal,
        at::Tensor outLocX,
        at::Tensor outLocY)
{
    CHECK_FLAT(pos);
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);

    CHECK_FLAT(site_xy);
    CHECK_CONTIGUOUS(site_xy);

    CHECK_FLAT(regionBox2xl);
    CHECK_CONTIGUOUS(regionBox2xl);
    CHECK_FLAT(regionBox2yl);
    CHECK_CONTIGUOUS(regionBox2yl);
    CHECK_FLAT(regionBox2xh);
    CHECK_CONTIGUOUS(regionBox2xh);
    CHECK_FLAT(regionBox2yh);
    CHECK_CONTIGUOUS(regionBox2yh);

    CHECK_FLAT(resource_size_x);
    CHECK_CONTIGUOUS(resource_size_x);
    CHECK_FLAT(resource_size_y);
    CHECK_CONTIGUOUS(resource_size_y);

    CHECK_FLAT(cascade_sites);
    CHECK_CONTIGUOUS(cascade_sites);

    CHECK_FLAT(spiral_accessor);
    CHECK_CONTIGUOUS(spiral_accessor);

    CHECK_FLAT(site_types);
    CHECK_CONTIGUOUS(site_types);

    CHECK_FLAT(node2regionBox_map);
    CHECK_CONTIGUOUS(node2regionBox_map);

    CHECK_FLAT(cascade_insts_index);
    CHECK_CONTIGUOUS(cascade_insts_index);

    CHECK_FLAT(cascade_sites_cumsum);
    CHECK_CONTIGUOUS(cascade_sites_cumsum);

    int numNodes = pos.numel() / 2;

    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "legalizeCascadeInstsLauncher", [&] {
            legalizeCascadeInstsLauncher<scalar_t>(
                    DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + numNodes,
                    DREAMPLACE_TENSOR_DATA_PTR(site_xy, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(regionBox2xl, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(regionBox2yl, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(regionBox2xh, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(regionBox2yh, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(resource_size_x, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(resource_size_y, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(cascade_sites, int),
                    DREAMPLACE_TENSOR_DATA_PTR(spiral_accessor, int),
                    DREAMPLACE_TENSOR_DATA_PTR(site_types, int),
                    DREAMPLACE_TENSOR_DATA_PTR(node2regionBox_map, int),
                    DREAMPLACE_TENSOR_DATA_PTR(cascade_insts_index, int),
                    DREAMPLACE_TENSOR_DATA_PTR(cascade_sites_cumsum, int),
                    xl, yl, xh, yh, site_type_id, spiralBegin,
                    spiralEnd, num_cascade_insts,
                    num_sites_x, num_sites_y,
                    DREAMPLACE_TENSOR_DATA_PTR(movVal, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(outLocX, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(outLocY, scalar_t));
    });
    //std::cout << "Completed legalizeCascadeInsts" << std::endl;
}


DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("legalizeCascadeInsts", &DREAMPLACE_NAMESPACE::legalizeCascadeInsts, "Legalize DSP & RAM cascade shapes");
    m.def("legalize", &DREAMPLACE_NAMESPACE::legalize, "Legalize DSP & RAM instances");
}
