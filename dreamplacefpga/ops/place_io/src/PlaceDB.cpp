/*************************************************************************
    > File Name: PlaceDB.cpp
    > Author: Yibo Lin (DREAMPlace), Rachel Selina Rajarathnam (DREAMPlaceFPGA)
    > Mail: yibolin@utexas.edu
    > Created Time: Mon 14 Mar 2016 09:22:46 PM CDT
    > Updated: Mar 2021
 ************************************************************************/

#include "PlaceDB.h"
#include <numeric>
#include <algorithm>
#include <cmath>
#include "BookshelfWriter.h"
#include "Iterators.h"
#include "utility/src/Msg.h"
//#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

/// default constructor
PlaceDB::PlaceDB() {
  num_movable_nodes = 0;
  original_num_movable_nodes = 0;
  num_fixed_nodes = 0;
  m_numLibCell = 0;
  m_numLUT = 0;
  m_numFF = 0;
  m_numDSP = 0;
//   m_numRAM = 0;
  m_numBRAM = 0;
  m_numURAM = 0;
  num_physical_constraints = 0;
  num_region_constraint_boxes = 0;
  m_numCascadeShape = 0;
  m_numCascadeInst = 0;
}

void PlaceDB::add_bookshelf_node(std::string& name, std::string& type)
{   
    if (type.find("BUF") != std::string::npos)
    {
        fixed_node_name2id_map.insert(std::make_pair(name, fixed_node_names.size()));
        fixed_node_names.emplace_back(name);
        fixed_node_types.emplace_back(type);
        fixed_node_x.emplace_back(0.0);
        fixed_node_y.emplace_back(0.0);
        fixed_node_z.emplace_back(0);
        ++num_fixed_nodes;
    } else 
    {
        original_node_name2id_map.insert(std::make_pair(name, original_mov_node_names.size()));
        original_mov_node_names.emplace_back(name);
        original_mov_node_types.emplace_back(type);
        original_node_is_cascade.emplace_back(0);
        original_node2node_map.emplace_back(0);
        org_cascade_node_pin_offset_x.emplace_back(0.0);
        org_cascade_node_pin_offset_y.emplace_back(0.0);
        org_cascade_node_x_offset.emplace_back(0.0);
        org_cascade_node_y_offset.emplace_back(0.0);
        ++original_num_movable_nodes;
    }  
}

void PlaceDB::add_bookshelf_net(BookshelfParser::Net const& n) {
    // check the validity of nets
    // if a node has multiple pins in the net, only one is kept
    std::vector<BookshelfParser::NetPin> vNetPin = n.vNetPin;

    index_type netId(net_names.size());
    net2pincount_map.emplace_back(vNetPin.size());
    net_name2id_map.insert(std::make_pair(n.net_name, netId));
    net_names.emplace_back(n.net_name);

    std::vector<index_type> netPins;
    if (flat_net2pin_start_map.size() == 0)
    {
        flat_net2pin_start_map.emplace_back(0);
    }

    for (unsigned i = 0, ie = vNetPin.size(); i < ie; ++i) 
    {
        BookshelfParser::NetPin const& netPin = vNetPin[i];
        index_type nodeId, pinId(pin_names.size()), org_nodeId;

        pin_names.emplace_back(netPin.pin_name);
        pin2net_map.emplace_back(netId);

        string2index_map_type::iterator found = node_name2id_map.find(netPin.node_name);
        string2index_map_type::iterator cas_fnd = original_node_name2id_map.find(netPin.node_name);
        std::string nodeType;

        if (found != node_name2id_map.end())
        {
            nodeId = node_name2id_map.at(netPin.node_name);
            org_nodeId = original_node_name2id_map.at(netPin.node_name);
            pin2nodeType_map.emplace_back(node2fence_region_map[nodeId]);
            if (nodeId < m_numCascadeInst)
            {   
                pin_offset_x.emplace_back(org_cascade_node_pin_offset_x[org_nodeId]);
                pin_offset_y.emplace_back(org_cascade_node_pin_offset_y[org_nodeId]);
            } else
            {
                pin_offset_x.emplace_back(0.5*mov_node_size_x[nodeId]);
                pin_offset_y.emplace_back(0.5*mov_node_size_y[nodeId]);
            }
            nodeType = mov_node_types[nodeId];
        } else if (cas_fnd != original_node_name2id_map.end())
        {
            org_nodeId = original_node_name2id_map.at(netPin.node_name);
            nodeId = original_node2node_map[org_nodeId];
            pin2nodeType_map.emplace_back(node2fence_region_map[nodeId]);
            pin_offset_x.emplace_back(org_cascade_node_pin_offset_x[org_nodeId]);
            pin_offset_y.emplace_back(org_cascade_node_pin_offset_y[org_nodeId]);
            nodeType = mov_node_types[nodeId];
        } else
        {
            string2index_map_type::iterator fnd = fixed_node_name2id_map.find(netPin.node_name);
            if (fnd != fixed_node_name2id_map.end())
            {
                nodeId = fixed_node_name2id_map.at(netPin.node_name);
                pin2nodeType_map.emplace_back(4);
                pin_offset_x.emplace_back(0.5);
                pin_offset_y.emplace_back(0.5);
                nodeType = fixed_node_types[nodeId];
                nodeId += num_movable_nodes;
            } else
            {
                dreamplacePrint(kWARN, "Net %s connects to instance %s pin %s. However instance %s is not specified in .nodes file. FIX\n",
                        n.net_name.c_str(), netPin.node_name.c_str(), netPin.pin_name.c_str(), netPin.node_name.c_str());
            }
        }

        std::string pType("");
        LibCell const& lCell = m_vLibCell.at(m_LibCellName2Index.at(nodeType));
        int pinTypeId(lCell.pinType(netPin.pin_name));

        if (pinTypeId == -1)
        {
            dreamplacePrint(kWARN, "Net %s connects to instance %s pin %s. However pin %s is not listed in .lib as a valid pin for instance type %s. FIX\n",
                    n.net_name.c_str(), netPin.node_name.c_str(), netPin.pin_name.c_str(), netPin.pin_name.c_str(), nodeType.c_str());
        }

        switch(pinTypeId)
        {
            case 2: //CLK
                {
                    pType = "CK";
                    break;
                }
            case 3: //CTRL
                {
                    if (netPin.pin_name.find("CE") != std::string::npos)
                    {
                        pType = "CE";
                    } else
                    {
                        pType = "SR";
                        pinTypeId = 4;
                    } 
                    break;
                }
            default:
                {
                    break;
                }
        }
        pin_types.emplace_back(pType);
        pin_typeIds.emplace_back(pinTypeId);

        ++node2pincount_map[nodeId];
        pin2node_map.emplace_back(nodeId);
        node2pin_map[nodeId].emplace_back(pinId);
        if (pinTypeId == 0) //Output pin
        {
            node2outpinIdx_map[nodeId] = pinId;
        }

        netPins.emplace_back(pinId);
        flat_net2pin_map.emplace_back(pinId);
    }
    flat_net2pin_start_map.emplace_back(flat_net2pin_map.size());
    net2pin_map.emplace_back(netPins);

    // std::cout << "End of add_bookshelf_net........" << std::endl;
}
void PlaceDB::resize_sites(int xSize, int ySize)
{
    m_dieArea.set(0, 0, xSize, ySize);
    m_siteDB.resize(xSize, std::vector<index_type>(ySize, 0));
}
void PlaceDB::site_info_update(int x, int y, int val)
{
    m_siteDB[x][y] = val;
}
void PlaceDB::resize_clk_regions(int xReg, int yReg)
{
    m_clkRegX = xReg;
    m_clkRegY = yReg;
}
void PlaceDB::add_clk_region(std::string const& name, int xl, int yl, int xh, int yh, int xm, int ym)
{
    clk_region temp;
    temp.xl = xl;
    temp.yl = yl;
    temp.xh = xh;
    temp.yh = yh;
    temp.xm = xm;
    temp.ym = ym;
    m_clkRegionDB.emplace_back(temp);
    m_clkRegions.emplace_back(name);
}
void PlaceDB::add_lib_cell(std::string const& name)
{
  string2index_map_type::iterator found = m_LibCellName2Index.find(name);
  if (found == m_LibCellName2Index.end())  // Ignore if already exists
  {
    m_vLibCell.push_back(LibCell(name));
    LibCell& lCell = m_vLibCell.back();
    //lCell.setName(name);
    lCell.setId(m_vLibCell.size() - 1);
    std::pair<string2index_map_type::iterator, bool> insertRet =
        m_LibCellName2Index.insert(std::make_pair(lCell.name(), lCell.id()));
    dreamplaceAssertMsg(insertRet.second, "failed to insert libCell (%s, %d)",
                        lCell.name().c_str(), lCell.id());

    m_numLibCell = m_vLibCell.size();  // update number of libCells 
  }
  m_libCellTemp = name;
}
void PlaceDB::add_input_pin(std::string& pName)
{
    string2index_map_type::iterator found = m_LibCellName2Index.find(m_libCellTemp);

    if (found != m_LibCellName2Index.end())  
    {
        LibCell& lCell = m_vLibCell.at(m_LibCellName2Index.at(m_libCellTemp));
        lCell.addInputPin(pName);
    } else
    {
        dreamplacePrint(kWARN, "libCell not found in .lib file: %s\n",
                m_libCellTemp.c_str());
    }
}
void PlaceDB::add_output_pin(std::string& pName)
{
    string2index_map_type::iterator found = m_LibCellName2Index.find(m_libCellTemp);

    if (found != m_LibCellName2Index.end())  
    {
        LibCell& lCell = m_vLibCell.at(m_LibCellName2Index.at(m_libCellTemp));
        lCell.addOutputPin(pName);
    } else
    {
        dreamplacePrint(kWARN, "libCell not found in .lib file: %s\n",
                m_libCellTemp.c_str());
    }
}
void PlaceDB::add_clk_pin(std::string& pName)
{
    string2index_map_type::iterator found = m_LibCellName2Index.find(m_libCellTemp);

    if (found != m_LibCellName2Index.end())  
    {
        LibCell& lCell = m_vLibCell.at(m_LibCellName2Index.at(m_libCellTemp));
        lCell.addClkPin(pName);
    } else
    {
        dreamplacePrint(kWARN, "libCell not found in .lib file: %s\n",
                m_libCellTemp.c_str());
    }
}
void PlaceDB::add_ctrl_pin(std::string& pName)
{
    string2index_map_type::iterator found = m_LibCellName2Index.find(m_libCellTemp);

    if (found != m_LibCellName2Index.end())  
    {
        LibCell& lCell = m_vLibCell.at(m_LibCellName2Index.at(m_libCellTemp));
        lCell.addCtrlPin(pName);
    } else
    {
        dreamplacePrint(kWARN, "libCell not found in .lib file:.end() %s\n",
                m_libCellTemp.c_str());
    }
}
void PlaceDB::add_region_constraint(int RegionIdx, int numBoxes)
{
    std::vector<index_type> regionBoxes;
    flat_constraint2box_start.emplace_back(flat_constraint2box.size());
    ++num_physical_constraints;
    num_region_constraint_boxes += numBoxes;

    std::vector<index_type> temp;
    constraint2node_map.emplace_back(temp);
    
}
void PlaceDB::add_region_box(int xl, int yl, int xh, int yh)
{
    region_box2xl.emplace_back(xl);
    region_box2yl.emplace_back(yl);
    // low close and high open, [xl, xh) x [yl, yh)
    region_box2xh.emplace_back(xh);
    region_box2yh.emplace_back(yh);

    index_type boxId(region_box2xl.size() - 1);
    flat_constraint2box.emplace_back(boxId);

}
void PlaceDB::add_instance_to_region(std::string const& instName, int regionIdx)
{
    index_type nodeId;
    string2index_map_type::iterator found = node_name2id_map.find(instName);

    if (found != node_name2id_map.end())
    {
        nodeId = node_name2id_map.at(instName);
    }
    
    constraint2node_map[regionIdx].emplace_back(nodeId);
    
}
void PlaceDB::add_cascade_shape(std::string const& name, int numRows, int numCols)
{   
    cascade_shape_name2id_map.insert(std::make_pair(name, cascade_shape_names.size()));
    cascade_shape_names.emplace_back(name);
    cascade_shape_heights.emplace_back(numRows);
    cascade_shape_widths.emplace_back(numCols);
    
    ++m_numCascadeShape;
    m_cascadeShapeTemp = name;
    cascade_shape2macro_type.emplace_back(" ");

}
void PlaceDB::add_cascade_shape_single_col(std::string macroType)
{
    //DBG
    //std::cout << "Add single col cascade shape " << macroType << " for " << m_cascadeShapeTemp << std::endl;
    //DBG
    string2index_map_type::iterator found = cascade_shape_name2id_map.find(m_cascadeShapeTemp);
    if (found != cascade_shape_name2id_map.end())
    {
        cascade_shape2macro_type.at(cascade_shape_name2id_map.at(m_cascadeShapeTemp)) = macroType;
    }

}
void PlaceDB::add_cascade_shape_double_col(std::string macroType)
{
    string2index_map_type::iterator found = cascade_shape_name2id_map.find(m_cascadeShapeTemp);
    if (found != cascade_shape_name2id_map.end())
    {
        cascade_shape2macro_type.at(cascade_shape_name2id_map.at(m_cascadeShapeTemp)) = macroType;
    }
}
void PlaceDB::add_cascade_instance_to_shape(std::string const& shapeName, std::string const& instName)
{
    string2index_map_type::iterator found = cascade_shape_name2id_map.find(shapeName);
    index_type shapeId;
    if (found != cascade_shape_name2id_map.end())
    {
        shapeId = cascade_shape_name2id_map.at(shapeName);
    }else{
        dreamplacePrint(kWARN, "Cascade shape not found in .cascade_shapes file: %s\n",
                shapeName.c_str());
        return;
    }
    
    shapeIdTemp = shapeId;
    instNameTemp = instName;

    std::string type = cascade_shape2macro_type[shapeId];
    if (type.find("DSP") != std::string::npos)
    {
        m_cascade_nodeSizeXTemp = 1.0;
        m_cascade_nodeSizeYTemp = 2.5;
    } else if (type.find("RAMB") != std::string::npos)
    {
        m_cascade_nodeSizeXTemp = 1.0;
        m_cascade_nodeSizeYTemp = 5.0;
    }

    num_cascade_nodesTemp = 0;

}
void PlaceDB::add_node_to_cascade_inst(std::string const& nodeName)
{
    string2index_map_type::iterator found = original_node_name2id_map.find(nodeName);

    index_type org_nodeId;
    if (found != original_node_name2id_map.end())
    {
        org_nodeId = original_node_name2id_map.at(nodeName);
    }else{
        return;
    }

    index_type cascade_inst_id = m_numCascadeInst;

    if (num_cascade_nodesTemp == 0)
    {
        cascade_inst2org_start_node.emplace_back(org_nodeId);
        std::vector<index_type> temp;
        cascade_inst2org_node_map.emplace_back(temp);
    }

    cascade_inst2org_node_map[cascade_inst_id].emplace_back(org_nodeId);
    original_node_is_cascade[org_nodeId] = 1;
    org_cascade_node_pin_offset_x[org_nodeId] = 0.5*m_cascade_nodeSizeXTemp;
    org_cascade_node_pin_offset_y[org_nodeId] = (num_cascade_nodesTemp+0.5)*m_cascade_nodeSizeYTemp;
    org_cascade_node_x_offset[org_nodeId] = 0.0;
    org_cascade_node_y_offset[org_nodeId] = num_cascade_nodesTemp*m_cascade_nodeSizeYTemp;

    ++num_cascade_nodesTemp;
}

void PlaceDB::end_of_cascade_inst()
{
    if (num_cascade_nodesTemp != 0)
    {
        cascade_inst2shape.emplace_back(shapeIdTemp);
        cascade_inst_names.emplace_back(instNameTemp);
        ++m_numCascadeInst;
    }
    else{
        dreamplacePrint(kWARN, "Cascade instance %s has no nodes in this design\n",
                instNameTemp.c_str());
    }
}

void PlaceDB::update_nodes(){
    // before parsing .nets file, update 

    // Add cascaded instances first, easier debugging
    for (int i = 0; i < m_numCascadeInst; ++i)
    {
        index_type org_start_nodeId;
        org_start_nodeId = cascade_inst2org_start_node[i];
        std::string name = original_mov_node_names[org_start_nodeId];
        std::string type = original_mov_node_types[org_start_nodeId];

        // std::cout << "Cascade instance " << name << " " << type << std::endl;

        for (int j = 0; j < cascade_inst2org_node_map[i].size(); ++j)
        {
            original_node2node_map[cascade_inst2org_node_map[i][j]] = mov_node_names.size();

            // std::cout << "Cascade node original id " << cascade_inst2org_node_map[i][j] << "new id " << mov_node_names.size() << std::endl;
        }

        if (type.find("DSP") != std::string::npos)
        {
            node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
            mov_node_names.emplace_back(name);
            mov_node_types.emplace_back(type);
            node2fence_region_map.emplace_back(2);
            mov_node_size_x.push_back(1.0*cascade_shape_widths[cascade_inst2shape[i]]);
            mov_node_size_y.push_back(2.5*cascade_shape_heights[cascade_inst2shape[i]]);
            mov_node_x.emplace_back(0.0);
            mov_node_y.emplace_back(0.0);
            mov_node_z.emplace_back(0);
            lut_type.emplace_back(0);
            m_numDSP += 1;
            ++num_movable_nodes;
        } else if (type.find("RAMB") != std::string::npos)
        {
            node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
            mov_node_names.emplace_back(name);
            mov_node_types.emplace_back(type);
            node2fence_region_map.emplace_back(3);
            mov_node_size_x.push_back(1.0*cascade_shape_widths[cascade_inst2shape[i]]);
            mov_node_size_y.push_back(5.0*cascade_shape_heights[cascade_inst2shape[i]]);
            mov_node_x.emplace_back(0.0);
            mov_node_y.emplace_back(0.0);
            mov_node_z.emplace_back(0);
            lut_type.emplace_back(0);
            m_numBRAM += 1;
            ++num_movable_nodes;
        } else
        {
            dreamplacePrint(kWARN, "Unknown cascade macro type component found: %s, %s\n",
                    name.c_str(), type.c_str());
        }
        std::vector<index_type> temp;
        node2pin_map.emplace_back(temp);
        node2outpinIdx_map.emplace_back(0);
        node2pincount_map.emplace_back(0);
    }
    
    double sqrt0p0625(std::sqrt(0.0625)), sqrt0p125(std::sqrt(0.125));

    // std::cout << "Original movable nodes num" << original_num_movable_nodes << std::endl;
    // std::cout << "size of original node_names map " << original_mov_node_names.size() << std::endl;

    for (int i = 0; i < original_num_movable_nodes; ++i)
    {   
        std::string name = original_mov_node_names[i];
        std::string type = original_mov_node_types[i];
        
        if (original_node_is_cascade[i] == 0)
        {   
            // std::cout << "Original node " << name << " " << type << std::endl;
            //Updated approach
            if (limbo::iequals(type, "FDRE"))
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                original_node2node_map[i] = mov_node_names.size();
                // std::cout << "FDRE node " << name << " " << type << " " << "new id " << mov_node_names.size() << std::endl;
                flop_indices.emplace_back(mov_node_names.size());
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(1);
                mov_node_size_x.push_back(sqrt0p0625);
                mov_node_size_y.push_back(sqrt0p0625);
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(0);
                lut_type.emplace_back(0);
                m_numFF += 1;
                ++num_movable_nodes;
            }
            else if (limbo::iequals(type, "FDSE"))
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                original_node2node_map[i] = mov_node_names.size();
                flop_indices.emplace_back(mov_node_names.size());
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(1);
                mov_node_size_x.push_back(sqrt0p0625);
                mov_node_size_y.push_back(sqrt0p0625);
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(0);
                lut_type.emplace_back(0);
                m_numFF += 1;
                ++num_movable_nodes;
            }
            else if (limbo::iequals(type, "LUT0"))
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                original_node2node_map[i] = mov_node_names.size();
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(0);
                mov_node_size_x.push_back(sqrt0p0625);
                mov_node_size_y.push_back(sqrt0p0625);
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(0);
                lut_type.emplace_back(0);
                m_numLUT += 1;
                ++num_movable_nodes;
            }
            else if (limbo::iequals(type, "LUT1"))
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                original_node2node_map[i] = mov_node_names.size();
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(0);
                mov_node_size_x.push_back(sqrt0p0625);
                mov_node_size_y.push_back(sqrt0p0625);
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(0);
                lut_type.emplace_back(0);
                m_numLUT += 1;
                ++num_movable_nodes;
            }
            else if (limbo::iequals(type, "LUT2"))
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                original_node2node_map[i] = mov_node_names.size();
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(0);
                mov_node_size_x.push_back(sqrt0p0625);
                mov_node_size_y.push_back(sqrt0p0625);
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(0);
                lut_type.emplace_back(0);
                m_numLUT += 1;
                ++num_movable_nodes;
            }
            else if (limbo::iequals(type, "LUT3"))
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                original_node2node_map[i] = mov_node_names.size();
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(0);
                mov_node_size_x.push_back(sqrt0p0625);
                mov_node_size_y.push_back(sqrt0p0625);
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(0);
                lut_type.emplace_back(2);
                m_numLUT += 1;
                ++num_movable_nodes;
            }
            else if (limbo::iequals(type, "LUT4"))
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                original_node2node_map[i] = mov_node_names.size();
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(0);
                mov_node_size_x.push_back(sqrt0p125);
                mov_node_size_y.push_back(sqrt0p125);
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(0);
                lut_type.emplace_back(3);
                m_numLUT += 1;
                ++num_movable_nodes;
            }
            else if (limbo::iequals(type, "LUT5"))
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                original_node2node_map[i] = mov_node_names.size();
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(0);
                mov_node_size_x.push_back(sqrt0p125);
                mov_node_size_y.push_back(sqrt0p125);
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(0);
                lut_type.emplace_back(4);
                m_numLUT += 1;
                ++num_movable_nodes;
            }
            else if (limbo::iequals(type, "LUT6"))
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                original_node2node_map[i] = mov_node_names.size();
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(0);
                mov_node_size_x.push_back(sqrt0p125);
                mov_node_size_y.push_back(sqrt0p125);
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(0);
                lut_type.emplace_back(5);
                m_numLUT += 1;
                ++num_movable_nodes;
            }
            else if (limbo::iequals(type, "LUT6_2"))
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                original_node2node_map[i] = mov_node_names.size();
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(0);
                mov_node_size_x.push_back(sqrt0p125);
                mov_node_size_y.push_back(sqrt0p125);
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(0);
                lut_type.emplace_back(5); //Treating same as LUT6
                m_numLUT += 1;
                ++num_movable_nodes;
            }
            else if (type.find("DSP") != std::string::npos)
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                original_node2node_map[i] = mov_node_names.size();
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(2);
                mov_node_size_x.push_back(1.0);
                mov_node_size_y.push_back(2.5);
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(0);
                lut_type.emplace_back(0);
                m_numDSP += 1;
                ++num_movable_nodes;
            }
            else if (type.find("RAMB") != std::string::npos)
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                original_node2node_map[i] = mov_node_names.size();
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(3);
                mov_node_size_x.push_back(1.0);
                mov_node_size_y.push_back(5.0);
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(0);
                lut_type.emplace_back(0);
                m_numBRAM += 1;
                ++num_movable_nodes;
            }
            else if (type.find("URAM") != std::string::npos)
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                original_node2node_map[i] = mov_node_names.size();
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(4);
                mov_node_size_x.push_back(1.0);
                mov_node_size_y.push_back(15.0);  // 15.0? 
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(0);
                lut_type.emplace_back(0);
                m_numURAM += 1;
                ++num_movable_nodes; 
            }
            else
            {
                dreamplacePrint(kWARN, "Unknown type component found in the movable nodes: %s, %s\n",
                        name.c_str(), type.c_str());
            }

            std::vector<index_type> temp;
            node2pin_map.emplace_back(temp);
            node2outpinIdx_map.emplace_back(0);
            node2pincount_map.emplace_back(0);

        } 
    }

    for (int i = 0; i < num_fixed_nodes; ++i)
    {
        original_node2node_map.emplace_back(i+num_movable_nodes);
        std::vector<index_type> temp;
        node2pin_map.emplace_back(temp);
        node2outpinIdx_map.emplace_back(0);
        node2pincount_map.emplace_back(0);
    }
}

void PlaceDB::set_bookshelf_node_pos(std::string const& name, double x, double y, int z)
{
    string2index_map_type::iterator found = fixed_node_name2id_map.find(name);
    //bool fixed(true);

    if (found != fixed_node_name2id_map.end())
    {
        fixed_node_x.at(fixed_node_name2id_map.at(name)) = x;
        fixed_node_y.at(fixed_node_name2id_map.at(name)) = y;
        fixed_node_z.at(fixed_node_name2id_map.at(name)) = z;
    } else
    {
        //string2index_map_type::iterator fnd = mov_node_name2id_map.find(name);
        mov_node_x.at(node_name2id_map.at(name)) = x;
        mov_node_y.at(node_name2id_map.at(name)) = y;
        mov_node_z.at(node_name2id_map.at(name)) = z;
    }

}

void PlaceDB::add_macro(std::string const& name) {
    original_macro_nodes.emplace_back(original_node_name2id_map.at(name));
}

void PlaceDB::set_bookshelf_design(std::string& name) {
  m_designName.swap(name);
}
void PlaceDB::bookshelf_end() {
    //  // parsing bookshelf format finishes
    //  // now it is necessary to init data that is not set in bookshelf
    //Flatten node2pin
    flat_node2pin_map.reserve(pin_names.size());
    flat_node2pin_start_map.emplace_back(0);
    for (const auto& sub : node2pin_map)
    {
        flat_node2pin_map.insert(flat_node2pin_map.end(), sub.begin(), sub.end());
        flat_node2pin_start_map.emplace_back(flat_node2pin_map.size());
    }

    for (auto& el : fixed_node_name2id_map)
    {
        el.second += num_movable_nodes;
    }

    node_name2id_map.insert(fixed_node_name2id_map.begin(), fixed_node_name2id_map.end());

    for (auto& el : fixed_node_name2id_map)
    {
        el.second -= num_movable_nodes;
        el.second += original_num_movable_nodes;
        org_cascade_node_pin_offset_x.emplace_back(0.0);
        org_cascade_node_pin_offset_y.emplace_back(0.0);
        org_cascade_node_x_offset.emplace_back(0.0);
        org_cascade_node_y_offset.emplace_back(0.0);
    }
    
    original_node_name2id_map.insert(fixed_node_name2id_map.begin(), fixed_node_name2id_map.end());
    
    
    flat_constraint2node_start.emplace_back(0);
    for (const auto& sub : constraint2node_map)
    {
        flat_constraint2node.insert(flat_constraint2node.end(), sub.begin(), sub.end());
        flat_constraint2node_start.emplace_back(flat_constraint2node.size());
    }

    flat_constraint2box_start.emplace_back(flat_constraint2box.size());
}

bool PlaceDB::write(std::string const& filename) const {

  return write(filename, NULL, NULL);
}

bool PlaceDB::write(std::string const& filename,
                    float const* x,
                    float const* y,
                    PlaceDB::index_type const* z) const {
  return BookShelfWriter(*this).write(filename, x, y, z);
}

DREAMPLACE_END_NAMESPACE

