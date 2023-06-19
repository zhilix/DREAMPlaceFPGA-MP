/*************************************************************************
    > File Name: PlaceDB.h
    > Author: Yibo Lin (DREAMPlace), Rachel Selina Rajarathnam (DREAMPlaceFPGA)
    > Mail: yibolin@utexas.edu
    > Created Time: Mon 14 Mar 2016 09:22:46 PM CDT
    > Updated: Mar 2021
 ************************************************************************/

#ifndef DREAMPLACE_PLACEDB_H
#define DREAMPLACE_PLACEDB_H

#include <limbo/parsers/bookshelf/bison/BookshelfDriver.h> // bookshelf parser 
#include <limbo/string/String.h>

#include "Node.h"
#include "Net.h"
#include "Pin.h"
#include "LibCell.h"
#include "Params.h"

DREAMPLACE_BEGIN_NAMESPACE

class PlaceDB;

//Introduce new struct for clk region information
struct clk_region
{
    int xl;
    int yl;
    int xm;
    int ym;
    int xh;
    int yh;
};

class PlaceDB : public BookshelfParser::BookshelfDataBase
{
    public:
        typedef Object::coordinate_type coordinate_type;
        typedef coordinate_traits<coordinate_type>::manhattan_distance_type manhattan_distance_type;
        typedef coordinate_traits<coordinate_type>::index_type index_type;
        typedef coordinate_traits<coordinate_type>::area_type area_type;
        typedef hashspace::unordered_map<std::string, index_type> string2index_map_type;
        typedef Box<coordinate_type> diearea_type;

        /// default constructor
        PlaceDB(); 

        /// destructor
        virtual ~PlaceDB() {}

        /// member functions 
        /// data access

        std::vector<std::string> const& movNodeNames() const {return mov_node_names;}
        std::vector<std::string>& movNodeNames() {return mov_node_names;}
        std::string const& movNodeName(index_type id) const {return mov_node_names.at(id);}
        std::string& movNodeName(index_type id) {return mov_node_names.at(id);}

        std::vector<std::string> const& movNodeTypes() const {return mov_node_types;}
        std::vector<std::string>& movNodeTypes() {return mov_node_types;}
        std::string const& movNodeType(index_type id) const {return mov_node_types.at(id);}
        std::string& movNodeType(index_type id) {return mov_node_types.at(id);}

        std::vector<double> const& movNodeXLocs() const {return mov_node_x;}
        std::vector<double>& movNodeXLocs() {return mov_node_x;}
        double const& movNodeX(index_type id) const {return mov_node_x.at(id);}
        double& movNodeX(index_type id) {return mov_node_x.at(id);}

        std::vector<double> const& movNodeYLocs() const {return mov_node_y;}
        std::vector<double>& movNodeYLocs() {return mov_node_y;}
        double const& movNodeY(index_type id) const {return mov_node_y.at(id);}
        double& movNodeY(index_type id) {return mov_node_y.at(id);}

        std::vector<index_type> const& movNodeZLocs() const {return mov_node_z;}
        std::vector<index_type>& movNodeZLocs() {return mov_node_z;}
        index_type const& movNodeZ(index_type id) const {return mov_node_z.at(id);}
        index_type& movNodeZ(index_type id) {return mov_node_z.at(id);}

        std::vector<double> const& movNodeXSizes() const {return mov_node_size_x;}
        std::vector<double>& movNodeXSizes() {return mov_node_size_x;}
        double const& movNodeXSize(index_type id) const {return mov_node_size_x.at(id);}
        double& movNodeXSize(index_type id) {return mov_node_size_x.at(id);}

        std::vector<double> const& movNodeYSizes() const {return mov_node_size_y;}
        std::vector<double>& movNodeYSizes() {return mov_node_size_y;}
        double const& movNodeYSize(index_type id) const {return mov_node_size_y.at(id);}
        double& movNodeYSize(index_type id) {return mov_node_size_y.at(id);}

        std::vector<std::string> const& fixedNodeNames() const {return fixed_node_names;}
        std::vector<std::string>& fixedNodeNames() {return fixed_node_names;}
        std::string const& fixedNodeName(index_type id) const {return fixed_node_names.at(id);}
        std::string& fixedNodeName(index_type id) {return fixed_node_names.at(id);}

        std::vector<std::string> const& fixedNodeTypes() const {return fixed_node_types;}
        std::vector<std::string>& fixedNodeTypes() {return fixed_node_types;}
        std::string const& fixedNodeType(index_type id) const {return fixed_node_types.at(id);}
        std::string& fixedNodeType(index_type id) {return fixed_node_types.at(id);}

        std::vector<double> const& fixedNodeXLocs() const {return fixed_node_x;}
        std::vector<double>& fixedNodeXLocs() {return fixed_node_x;}
        double const& fixedNodeX(index_type id) const {return fixed_node_x.at(id);}
        double& fixedNodeX(index_type id) {return fixed_node_x.at(id);}

        std::vector<double> const& fixedNodeYLocs() const {return fixed_node_y;}
        std::vector<double>& fixedNodeYLocs() {return fixed_node_y;}
        double const& fixedNodeY(index_type id) const {return fixed_node_y.at(id);}
        double& fixedNodeY(index_type id) {return fixed_node_y.at(id);}

        std::vector<index_type> const& fixedNodeZLocs() const {return fixed_node_z;}
        std::vector<index_type>& fixedNodeZLocs() {return fixed_node_z;}
        index_type const& fixedNodeZ(index_type id) const {return fixed_node_z.at(id);}
        index_type& fixedNodeZ(index_type id) {return fixed_node_z.at(id);}

        std::vector<index_type> const& node2FenceRegionMap() const {return node2fence_region_map;}
        std::vector<index_type>& node2FenceRegionMap() {return node2fence_region_map;}
        index_type const& nodeFenceRegion(index_type id) const {return node2fence_region_map.at(id);}
        index_type& nodeFenceRegion(index_type id) {return node2fence_region_map.at(id);}

        std::vector<index_type> const& node2OutPinId() const {return node2outpinIdx_map;}
        std::vector<index_type>& node2OutPinId() {return node2outpinIdx_map;}

        std::vector<index_type> const& node2PinCount() const {return node2pincount_map;}
        std::vector<index_type>& node2PinCount() {return node2pincount_map;}
        index_type const node2PinCnt(index_type id) const {return node2pincount_map.at(id);}
        index_type node2PinCnt(index_type id) {return node2pincount_map.at(id);}

        std::vector<index_type> const& flopIndices() const {return flop_indices;}
        std::vector<index_type>& flopIndices() {return flop_indices;}

        std::vector<index_type> const& lutTypes() const {return lut_type;}
        std::vector<index_type>& lutTypes() {return lut_type;}

        std::vector<std::vector<index_type> > const& node2PinMap() const {return node2pin_map;}
        std::vector<std::vector<index_type> >& node2PinMap() {return node2pin_map;}
        index_type const& node2PinIdx(index_type xloc, index_type yloc) const {return node2pin_map.at(xloc).at(yloc);}
        index_type& node2PinIdx(index_type xloc, index_type yloc) {return node2pin_map.at(xloc).at(yloc);}

        std::vector<std::string> const& netNames() const {return net_names;}
        std::vector<std::string>& netNames() {return net_names;}
        std::string const& netName(index_type id) const {return net_names.at(id);}
        std::string& netName(index_type id) {return net_names.at(id);}

        std::size_t numNets() const {return net_names.size();}

        std::vector<index_type> const& net2PinCount() const {return net2pincount_map;}
        std::vector<index_type>& net2PinCount() {return net2pincount_map;}

        std::vector<std::vector<index_type> > const& net2PinMap() const {return net2pin_map;}
        std::vector<std::vector<index_type> >& net2PinMap() {return net2pin_map;}

        std::vector<index_type> const& flatNet2PinMap() const {return flat_net2pin_map;}
        std::vector<index_type>& flatNet2PinMap() {return flat_net2pin_map;}

        std::vector<index_type> const& flatNet2PinStartMap() const {return flat_net2pin_start_map;}
        std::vector<index_type>& flatNet2PinStartMap() {return flat_net2pin_start_map;}

        std::vector<index_type> const& flatNode2PinStartMap() const {return flat_node2pin_start_map;}
        std::vector<index_type>& flatNode2PinStartMap() {return flat_node2pin_start_map;}

        std::vector<index_type> const& flatNode2PinMap() const {return flat_node2pin_map;}
        std::vector<index_type>& flatNode2PinMap() {return flat_node2pin_map;}

        std::vector<std::string> const& pinNames() const {return pin_names;}
        std::vector<std::string>& pinNames() {return pin_names;}
        std::string const& pinName(index_type id) const {return pin_names.at(id);}
        std::string& pinName(index_type id) {return pin_names.at(id);}

        std::size_t numPins() const {return pin_names.size();}

        std::vector<index_type> const& pin2NetMap() const {return pin2net_map;}
        std::vector<index_type>& pin2NetMap() {return pin2net_map;}

        std::vector<index_type> const& pin2NodeMap() const {return pin2node_map;}
        std::vector<index_type>& pin2NodeMap() {return pin2node_map;}

        std::vector<index_type> const& pin2NodeTypeMap() const {return pin2nodeType_map;}
        std::vector<index_type>& pin2NodeTypeMap() {return pin2nodeType_map;}

        std::vector<std::string> const& pinTypes() const {return pin_types;}
        std::vector<std::string>& pinTypes() {return pin_types;}

        std::vector<index_type> const& pinTypeIds() const {return pin_typeIds;}
        std::vector<index_type>& pinTypeIds() {return pin_typeIds;}

        std::vector<double> const& pinOffsetX() const {return pin_offset_x;}
        std::vector<double>& pinOffsetX() {return pin_offset_x;}

        std::vector<double> const& pinOffsetY() const {return pin_offset_y;}
        std::vector<double>& pinOffsetY() {return pin_offset_y;}

        std::vector<LibCell> const& libCells() const {return m_vLibCell;}
        std::vector<LibCell>& libCells() {return m_vLibCell;}
        LibCell const& libCell(index_type id) const {return m_vLibCell.at(id);}
        LibCell& libCell(index_type id) {return m_vLibCell.at(id);}
        std::size_t numMacro() const {return m_vLibCell.size();}

        std::size_t siteRows() const {return m_siteDB.size();}
        std::size_t siteCols() const {return m_siteDB[0].size();}
        index_type const& siteVal(index_type xloc, index_type yloc) const {return m_siteDB.at(xloc).at(yloc);}
        index_type& siteVal(index_type xloc, index_type yloc) {return m_siteDB.at(xloc).at(yloc);}

        /// be careful to use die area because it is larger than the actual rowBbox() which is the placement area 
        /// it is safer to use rowBbox()
        diearea_type const& dieArea() const {return m_dieArea;}

        string2index_map_type const& libCellName2Index() const {return m_LibCellName2Index;}
        string2index_map_type& libCellName2Index() {return m_LibCellName2Index;}

        string2index_map_type const& nodeName2Index() const {return node_name2id_map;}
        string2index_map_type& nodeName2Index() {return node_name2id_map;}

        string2index_map_type const& netName2Index() const {return net_name2id_map;}
        string2index_map_type& netName2Index() {return net_name2id_map;}

        std::size_t numMovable() const {return mov_node_names.size();}
        std::size_t numFixed() const {return fixed_node_names.size();}
        std::size_t numLibCell() const {return m_numLibCell;}
        std::size_t numLUT() const {return m_numLUT;}
        std::size_t numFF() const {return m_numFF;}
        std::size_t numDSP() const {return m_numDSP;}
        // std::size_t numRAM() const {return m_numRAM;}
        std::size_t numBRAM() const {return m_numBRAM;}
        std::size_t numURAM() const {return m_numURAM;}
        std::size_t numPhysicalConstraints() const {return num_physical_constraints;}
        std::size_t numRegionConstraintBoxes() const {return num_region_constraint_boxes;}
        std::size_t numCascadeShape() const {return m_numCascadeShape;}
        std::size_t numCascadeInst() const {return m_numCascadeInst;}

        std::vector<double> const& regionBoxXLows() const {return region_box2xl;}
        std::vector<double>& regionBoxXLows() {return region_box2xl;}
        double const& regionBoxXl(index_type id) const {return region_box2xl.at(id);}
        double& regionBoxXl(index_type id) {return region_box2xl.at(id);}

        std::vector<double> const& regionBoxYLows() const {return region_box2yl;}
        std::vector<double>& regionBoxYLows() {return region_box2yl;}
        double const& regionBoxYl(index_type id) const {return region_box2yl.at(id);}
        double& regionBoxYl(index_type id) {return region_box2yl.at(id);}

        std::vector<double> const& regionBoxXHighs() const {return region_box2xh;}
        std::vector<double>& regionBoxXHighs() {return region_box2xh;}
        double const& regionBoxXh(index_type id) const {return region_box2xh.at(id);}
        double& regionBoxXh(index_type id) {return region_box2xh.at(id);}

        std::vector<double> const& regionBoxYHighs() const {return region_box2yh;}
        std::vector<double>& regionBoxYHighs() {return region_box2yh;}
        double const& regionBoxYh(index_type id) const {return region_box2yh.at(id);}
        double& regionBoxYh(index_type id) {return region_box2yh.at(id);}

        std::vector<index_type> const& flatConstraint2Box() const {return flat_constraint2box;}
        std::vector<index_type>& flatConstraint2Box() {return flat_constraint2box;}

        std::vector<index_type> const& flatConstraint2BoxStart() const {return flat_constraint2box_start;}
        std::vector<index_type>& flatConstraint2BoxStart() {return flat_constraint2box_start;}

        std::vector<index_type> const& flatConstraint2Node() const {return flat_constraint2node;}
        std::vector<index_type>& flatConstraint2Node() {return flat_constraint2node;}

        std::vector<index_type> const& flatConstraint2NodeStart() const {return flat_constraint2node_start;}
        std::vector<index_type>& flatConstraint2NodeStart() {return flat_constraint2node_start;}


        std::vector<std::string> const& cascadeShapeNames() const {return cascade_shape_names;}
        std::vector<std::string>& cascadeShapeNames() {return cascade_shape_names;}
        std::string const& cascadeShapeName(index_type id) const {return cascade_shape_names.at(id);}
        std::string& cascadeShapeName(index_type id) {return cascade_shape_names.at(id);}

        std::vector<double> const& cascadeShapeHeights() const {return cascade_shape_heights;}
        std::vector<double>& cascadeShapeHeights() {return cascade_shape_heights;}

        std::vector<double> const& cascadeShapeWidths() const {return cascade_shape_widths;}
        std::vector<double>& cascadeShapeWidths() {return cascade_shape_widths;}

        std::vector<std::string> const& cascadeShape2MacroType() const {return cascade_shape2macro_type;}
        std::vector<std::string>& cascadeShape2MacroType() {return cascade_shape2macro_type;}

        std::vector<std::string> const& cascadeInstNames() const {return cascade_inst_names;}
        std::vector<std::string>& cascadeInstNames() {return cascade_inst_names;}
        std::string const& cascadeInstName(index_type id) const {return cascade_inst_names.at(id);}
        std::string& cascadeInstName(index_type id) {return cascade_inst_names.at(id);}

        std::vector<index_type> const& cascadeInst2Shape() const {return cascade_inst2shape;}
        std::vector<index_type>& cascadeInst2Shape() {return cascade_inst2shape;}

        std::vector<index_type> const& flatCascadeInst2Node() const {return flat_cascade_inst2node;}
        std::vector<index_type>& flatCascadeInst2Node() {return flat_cascade_inst2node;}

        std::vector<index_type> const& flatCascadeInst2NodeStart() const {return flat_cascade_inst2node_start;}
        std::vector<index_type>& flatCascadeInst2NodeStart() {return flat_cascade_inst2node_start;}

        string2index_map_type const& cascadeInstName2Index() const {return cascade_inst_name2id_map;}
        string2index_map_type& cascadeInstName2Index() {return cascade_inst_name2id_map;}

        string2index_map_type const& cascadeShapeName2Index() const {return cascade_shape_name2id_map;}
        string2index_map_type& cascadeShapeName2Index() {return cascade_shape_name2id_map;}

        std::string designName() const {return m_designName;}

        /// \return die area information of layout 
        double xl() const {return m_dieArea.xl();}
        double yl() const {return m_dieArea.yl();}
        double xh() const {return m_dieArea.xh();}
        double yh() const {return m_dieArea.yh();}
        manhattan_distance_type width() const {return m_dieArea.width();}
        manhattan_distance_type height() const {return m_dieArea.height();}

        ///==== Bookshelf Callbacks ====
        virtual void add_bookshelf_node(std::string& name, std::string& type); //Updated for FPGA
        virtual void add_bookshelf_net(BookshelfParser::Net const& n);
        virtual void set_bookshelf_node_pos(std::string const& name, double x, double y, int z);
        virtual void resize_sites(int xSize, int ySize);
        virtual void site_info_update(int x, int y, int val);
        virtual void resize_clk_regions(int xReg, int yReg);
        virtual void add_clk_region(std::string const& name, int xl, int yl, int xh, int yh, int xm, int ym);
        virtual void add_lib_cell(std::string const& name);
        virtual void add_input_pin(std::string& pName);
        virtual void add_output_pin(std::string& pName);
        virtual void add_clk_pin(std::string& pName);
        virtual void add_ctrl_pin(std::string& pName);
        virtual void add_region_constraint(int RegionIdx, int numBoxes);
        virtual void add_region_box(int xl, int yl, int xh, int yh);
        virtual void add_instance_to_region(std::string const& instName, int regionIdx);
        virtual void add_cascade_shape(std::string const& name, int numRows, int numCols);
        virtual void add_cascade_shape_single_col(std::string macroType);
        virtual void add_cascade_shape_double_col(std::string macroType);
        virtual void add_cascade_instance_to_shape(std::string const& shapeName, std::string const& instName);
        virtual void add_node_to_cascade_inst(std::string const& nodeName);
        virtual void set_bookshelf_design(std::string& name);
        virtual void bookshelf_end(); 

        /// write placement solutions 
        virtual bool write(std::string const& filename) const;
        virtual bool write(std::string const& filename, float const* x = NULL, float const* y = NULL, index_type const* z = NULL) const;

        std::vector<std::vector<index_type> > m_siteDB; //FPGA Site Information
        std::vector<clk_region> m_clkRegionDB; //FPGA clkRegion Information
        std::vector<std::string> m_clkRegions; //FPGA clkRegion Names 
        int m_clkRegX;
        int m_clkRegY;
        std::vector<LibCell> m_vLibCell; ///< library macro for cells
        index_type m_coreSiteId; ///< id of core placement site 
        diearea_type m_dieArea; ///< die area, it can be larger than actual placement area 
        string2index_map_type m_LibCellName2Index; ///< map name of lib cell to index of m_vLibCell


        //Temp storage for libcell name considered
        std::string m_libCellTemp;

        std::size_t num_movable_nodes; ///< number of movable cells 
        std::size_t num_fixed_nodes; ///< number of fixed cells 
        std::size_t m_numLibCell; ///< number of standard cells in the library
        std::size_t m_numLUT; ///< number of LUTs in design
        std::size_t m_numFF; ///< number of FFs in design
        std::size_t m_numDSP; ///< number of DSPs in design
        // std::size_t m_numRAM; ///< number of RAMs in design
        std::size_t m_numBRAM; ///< number of BRAMs in design
        std::size_t m_numURAM; ///< number of URAMs in design
        std::size_t num_physical_constraints; ///< number of physical constraints
        std::size_t num_region_constraint_boxes; ///< number of regions
        std::size_t m_numCascadeShape; ///< number of cascade shapes 
        std::size_t m_numCascadeInst; ///< number of cascade instances

        std::string m_designName; ///< for writing def file

        //New approach to parsing
        std::vector<std::string> mov_node_names; 
        std::vector<std::string> fixed_node_names;
        std::vector<std::string> fixed_node_types;
        std::vector<std::string> mov_node_types;
        std::vector<std::string> net_names;
        std::vector<std::string> pin_names;
        std::vector<std::string> pin_types;
        std::vector<index_type > node2fence_region_map;
        std::vector<std::vector<index_type> > node2pin_map;
        std::vector<index_type> node2pincount_map;
        std::vector<index_type> net2pincount_map;
        std::vector<index_type> node2outpinIdx_map;
        std::vector<index_type> pin_typeIds;
        std::vector<index_type> pin2node_map;
        std::vector<index_type> pin2net_map;
        std::vector<index_type> pin2nodeType_map;
        std::vector<std::vector<index_type> > net2pin_map;
        std::vector<index_type> flat_net2pin_map;
        std::vector<index_type> flat_net2pin_start_map;
        std::vector<index_type> flat_node2pin_map;
        std::vector<index_type> flat_node2pin_start_map;
        std::vector<double> mov_node_size_x;
        std::vector<double> mov_node_size_y;
        std::vector<index_type> lut_type;
        std::vector<index_type> flop_indices;

        std::vector<double> mov_node_x;
        std::vector<double> mov_node_y;
        std::vector<index_type> mov_node_z;
        std::vector<double> fixed_node_x;
        std::vector<double> fixed_node_y;
        std::vector<index_type> fixed_node_z;
        std::vector<double> pin_offset_x;
        std::vector<double> pin_offset_y;

        //string2index_map_type mov_node_name2id_map;
        string2index_map_type fixed_node_name2id_map;
        string2index_map_type node_name2id_map;
        string2index_map_type net_name2id_map;

        std::vector<double> dspSiteXYs;
        std::vector<double> ramSiteXYs;

        std::vector<double> region_box2xl;
        std::vector<double> region_box2yl;
        std::vector<double> region_box2xh;
        std::vector<double> region_box2yh;
        
        std::vector<index_type> flat_constraint2box;
        std::vector<index_type> flat_constraint2box_start;
        std::vector<index_type> flat_constraint2node;
        std::vector<index_type> flat_constraint2node_start;

        std::string m_cascadeShapeTemp;
        
        std::vector<std::string> cascade_shape_names;
        std::vector<double> cascade_shape_heights;
        std::vector<double> cascade_shape_widths;
        std::vector<std::string> cascade_shape2macro_type;

        std::vector<std::string> cascade_inst_names;
        std::vector<index_type> cascade_inst2shape;
        std::vector<index_type> flat_cascade_inst2node;
        std::vector<index_type> flat_cascade_inst2node_start;

        string2index_map_type cascade_inst_name2id_map;
        string2index_map_type cascade_shape_name2id_map;

};

DREAMPLACE_END_NAMESPACE

#endif

