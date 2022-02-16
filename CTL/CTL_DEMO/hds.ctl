&inputfiles
 vtk_input_file='<Shadow>_geoqm.vtk',
/
&hdsgenparameters
 geometrical_type=2,
 limit_geobj_in_bin=10000,
 margin_type=2,
/
&btreeparameters
 btree_sizel=40000000,
 tree_type=3, ! multi-octree
 tree_ttalg=2, ! use nxyz
 tree_nxyz=4,4,1,
/
&positionparameters
position_transform=1,
/
&plotselections
      plot_hdsm = .true.,
      plot_hdsq = .true.,
      plot_geobjq = .true.,
      plot_geoptq = .true.,
/
