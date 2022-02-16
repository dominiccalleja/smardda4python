&inputfiles
 vtk_input_file='SHAD.vtk',
 eqdsk_input_file='EQDSK.eqdsk'
/
&miscparameters
/
&plotselections
      plot_geoqx = .true.,
      plot_geoqvolx = .false.,
      plot_geoqm = .true.,
      plot_geofldx = .false.,
      plot_gnu = .true.,
      plot_gnusil = .false.,
       plot_eqdsk_boundary = .true.
/
&beqparameters
      beq_fldspec=3,
      beq_cenopt=4,
      beq_bdryopt=7,
      beq_xiopt=2,
      beq_nzetap=1,
      skylight_flux_limits=.true.
      skylight_centre_line=.true.
      skylight_objects=0
      skylight_debug=1
/
&skylparameters
     skylight_control=0,1,1
     number_geoml_in_bin=1.
     number_flux_boxes=60
     flux_extent_margin=0.01
     number_of_divisions=64
     set_r_extent=.true.
     r_extent_min=0.1
     r_extent_max=3.9
     skylight_lower=.true.
/
