&inputfiles
 vtk_input_file='<Shadow>',
 eqdsk_input_file='<Equid>',
/
&miscparameters
/
&plotselections
      plot_geoqx = .true.,
      plot_geoqvolx = .false.,
      plot_geoqm = .true.,
      plot_geofldx = .true.,
      plot_gnu = .true.,
      plot_gnusil = .true.,
      plot_eqdsk_boundary = .true.
/
&beqparameters
      beq_fldspec=3,
      beq_cenopt=2,
      beq_bdryopt=8,
      beq_xiopt=2,
      beq_nzetap=<Nzetp>,
      skylight_flux_limits=.false.
      skylight_centre_line=.true.
      skylight_objects=0
      skylight_debug=1
      cutout_objects=2
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
&datvtkparameters
     length_units='metres'
     description='cutout'
     line_divisions=160
     start_position=1.8,-1.66
     finish_position=2.4,-1.66
/
&datvtkparameters
     length_units='metres'
     description='cutout'
     line_divisions=160
     start_position=2.9,-1.66
     finish_position=3.3,-1.66
/
