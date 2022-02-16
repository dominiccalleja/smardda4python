&inputfiles
geoq_input_file='<run_id>RES_geoq.out',
hds_input_file='<run_id>HDS_hds.hds',
vtk_input_file='<run_id>SHAD_geoqm.vtk', ! shadowing geometry
vtkres_input_file='<run_id>RES_geoqm.vtk', ! results geometry
/
&miscparameters
/
&plotselections
      plot_powx = .true.,
      plot_flinx = .true.,
      plot_flinm = .true.,
      plot_flinends = .true.,
/
&powcalparameters
      refine_level=2,
      shadow_control=1,
      calculation_type='global',
      skylight=.false.
      termination_planes = .true.,
      more_profiles = .true.
      number_of_tracks = <n_elements>,
      element_numbers = <element_numbers>
/
&termplaneparameters
      termplane_intersection=2
      termplane_direction=0
      termplane_position=0
/
&edgprofparameters
      decay_length=<decay_length>,
      power_loss=1E7,
      diffusion_length=<diffusion_length>,
      profile_formula='eich'
/
&odesparameters
      initial_dt=1.e-5,
      abs_error=1.e-5,
      rel_error=1.e-5,
      termination_parameters(1)=3.,
      termination_parameters(2)=0.1,
      max_numsteps=40000,
      max_zeta=80.,
/
