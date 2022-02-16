&inputfiles
geoq_input_file='<SimName>RES_geoq.out',
hds_input_file='<SimName>HDS_hds.hds',
vtk_input_file='<SimName>SHAD_geoqm.vtk', ! shadowing geometry
vtkres_input_file='<SimName>RES_geoqm.vtk', ! results geometry
/
&miscparameters
/
&plotselections
      plot_flinends = .false.,
      plot_flinm = .false.,
      plot_flinx = .false.,
      plot_powx = .true.,
/
&powcalparameters
      calculation_type='global',
      more_profiles = .true.
      refine_level=1,
      shadow_control=1,
      termination_planes = .true.,
/
&termplaneparameters
      termplane_direction=0
      termplane_intersection=2
      termplane_position=0
      termplane_condition=9.5
      termplane_condition_dir=1
/
&edgprofparameters
      decay_length=<DECL>,
      power_loss=<POW>,
      diffusion_length=<DIFL>,
      profile_formula='<PROF>'
/
&odesparameters
      abs_error=1.e-5,
      initial_dt=1.e-5,
      max_numsteps=100000,
      max_zeta=60.,
      rel_error=1.e-5,
      termination_parameters(1)=1.,
      termination_parameters(2)=0.1,
/
